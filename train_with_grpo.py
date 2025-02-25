import os
import copy
import random
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
from eval import eval
from config import Config
from model import CaptionModel
from read_file import build_data
from evaluation import compute_scores_cider
from module.utils import _sample

def set_seed(seed):
    random.seed(seed)  # 配置Python random库的随机种子
    np.random.seed(seed)  # 配置Numpy库的随机种子
    torch.manual_seed(seed)  # 配置torch的随机种子
    torch.cuda.manual_seed(seed)  # 配置单个GPU上的种子
    torch.cuda.manual_seed_all(seed)  # 配置所有GPU上的种子
    # # cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    # # 如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    # torch.backends.cudnn.enabled = False
    # # 将benchmark设置为False会让cudnn在有多种算法可选的情况下选择固定的一种
    # # 假如是True的话，cudnn会对多种算法进行测试，找到在你硬件上运行最快的那个算法，
    # # 然后再固定使用这个算法进行计算。
    # # 假如模型输入不会变化，比较规则，那设置成True可能会提高性能
    # # 假如模型输入会变化，那设置成True反而可能导致性能降低
    # # 不过要复现那还是设置成False吧~
    # torch.backends.cudnn.benchmark = False
    # # benchmark=False让选择的算法是固定的，然而这个算法本身可能还是non-deterministic的
    # # 所以设置deterministic=True可以让torch选择可确定的算法
    # torch.backends.cudnn.deterministic = True

def make_experience(model, img, caption_index, config, img_id, train_dict):
    bs = img.size(0)
    eos_index = torch.tensor([config.eos_token_id]).to(img.device)

    # 生成sample_nums个回答
    img = img.unsqueeze(0).repeat(config.sample_nums, 1, 1, 1, 1).reshape(-1, img.size(-3), img.size(-2), img.size(-1))
    caption_index = caption_index.unsqueeze(0).repeat(config.sample_nums, 1, 1).reshape(-1, caption_index.size(-1))

    sample_index = _sample(model, img, caption_index, config)
    if sample_index.size(1) < config.max_length:
        add_eos = torch.empty([sample_index.size(0), config.max_length - sample_index.size(1)], dtype = sample_index.dtype, device = sample_index.device).fill_(eos_index[0])
        sample_index = torch.concat([sample_index, add_eos], dim = 1)
    assert sample_index.size(1) == config.max_length

    # 计算生成时的概率
    caption_mask = config.generator_fun(sample_index.size(1)).to(img.device).unsqueeze(0).repeat(img.size(0), 1, 1)
    pred = model(img, sample_index, caption_mask)
    logits = pred.log_softmax(dim = -1)

    # 取出采样概率
    get_sample_index = torch.empty_like(sample_index).fill_(config.eos_token_id)
    get_sample_index[:, :-1] = sample_index[:, 1:]
    logits = torch.gather(logits, dim = -1, index = get_sample_index.unsqueeze(-1)).squeeze(-1)
    logits = logits * (sample_index != config.eos_token_id).to(torch.float32)

    img = img.reshape(config.sample_nums, bs, img.size(-3), img.size(-2), img.size(-1))
    sample_index = sample_index.reshape(config.sample_nums, bs, sample_index.size(-1))
    caption_mask = caption_mask.reshape(config.sample_nums, bs, caption_mask.size(-2), caption_mask.size(-1))
    logits = logits.reshape(config.sample_nums, bs, logits.size(-1))

    all_reward_score = []
    for i in range(config.sample_nums):
        sample_pred_str = config.tokenizer.batch_decode(sample_index[i].reshape(bs, -1).tolist(), skip_special_tokens=True)

        gts = {}
        sample_res = {}
        for k in range(bs):
            image_id = int(img_id[k])
            gts[image_id] = train_dict.imgid_to_sentences[image_id]
            sample_res[image_id] = [sample_pred_str[k]]

        reward_score = compute_scores_cider(gts, sample_res)[1]['CIDEr']
        reward_score = torch.tensor(reward_score).to(img.device)
        all_reward_score.append(reward_score)

    all_reward_score = torch.stack(all_reward_score, dim = 0)
    all_advantage = (all_reward_score - torch.mean(all_reward_score, dim = 0)) / torch.std(all_reward_score, dim = 0)
    all_advantage = torch.where(torch.isfinite(all_advantage), all_advantage, torch.zeros_like(all_advantage))

    img = img.permute(1, 0, 2, 3, 4)
    sample_index = sample_index.permute(1, 0, 2)
    caption_mask = caption_mask.permute(1, 0, 2, 3)
    logits = logits.permute(1, 0, 2)
    all_reward_score = all_reward_score.permute(1, 0)
    all_advantage = all_advantage.permute(1, 0)

    experience = []
    for i in range(bs):
        temp = {'img': img[i], 'sample_index': sample_index[i], 'attention_mask': caption_mask[i],
                        'log_probs': logits[i], 'reward_score': all_reward_score[i], 'advantage': all_advantage[i]}
        experience.append(temp)
    return experience

def train(config):

    dist.init_process_group('nccl')
    # 分布式训练时获取该进程的rank值, 并只让rank为0的进程进行测试与指标评估
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda:{}".format(rank) if config.use_cuda else "cpu")

    # 加载模型
    model = CaptionModel(config)
    model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.ck), map_location='cpu'))
    if rank == 0:
        print(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, [device])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 参考模型
    ref_model = copy.deepcopy(model)
    for p in ref_model.module.parameters():
        p.requires_grad = False

    if rank == 0:
        # 读取数据
        print("读取数据")

    train_dict = build_data(config)
    train_sampler = DistributedSampler(train_dict, seed = config.seed)
    train_data = DataLoader(train_dict, config.batch_size, shuffle = (train_sampler is None), 
                            sampler = train_sampler, num_workers = config.num_workers)

    if rank == 0:
        configVal = Config(TrainOrVal = 'val')
        val_dict = build_data(configVal)
        val_data = DataLoader(val_dict, configVal.batch_size, shuffle = False, num_workers = configVal.num_workers)

        configTest = Config(TrainOrVal = 'test')
        test_dict = build_data(configTest)
        test_data = DataLoader(test_dict, configTest.batch_size, shuffle = False, num_workers = configTest.num_workers)

        print("train data is: ", len(train_dict))
        print("val data is: ", len(val_dict))
        print("test data is: ", len(test_dict))
        print("读取数据结束")

    optimizer = torch.optim.AdamW(model.module.parameters(), lr = config.grpo_lr, weight_decay = config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, config.grpo_all_epoch * config.grpo_epoch * config.batch_size * len(train_data) // config.grpo_batch_size)

    # 开始训练
    for epoch in range(config.grpo_all_epoch):

        experience_list = []
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_data):
            model.eval()
            with torch.no_grad():
                img = batch[0].to(device)
                caption_index = batch[1].to(device)
                experience = make_experience(model, img, caption_index, config, batch[2], train_dict)
                experience_list.extend(experience)

            if ((i + 1) % config.grpo_step == 0) or ((i + 1) == len(train_data)):
                # 对数据进行打乱操作
                random.shuffle(experience_list)
                grpo_batch_img = torch.stack([temp['img'] for temp in experience_list], dim = 0)
                grpo_batch_sample_index = torch.stack([temp['sample_index'] for temp in experience_list], dim = 0)
                grpo_batch_caption_mask = torch.stack([temp['attention_mask'] for temp in experience_list], dim = 0)
                grpo_batch_log_probs = torch.stack([temp['log_probs'] for temp in experience_list], dim = 0)
                grpo_batch_advantage = torch.stack([temp['advantage'] for temp in experience_list], dim = 0)

                for grpo_epoch in range(config.grpo_epoch):

                    if rank == 0:
                        print(scheduler.get_last_lr())
                    j = 0
                    while j < len(experience_list):

                        model.zero_grad()

                        img = grpo_batch_img[j:min(j + config.grpo_batch_size, len(experience_list))]
                        sample_index = grpo_batch_sample_index[j:min(j + config.grpo_batch_size, len(experience_list))]
                        caption_mask = grpo_batch_caption_mask[j:min(j + config.grpo_batch_size, len(experience_list))]
                        log_probs = grpo_batch_log_probs[j:min(j + config.grpo_batch_size, len(experience_list))]
                        advantage = grpo_batch_advantage[j:min(j + config.grpo_batch_size, len(experience_list))]

                        img = img.reshape(-1, img.size(-3), img.size(-2), img.size(-1))
                        sample_index = sample_index.reshape(-1, sample_index.size(-1))
                        caption_mask = caption_mask.reshape(-1, caption_mask.size(-2), caption_mask.size(-2))
                        log_probs = log_probs.reshape(-1, log_probs.size(-1))
                        advantage = advantage.reshape(-1)

                        pred = model(img, sample_index, caption_mask)
                        logits = pred.log_softmax(dim = -1)

                        # 计算参考模型的loss
                        ref_pred = ref_model(img, sample_index, caption_mask)
                        ref_logits = ref_pred.log_softmax(dim = -1)

                        # 取出采样概率
                        get_sample_index = torch.empty_like(sample_index).fill_(config.eos_token_id)
                        get_sample_index[:, :-1] = sample_index[:, 1:]

                        logits = torch.gather(logits, dim = -1, index = get_sample_index.unsqueeze(-1)).squeeze(-1)
                        logits = logits * (sample_index != config.eos_token_id).to(torch.float32)

                        ref_logits = torch.gather(ref_logits, dim = -1, index = get_sample_index.unsqueeze(-1)).squeeze(-1)
                        ref_logits = ref_logits * (sample_index != config.eos_token_id).to(torch.float32)
                        kl_loss = torch.exp(ref_logits - logits) - (ref_logits - logits) - 1

                        # 计算模型损失
                        ratio = torch.exp(logits - log_probs)
                        advantage = advantage.unsqueeze(1)
                        grpo_loss1 = advantage * ratio
                        grpo_loss2 = advantage * torch.clamp(ratio, 1.0 - config.policy_clip_eps, 1.0 + config.policy_clip_eps)
                        padding_mask = (sample_index != config.eos_token_id).to(torch.float32)
                        loss = -torch.sum((torch.min(grpo_loss1, grpo_loss2) - config.beta * kl_loss) * padding_mask) / padding_mask.sum()

                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        if rank == 0 and j % (50 * config.grpo_batch_size) == 0:
                            print('j/batch: {}/{} | grpo_epoch/grpo_epochs: {}/{} | i/batch: {}/{} | epoch/epochs: {}/{} | loss: {}'.format(j, len(experience_list), grpo_epoch, config.grpo_epoch, i, len(train_data), epoch, config.grpo_all_epoch, loss.item()))

                        j = j + config.grpo_batch_size

                experience_list = []

            if (rank == 0) and (((i + 1) % (len(train_data) // config.grpo_save_frequency) == 0) or ((i + 1) == len(train_data))):
                torch.save(model.module.state_dict(), os.path.join(config.model_save_path, 'rl_epoch_{}_i_{}.pt'.format(epoch, i + 1)))
                print("test:", end = ' ')
                with torch.no_grad():
                    eval(configVal, model, val_data, val_dict)

    if rank == 0:
        with torch.no_grad():
            eval(configTest, model, test_data, test_dict)

if __name__ == '__main__':
    set_seed(Config().seed)
    config = Config(with_rl = True)
    train(config)