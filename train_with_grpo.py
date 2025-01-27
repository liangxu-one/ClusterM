import os
import random
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
from eval import eval
from config import Config
from model import CaptionModel
from read_file import build_data
from evaluation import compute_scores_cider
from module.utils import _sample

class GRPODataset(Dataset):
    def __init__(self, img, sample_index, caption_mask, log_probs, advantage) -> None:
        super(GRPODataset, self).__init__()

        self.img = img
        self.sample_index = sample_index
        self.caption_mask = caption_mask
        self.log_probs = log_probs
        self.advantage = advantage

    def __getitem__(self, index):
        return self.img[index], self.sample_index[index], self.caption_mask[index], self.log_probs[index], self.advantage[index]

    def __len__(self):
        return len(self.img)

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
    experience = dict()
    all_reward_scocr = []
    eos_index = torch.tensor([config.eos_token_id]).to(img.device)
    for i in range(config.sample_nums):
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

        sample_pred_str = config.tokenizer.batch_decode(sample_index.reshape(img.size(0), -1).tolist(), skip_special_tokens=True)

        gts = {}
        sample_res = {}
        bs = img.size(0)
        for k in range(bs):
            image_id = int(img_id[k])
            gts[image_id] = train_dict.imgid_to_sentences[image_id]
            sample_res[image_id] = [sample_pred_str[k]]

        reward_score = compute_scores_cider(gts, sample_res)[1]['CIDEr']
        reward_score = torch.tensor(reward_score).to(img.device)

        all_reward_scocr.append(reward_score)
        temp_experience = {'img': img, 'sample_index': sample_index, 'attention_mask': caption_mask,
                        'log_probs': logits, 'reward_score': reward_score, 'advantages': 0}
        experience.setdefault(i, temp_experience)

    all_reward_scocr = torch.stack(all_reward_scocr, dim = 0)
    all_advantage = (all_reward_scocr - torch.mean(all_reward_scocr, dim = 0)) / torch.std(all_reward_scocr, dim = 0)
    all_advantage = torch.where(torch.isfinite(all_advantage), all_advantage, torch.zeros_like(all_advantage))
    for key in experience.keys():
        experience[key]['advantages'] = all_advantage[key]
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
    ref_model = CaptionModel(config)
    ref_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.ck), map_location='cpu'))
    ref_model.to(device)
    ref_model = torch.nn.parallel.DistributedDataParallel(ref_model, [device])
    ref_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ref_model)
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
        model.eval()
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_data):

            with torch.no_grad():
                img = batch[0].to(device)
                caption_index = batch[1].to(device)
                experience = make_experience(model, img, caption_index, config, batch[2], train_dict)
                experience_list.append(experience)

            if ((i + 1) % config.grpo_step == 0) or ((i + 1) == len(train_data)):
                # 将experience_list中的数据组装为一个dataset
                img_list = []
                sample_index_list = []
                caption_mask_list = []
                log_probs_list = []
                advantage_list = []

                # 每条experience由config.sample_nums * batch_size条数据构成, 先进行处理
                for experience in experience_list:
                    temp_img_list = []
                    temp_sample_index_list = []
                    temp_caption_mask_list = []
                    temp_log_probs_list = []
                    temp_advantage_list = []
                    for key in experience.keys():
                        temp_img_list.append(experience[key]['img'])
                        temp_sample_index_list.append(experience[key]['sample_index'])
                        temp_caption_mask_list.append(experience[key]['attention_mask'])
                        temp_log_probs_list.append(experience[key]['log_probs'])
                        temp_advantage_list.append(experience[key]['advantages'])

                    # 维度转换, 将一个样本的config.sample_nums条数据放在一起
                    temp_img = torch.stack(temp_img_list, dim = 0).permute(1, 0, 2, 3, 4)
                    temp_sample_index = torch.stack(temp_sample_index_list, dim = 0).permute(1, 0, 2)
                    temp_caption_mask = torch.stack(temp_caption_mask_list, dim = 0).permute(1, 0, 2, 3)
                    temp_log_probs = torch.stack(temp_log_probs_list, dim = 0).permute(1, 0, 2)
                    temp_advantage = torch.stack(temp_advantage_list, dim = 0).permute(1, 0)

                    # 加入统计列表中
                    img_list.append(temp_img)
                    sample_index_list.append(temp_sample_index)
                    caption_mask_list.append(temp_caption_mask)
                    log_probs_list.append(temp_log_probs)
                    advantage_list.append(temp_advantage)

                # 整合为一个grpo训练过程中要用的数据
                img = torch.concat(img_list, dim = 0)
                sample_index = torch.concat(sample_index_list, dim = 0)
                caption_mask = torch.concat(caption_mask_list, dim = 0)
                log_probs = torch.concat(log_probs_list, dim = 0)
                advantage = torch.concat(advantage_list, dim = 0)

                grpo_dataset = GRPODataset(img, sample_index, caption_mask, log_probs, advantage)
                grpo_dataloader = DataLoader(grpo_dataset, config.grpo_batch_size, shuffle = True)
                experience_list = []

                for grpo_epoch in range(config.grpo_epoch):

                    if rank == 0:
                        print(scheduler.get_last_lr())

                    for j, grpo_batch in enumerate(grpo_dataloader):

                        model.zero_grad()

                        img = grpo_batch[0]
                        sample_index = grpo_batch[1]
                        caption_mask = grpo_batch[2]
                        log_probs = grpo_batch[3]
                        advantage = grpo_batch[4]

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

                        if rank == 0 and j % 50 == 0:
                            print('j/batch: {}/{} | grpo_epoch/grpo_epochs: {}/{} | i/batch: {}/{} | epoch/epochs: {}/{} | loss: {}'.format(j, len(grpo_dataloader), grpo_epoch, config.grpo_epoch, i, len(train_data), epoch, config.grpo_all_epoch, loss.item()))

            if (rank == 0) and ((i + 1) % (len(train_data) // (20 // config.grpo_all_epoch)) == 0):
                torch.save(model.module.state_dict(), os.path.join(config.model_save_path, 'rl_epoch_{}_i_{}.pt'.format(epoch, (i + 1) // (len(train_data) // (20 // config.grpo_all_epoch)))))
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