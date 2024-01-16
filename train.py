import os
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

def train(config):

    dist.init_process_group('nccl')
    # 分布式训练时获取该进程的rank值, 并只让rank为0的进程进行测试与指标评估
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda:{}".format(rank) if config.use_cuda else "cpu")

    # 加载模型
    model = CaptionModel(config)
    if rank == 0:
        print(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, [device])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

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

    pre_trained_params = list(map(id, model.module.image_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in pre_trained_params, model.module.parameters())

    optimizer = torch.optim.AdamW([
        {'params':model.module.image_encoder.parameters(), 'lr':config.lr/10},
        {'params':base_params, 'lr':config.lr},],
        lr=config.lr, weight_decay=config.weight_decay)
    # optimizer = torch.optim.Adam(model.module.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.epoch * len(train_data) // 10, 
                                                config.epoch * len(train_data))

    # 开始训练
    for epoch in range(config.epoch):
        if rank == 0:
            print(scheduler.get_last_lr())

        train_sampler.set_epoch(epoch)
        model.train()
        for i, batch in enumerate(train_data):
            for j in range(len(batch)):
                batch[j] = batch[j].to(device)
            model.zero_grad()
            loss = model(*batch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            if rank == 0 and i % 100 == 0:
                print('i/batch: {}/{} | epoch/epochs: {}/{} | loss: {}'.format(i, len(train_data), epoch, config.epoch, loss.item()))

        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(config.model_save_path, 'epoch_{}.pt'.format(epoch)))
            print("test:", end = ' ')
            with torch.no_grad():
                eval(configVal, model, val_data, val_dict)

    if rank == 0:
        with torch.no_grad():
            eval(configTest, model, test_data, test_dict)

if __name__ == '__main__':
    set_seed(Config().seed)
    config = Config()
    train(config)