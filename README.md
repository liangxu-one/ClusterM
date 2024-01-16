# ClusterM
1. [Download](https://pan.baidu.com/s/1NT3Og0NQBGL4Kfca7Rc52w)数据集, 评测文件, 预训练模型以及java包, 密码fs2x  
2. 训练命令: CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 train.py > log/train.log 2>&1 &  
3. 测试命令: CUDA_VISIBLE_DEVICES=0 nohup python -u eval.py > log/eval.log 2>&1 &