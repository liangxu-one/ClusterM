# ClusterM
1. [Download](https://pan.baidu.com/s/1NT3Og0NQBGL4Kfca7Rc52w)数据集, 评测文件, 预训练模型以及java包, 密码fs2x
2. 设置java环境: 
#set java env
export JAVA_HOME="You Java Path"
export JRE_HOME=${JAVA_HOME}/jre    
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib    
export PATH=${JAVA_HOME}/bin:$PATH
3. 训练命令: CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 train.py > log/train.log 2>&1 &  
4. 测试命令: CUDA_VISIBLE_DEVICES=0 nohup python -u eval.py > log/eval.log 2>&1 &
