#!/bin/bash
#SBATCH -J inference_job           # 作业名为 inference_job
#SBATCH -o slurm/try.out           # 屏幕上的输出文件重定向到 inference.out
#SBATCH -p DGX                     # 作业提交的分区为 HGX
#SBATCH --qos=lv0b                 # 作业使用的 QoS 为 lv0b
#SBATCH -N 1                       # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1        # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=16         # 单任务使用的 CPU 核心数为 16
#SBATCH -t 8:00:00                 # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:4               # 单个节点使用 2 块 GPU 卡

# 激活 Anaconda 环境
source /scratch/hexinyi/anaconda3/bin/activate languagebind  # 激活 Conda 环境

# 设置 CUDA 环境变量
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1


# 检查 CUDA 驱动和库
echo "Checking CUDA setup..."
nvidia-smi

# 检查 PyTorch CUDA 支持
echo "Checking PyTorch CUDA support..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 设置环境变量
export CACHE_DIR="/share/sail/hexinyi/LanguageBind-new/pretrained_model/CLIP-ViT-L-14-DataComp"
export TRAIN_DATA="/share/sail/hexinyi/LanguageBind-new/jsonfold/obj_sample_scannetrgb_obj_newdir.json"


# 执行 Python 脚本
echo "Running inference script..."
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=1 --nproc_per_node=4 \
    -m main  \
    --train-data ${TRAIN_DATA} \
    --train-num-samples 5120 \
    --clip-type "dl" --max-depth 10 \
    --do_train \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 2 \
    --lr 5e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.5 \
    --epochs 80 --batch-size 256 --accum-freq 1 --warmup 200 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume "latest" \
    --do_eval \
    --val_d_cls_data "shapenet12"
