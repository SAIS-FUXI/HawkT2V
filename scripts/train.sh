export CUDA_VISIBLE_DEVICES=0

MASTER_ADDR="localhost"
MASTER_PORT="6667"
NNODES=1
NODE_RANK=0
NGPUS_PER_NODE=1 #$(nvidia-smi -L | wc -l)
    


CKPT="pretrained_models/transformer/diffusion_pytorch_model.pt"
CONFIG="configs/t2v/t2v_train_v3_3dvae_video_batch_release_dev_3B_512.yaml"

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_with_img_t2v.py \
    --config=${CONFIG} \
    --checkpoint=${CKPT} 