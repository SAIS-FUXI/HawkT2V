export CUDA_VISIBLE_DEVICES=0

RANK=0 LOCAL_RANK=0 WORLD_SIZE=1  MASTER_ADDR='127.0.0.1' MASTER_PORT=12356 python scripts/eval/inference.py --config="configs/t2v/infer_config_3B_512.yaml"
