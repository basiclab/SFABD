# find all configs in configs/
#config_file=configs/pool_charades_16x16_k5l8.yaml
#config_file=configs/pool_charades_16x16_k5l8_combined.yaml
config_file=configs/pool_charades_32x32_k5l8_combined.yaml


# the dir of the saved weight (also output dir)
#weight_dir=outputs/pool_charades_16x16_k5l8
#weight_dir=outputs/pool_charades_16x16_k5l8_combined
weight_dir=outputs/pool_charades_32x32_k5l8_combined

# select weight to evaluate
#weight_file=outputs/pool_charades_16x16_k5l8/best_charades.pth
#weight_file=outputs/pool_charades_16x16_k5l8_combined/pool_model_4e.pth
weight_file=outputs/pool_charades_32x32_k5l8_combined/pool_model_1e.pth

# test batch size
batch_size=48
# set your gpu id
gpus=0
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.2
master_port=29578

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $weight_dir TEST.BATCH_SIZE $batch_size

