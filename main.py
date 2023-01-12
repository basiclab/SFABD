import os
import tempfile

import click
import torch.multiprocessing

from src.training import training_loop, training_loop_PE
from src.testing import testing_loop
from src.misc import AttrDict, CommandAwareConfig


@click.command(cls=CommandAwareConfig('config'), context_settings={'show_default': True})
@click.option('--config', default=None, type=str)
@click.option('--seed', default=25285)
@click.option('--test_only/--no-test_only', default=False)
# train dataset
@click.option('--TrainDataset', "TrainDataset", default='src.datasets.charades.Charades')
# test dataset
@click.option('--TestDataset', "TestDataset", default='src.datasets.charades.Charades')
# dataset share
@click.option('--num_init_clips', default=32)
@click.option('--num_clips', default=16)
# model
@click.option('--feat1d_out_channel', default=512)
@click.option('--feat1d_pool_kerenl_size', default=2)
@click.option('--feat2d_pool_counts', default=[16], multiple=True)
@click.option('--conv2d_hidden_channel', default=512)
@click.option('--conv2d_kernel_size', default=5)
@click.option('--conv2d_num_layers', default=8)
@click.option('--dual_space/--no-dual_space', default=False)
# joint model
@click.option('--joint_space_size', default=256)

# Probabilistic Embedding
@click.option('--num_samples', default=7)
@click.option('--kl/--no-kl', default=True) ## kl constraint

# iou loss
@click.option('--min_iou', default=0.5)
@click.option('--max_iou', default=1.0)
@click.option('--iou_weight', default=1.0)
# contrastive loss
@click.option('--tau_video', default=0.1)
@click.option('--tau_query', default=0.1)
@click.option('--neg_iou', default=0.5)
@click.option('--pos_iou', default=0.9)
@click.option('--pos_topk', default=1)
@click.option('--margin', default=0.4)
@click.option('--inter/--no-inter', default=True)
@click.option('--intra/--no-intra', default=False)
@click.option('--intra_start_epoch', default=6)
@click.option('--contrastive_weight', default=0.05)
@click.option('--cont_weight_step', default=0.01)
# optimizer
@click.option('--base_lr', default=1e-4)
@click.option('--bert_lr', default=1e-5)
@click.option('--milestones', default=[8, 13], multiple=True)
@click.option('--batch_size', default=48)
@click.option('--epochs', default=18)
@click.option('--bert_freeze_epoch', default=4)
@click.option('--only_iou_epoch', default=7)
@click.option('--grad_clip', default=5.0)
@click.option('--step_gamma', default=0.1)
# testing options
@click.option('--test_batch_size', default=64)
@click.option('--nms_threshold', default=0.5)
@click.option('--recall_Ns', 'recall_Ns', default=[1, 5, 10], multiple=True)
@click.option('--recall_IoUs', 'recall_IoUs', default=[0.5, 0.7], multiple=True)
# logging
@click.option('--logdir', default="./logs/test", type=str)
@click.option('--best_metric', default="avg_mAP")
@click.option('--save_freq', default=5)
# visualization options (test_only)
@click.option('--draw_rec', default=5)
@click.option('--draw_iou', default=0.7)
def main(**kwargs):
    config = AttrDict(**kwargs)

    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(','))
    with tempfile.TemporaryDirectory() as temp_dir:
        processes = []
        for rank in range(num_gpus):
            p = torch.multiprocessing.Process(
                target=subprocess, args=(rank, num_gpus, temp_dir, config))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


def subprocess(rank, world_size, temp_dir, config):
    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
    init_method = f'file://{init_file}'
    torch.distributed.init_process_group('nccl', init_method, rank=rank, world_size=world_size)
    print(f"Node {rank} is initialized")

    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103#issuecomment-785815218
    torch.multiprocessing.set_sharing_strategy('file_system')

    # set default device
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    if config.test_only:
        # testing
        testing_loop(config)
    else:
        # training
        training_loop(config)
        #training_loop_PE(config)


if __name__ == "__main__":
    main()
