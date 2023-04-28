import os
import tempfile

import click
import torch.multiprocessing

from src.training import training_loop
from src.testing import testing_loop
from src.misc import AttrDict, CommandAwareConfig


@click.command(cls=CommandAwareConfig('config'), context_settings={'show_default': True})
@click.option('--config', default=None, type=str)
@click.option('--seed', default=25285)
@click.option('--test_only/--no-test_only', default=False)
# dataset common settings
@click.option('--num_init_clips', default=32)
@click.option('--num_clips', default=16)
# datasets
@click.option('--TrainDataset', "TrainDataset", default='src.datasets.charades.CharadesSTAI3DTrain')
@click.option('--ValDataset', "ValDataset", default='src.datasets.activitynet.ActivityNetI3DVal')
@click.option('--TestDataset', "TestDataset", default='src.datasets.charades.CharadesSTAI3DTest')
@click.option('--MultiTestDataset', "MultiTestDataset", default='src.datasets.charades.CharadesSTAI3DMultiTest')
# model
@click.option('--backbone', default="src.models.resnet.ProposalConv")
@click.option('--feat1d_out_channel', default=512)
@click.option('--feat1d_pool_kernel_size', default=2)
@click.option('--feat2d_pool_counts', default=[16], multiple=True)
@click.option('--conv2d_hidden_channel', default=512)
@click.option('--conv2d_kernel_size', default=5)
@click.option('--conv2d_num_layers', default=8)
@click.option('--joint_space_size', default=256)
@click.option('--dual_space/--no-dual_space', default=False)
@click.option('--resnet', default=34)
# confidence loss
@click.option('--IoULoss', 'IoULoss', default="src.losses.iou.ScaledIoULoss")
@click.option('--min_iou', default=0.5)
@click.option('--max_iou', default=1.0)
@click.option('--alpha', default=0.25)
@click.option('--gamma', default=2.0)
@click.option('--iou_weight', default=1.0)
# bbox regression loss
@click.option('--iou_threshold', default=0.75)
@click.option('--bbox_reg_weight', default=1.0)
# contrastive loss common settings
@click.option('--neg_iou', default=0.5)
@click.option('--pos_topk', default=1)
@click.option('--top_neg_removal_percent', default=0.01)
@click.option('--contrastive_decay', default=0.1)
@click.option('--contrastive_decay_start', default=6)
# inter contrastive loss
@click.option('--InterContrastiveLoss', 'InterContrastiveLoss', default="src.losses.contrastive.InterContrastiveLoss")
@click.option('--inter_t', default=0.1, help='temperature for inter contrastive loss')
@click.option('--inter_m', default=0.3, help='margin for inter contrastive loss')
@click.option('--inter_weight', default=1.0)
# intra contrastive loss
@click.option('--IntraContrastiveLoss', 'IntraContrastiveLoss', default="src.losses.contrastive.IntraContrastiveLoss")
@click.option('--intra_t', default=0.1, help='temperature for intra contrastive loss')
@click.option('--intra_m', default=0.0, help='margin for inter contrastive loss')
@click.option('--intra_weight', default=0.1)
@click.option('--do_augmentation', default=False)
@click.option('--mixup_alpha', default=0.9)
@click.option('--aug_expand_rate', default=1.2)
# Dynamic Negative Sampling
@click.option('--inter_query_threshold', default=0.7, help='threshold for finding false negative')
@click.option('--intra_video_threshold', default=0.9, help='threshold for finding false negative')
@click.option('--fusion_ratio', default=0.5, help='linear combination ratio of inter_query_sim and intra_video_sim')
@click.option('--exponent', default=1, help='exponent of the fused neg sim distribution')
@click.option('--neg_samples_num', default=512)
@click.option('--start_dns_epoch', default=999)
@click.option('--rate_step_change', default=0.05)
# multi positive contrastive loss
@click.option('--MultiPositiveContrastiveLoss', 'MultiPositiveContrastiveLoss', default="src.losses.contrastive.MultiPositiveContrastiveLoss")
# optimizer
@click.option('--epochs', default=10)
@click.option('--batch_size', default=24)
@click.option('--base_lr', default=1e-3)
@click.option('--bert_lr', default=1e-5)
@click.option('--milestones', default=[7], multiple=True)
@click.option('--step_gamma', default=0.1)
@click.option('--bert_fire_start', default=1)
@click.option('--grad_clip', default=5.0)
# testing options
@click.option('--test_batch_size', default=48)
@click.option('--nms_threshold', default=0.5)
@click.option('--recall_Ns', 'recall_Ns', default=[1, 5], multiple=True)
@click.option('--recall_IoUs', 'recall_IoUs', default=[0.5, 0.7], multiple=True)
# logging
@click.option('--logdir', default="./logs/test", type=str)
@click.option('--best_metric', default="all/mAP@avg")
@click.option('--save_freq', default=999)
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


if __name__ == "__main__":
    main()
