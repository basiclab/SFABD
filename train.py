import click
import torch.multiprocessing

from mmn.training import training_loop
from mmn.misc import CommandAwareConfig


@click.command(cls=CommandAwareConfig('config'), context_settings={'show_default': True})
@click.option('--config', default=None, type=str)
@click.option('--seed', default=25285)
# train dataset
@click.option('--TrainDataset', "TrainDataset", default='mmn.datasets.MultiTargetCharades')
@click.option('--train_ann_file', default="./data/Charades_STA/v1/combined_charades_train_remove_repeat_action_videos.json")
@click.option('--train_template_file', default=None)
# test dataset
@click.option('--TestDataset', "TestDataset", default='mmn.datasets.MultiTargetCharades')
@click.option('--test_ann_file', default="./data/Charades_STA/v1/charades_multi_target.json")
# dataset share
@click.option('--vgg_feat_file', default="./data/Charades_STA/Charades_vgg_rgb.hdf5")
@click.option('--num_init_clips', default=64)
# model
@click.option('--num_clips', default=32)
@click.option('--conv1d_in_channel', default=4096)
@click.option('--conv1d_out_channel', default=512)
@click.option('--conv1d_pool_kernel_size', default=2)
@click.option('--conv1d_pool_kernel_stride', default=2)
@click.option('--conv2d_in_channel', default=512)
@click.option('--conv2d_hidden_channel', default=512)
@click.option('--conv2d_kernel_size', default=5)
@click.option('--conv2d_num_layers', default=8)
@click.option('--joint_space_size', default=256)
# iou loss
@click.option('--min_iou', default=0.5)
@click.option('--max_iou', default=1.0)
@click.option('--iou_weight', default=1.0)
# contrastive loss
@click.option('--tau_video', default=0.1)
@click.option('--tau_query', default=0.1)
@click.option('--neg_video_iou', default=0.5)
@click.option('--pos_video_topk', default=1)
@click.option('--inter/--no-inter', is_flag=True, default=True)
@click.option('--intra/--no-intra', is_flag=True, default=True)
@click.option('--margin', default=0.4)
@click.option('--contrastive_weight', default=0.1)
# optimizer
@click.option('--base_lr', default=1e-4)
@click.option('--bert_lr', default=1e-5)
@click.option('--batch_size', default=24)
@click.option('--epochs', default=7)
@click.option('--bert_freeze_epoch', default=0)
@click.option('--clip_grad_norm', default=5.0)
# test
@click.option('--test_batch_size', default=64)
@click.option('--nms_threshold', default=0.5)
@click.option('--rec_metrics', default=[1, 5, 10], multiple=True)
@click.option('--iou_metrics', default=[0.5, 0.7], multiple=True)
# logging
@click.option('--logdir', required=True, type=str)
def main(**kwargs):
    del kwargs['config']
    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103#issuecomment-785815218
    torch.multiprocessing.set_sharing_strategy('file_system')

    # TODO: DDP
    training_loop(**kwargs, kwargs=kwargs)


if __name__ == "__main__":
    main()
