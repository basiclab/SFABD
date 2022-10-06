import click
import torch.multiprocessing

from src.training import training_loop
from src.misc import CommandAwareConfig


@click.command(cls=CommandAwareConfig('config'), context_settings={'show_default': True})
@click.option('--config', default=None, type=str)
@click.option('--seed', default=25285)
# train dataset
@click.option('--TrainDataset', "TrainDataset", default='src.datasets.charades.base.Charades')
@click.option('--train_ann_file', default="./data/CharadesSTA/train.json")
# test dataset
@click.option('--TestDataset', "TestDataset", default='src.datasets.charades.base.Charades')
@click.option('--test_ann_file', default="./data/CharadesSTA/test.json")
# dataset share
@click.option('--feat_file', default="./data/CharadesSTA/vgg_rgb_features_all.hdf5")
@click.option('--feat_channel', default=4096)
@click.option('--num_init_clips', default=64)
@click.option('--num_clips', default=32)
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
@click.option('--intra/--no-intra', is_flag=True, default=False)
@click.option('--margin', default=0.4)
@click.option('--contrastive_weight', default=0.05)
# optimizer
@click.option('--base_lr', default=1e-4)
@click.option('--bert_lr', default=1e-5)
@click.option('--batch_size', default=48)
@click.option('--epochs', default=7)
@click.option('--bert_freeze_epoch', default=0)
@click.option('--clip_grad_norm', default=5.0)
# test
@click.option('--test_batch_size', default=64)
@click.option('--nms_threshold', default=0.5)
@click.option('--rec_metrics', default=[1, 5, 10], multiple=True)
@click.option('--iou_metrics', default=[0.5, 0.7], multiple=True)
# logging
@click.option('--logdir', default="./logs/test", type=str)
def main(**kwargs):
    del kwargs['config']

    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103#issuecomment-785815218
    torch.multiprocessing.set_sharing_strategy('file_system')

    # TODO: DDP
    training_loop(**kwargs, kwargs=kwargs)


if __name__ == "__main__":
    main()
