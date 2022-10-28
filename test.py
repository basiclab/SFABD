import os

import click
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_mAPs, recall_name
from src.misc import AttrDict, CommandAwareConfig, print_table, construct_class
from src.models.model import MMN
from src.utils import nms, scores2ds_to_moments


@click.command(cls=CommandAwareConfig('config'), context_settings={'show_default': True})
@click.option('--config', default=None, type=str)
@click.option('--seed', default=25285)
# test dataset
@click.option('--TestDataset', "TestDataset", default='src.datasets.charades.Charades')
@click.option('--test_ann_file', default="./data/CharadesSTA/charades_test.json")
# dataset share
@click.option('--feat_file', default="./data/CharadesSTA/vgg_rgb_features.hdf5")
@click.option('--feat_channel', default=4096)
@click.option('--num_init_clips', default=32)
@click.option('--num_clips', default=16)
# model
@click.option('--feat1d_out_channel', default=512)
@click.option('--feat1d_pool_kerenl_size', default=2)
@click.option('--feat2d_pool_counts', default=[16], multiple=True)
@click.option('--conv2d_hidden_channel', default=512)
@click.option('--conv2d_kernel_size', default=5)
@click.option('--conv2d_num_layers', default=8)
@click.option('--joint_space_size', default=256)
# test
@click.option('--test_batch_size', default=64)
@click.option('--nms_threshold', default=0.5)
@click.option('--recall_Ns', 'recall_Ns', default=[1, 5], multiple=True)
@click.option('--recall_IoUs', 'recall_IoUs', default=[0.5, 0.7], multiple=True)
# logging
@click.option('--logdir', default="./logs/test", type=str)
# scripts only
@click.option('--draw_rec', default=5)
@click.option('--draw_iou', default=0.7)
def test(**kwargs):
    torch.multiprocessing.set_sharing_strategy('file_system')
    config = AttrDict(**kwargs)
    device = torch.device('cuda:0')

    dataset = construct_class(
        config.TestDataset,
        ann_file=config.test_ann_file,
        feat_file=config.feat_file,
        num_init_clips=config.num_init_clips,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.test_batch_size // dist.get_world_size(),
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=min(torch.get_num_threads(), 8),
    )

    model = MMN(
        num_init_clips=config.num_init_clips,
        feat1d_in_channel=config.feat_channel,
        feat1d_out_channel=config.feat1d_out_channel,
        feat1d_pool_kerenl_size=config.feat1d_pool_kerenl_size,
        feat1d_pool_stride_size=config.num_init_clips // config.num_clips,
        feat2d_pool_counts=config.feat2d_pool_counts,
        conv2d_hidden_channel=config.conv2d_hidden_channel,
        conv2d_kernel_size=config.conv2d_kernel_size,
        conv2d_num_layers=config.conv2d_num_layers,
        joint_space_size=config.joint_space_size,
    ).to(device)
    ckpt = torch.load(os.path.join(config.logdir, 'best.pth'))
    model.load_state_dict(ckpt['model'])

    vis_path = os.path.join(
        config.logdir,
        'vis',
        recall_name(config.draw_rec, config.draw_iou))
    os.makedirs(vis_path, exist_ok=True)

    model.eval()
    pred_moments = []
    true_moments = []
    for batch, info in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2ds, mask2d = model(**batch)

        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = {k: v.cpu() for k, v in pred_moments_batch.items()}
        pred_moments.append(pred_moments_batch)

        batch = {key: value.cpu() for key, value in batch.items()}
        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments.append(true_moments_batch)

        # ploting batch
        # iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips).cpu()
        # for batch_idx in range(len(scores2ds)):
        #     ious = result['ious'][batch_idx]
        #     if ious[:config.draw_rec].max() < config.draw_iou:
        #         iou2d = iou2ds[batch_idx]                                       # Gt
        #         scores2d = scores2ds[batch_idx]                                 # Pred
        #         moment = batch['tgt_moments'][batch_idx]
        #         moment = (moment * config.num_clips).round().long()             # Gt
        #         nms_moments = result['nms_moments'][batch_idx]
        #         nms_moments = (nms_moments * config.num_clips).round().long()   # Pred
        #         path = os.path.join(vis_path, info['vid_sid'][batch_idx])
        #         plot_moments_on_iou2d(
        #             iou2d, scores2d, moment, nms_moments, path, mask2d)

    recall = calculate_recall(
        pred_moments, true_moments, config.recall_Ns, config.recall_IoUs)
    print_table(epoch=0, rows={'test': recall})

    mAPs = calculate_mAPs(pred_moments, true_moments)
    print_table(epoch=0, rows={'test': mAPs})


def plot_moments_on_iou2d(iou2d, scores2d, moment, nms_moments, path, mask2d):
    N, _ = iou2d.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ticks = (torch.arange(0, N, 5) + 0.5).numpy()
    ticklabels = [f"{idx:d}" for idx in range(0, N, 5)]

    # plot iou2d and nms_moments on left subplot
    annot = [["" for _ in range(N)] for _ in range(N)]
    for i, (st, ed) in enumerate(nms_moments):
        annot[st][ed - 1] = f"{i+1:d}"
    sns.heatmap(
        ax=ax1,
        data=iou2d.numpy(),
        annot=annot,
        mask=~mask2d.numpy(),
        vmin=0, vmax=1, cmap="plasma", fmt="s", linewidths=0.5, square=True,
        annot_kws={"ha": "center", "va": "center_baseline"})
    ax1.set_title("Groundtruth IoU and Predicted Moments")

    # plot scores2d and groundtruth moment on right subplot
    annot = [["" for _ in range(N)] for _ in range(N)]
    annot[moment[0]][moment[1] - 1] = "x"
    sns.heatmap(
        ax=ax2,
        data=scores2d.numpy(),
        annot=annot,
        mask=~mask2d.numpy(),
        vmin=0, vmax=1, cmap="plasma", fmt="s", linewidths=0.5, square=True,
        annot_kws={"ha": "center", "va": "center_baseline"})
    ax2.set_title("Scores and Groundtruth Moment")

    for ax in [ax1, ax2]:
        # xlabel and xticks on top
        ax.set_facecolor("lightgray")
        ax.set_xlabel(r"End Time$\rightarrow$", loc='left', fontsize=10)
        ax.set_ylabel(r"$\leftarrow$Start Time", loc='top', fontsize=10)
        ax.set_xticks(ticks, ticklabels)
        ax.set_yticks(ticks, ticklabels, rotation='horizontal')
        ax.tick_params(labelbottom=False, labeltop=True, bottom=False, top=True)
        ax.xaxis.set_label_position('top')

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    test()


# +----------+-------------+-------------+-------------+-------------+
# | Epoch  0 | R@1,IoU=0.5 | R@1,IoU=0.7 | R@5,IoU=0.5 | R@5,IoU=0.7 |
# +----------+-------------+-------------+-------------+-------------+
# |   test   |    46.263   |    27.392   |    81.640   |    57.876   |
# +----------+-------------+-------------+-------------+-------------+
# +----------+---------+----------+----------+
# | Epoch  0 | avg_mAP | mAP@0.50 | mAP@0.75 |
# +----------+---------+----------+----------+
# |   test   |  34.439 |  61.021  |  32.890  |
# +----------+---------+----------+----------+
