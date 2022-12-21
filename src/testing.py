import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.multiprocessing
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_mAPs, recall_name
from src.misc import print_table, construct_class
from src.models.model import MMN
from src.utils import nms, scores2ds_to_moments


def testing_loop(config):
    device = dist.get_device()

    # test Dataset and DataLoader
    test_dataset = construct_class(config.TestDataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=config.seed)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test_batch_size // dist.get_world_size(),
        collate_fn=test_dataset.collate_fn,
        sampler=test_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

    # model
    model_local = MMN(
        num_init_clips=config.num_init_clips,
        feat1d_in_channel=test_dataset.get_feat_dim(),
        feat1d_out_channel=config.feat1d_out_channel,
        feat1d_pool_kerenl_size=config.feat1d_pool_kerenl_size,
        feat1d_pool_stride_size=config.num_init_clips // config.num_clips,
        feat2d_pool_counts=config.feat2d_pool_counts,
        conv2d_hidden_channel=config.conv2d_hidden_channel,
        conv2d_kernel_size=config.conv2d_kernel_size,
        conv2d_num_layers=config.conv2d_num_layers,
        joint_space_size=config.joint_space_size,
        dual_space=config.dual_space,
    ).to(device)
    # load from checkpoint
    ckpt = torch.load(os.path.join(config.logdir, 'best.pth'))
    model_local.load_state_dict(ckpt['model'])
    # DDP
    model = SyncBatchNorm.convert_sync_batchnorm(model_local)
    model = DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True)

    # vis_path = os.path.join(
    #     config.logdir,
    #     'vis',
    #     recall_name(config.draw_rec, config.draw_iou))
    # os.makedirs(vis_path, exist_ok=True)

    model.eval()
    pred_submission = []
    pred_moments = []
    true_moments = []
    pbar = tqdm(test_loader, ncols=0, leave=False, desc="Inferencing")
    for batch, batch_info in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2d, mask2d = model(**batch)

        out_moments, out_scores1ds = scores2ds_to_moments(scores2d, mask2d)
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

        assert len(batch_info['qids']) == len(batch_info['sentences']) == len(batch_info['vid']) == len(batch_info['duration'])
        shift = 0
        num_proposals = iter(pred_moments_batch['num_proposals'])
        for qids, sentences, vid, duration, in zip(
            batch_info['qids'],
            batch_info['sentences'],
            batch_info['vid'],
            batch_info['duration']
        ):
            assert len(qids) == len(sentences)
            for qid, query in zip(qids, sentences):
                num_proposal = next(num_proposals)
                out_scores1d = pred_moments_batch['out_scores1ds'][shift: shift + num_proposal]
                out_moments = pred_moments_batch['out_moments'][shift: shift + num_proposal] * duration
                pred_relevant_windows = torch.cat([out_moments, out_scores1d.unsqueeze(-1)], dim=-1)
                pred_submission.append(json.dumps({
                    'qid': qid.item(),
                    'query': query,
                    'vid': vid,
                    'pred_relevant_windows': pred_relevant_windows.tolist(),
                }))
                shift += num_proposal

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

    with open(os.path.join(config.logdir, 'submission.jsonl'), 'w') as f:
        for line in pred_submission:
            f.write(line + '\n')

    recall = calculate_recall(
        pred_moments, true_moments, config.recall_Ns, config.recall_IoUs)
    mAPs = calculate_mAPs(pred_moments, true_moments)

    print_table(epoch=0, rows={'test': recall})
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