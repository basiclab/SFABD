import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import SyncBatchNorm
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_multi_recall, calculate_mAPs
from src.misc import (
    AttrDict, set_seed, construct_class, print_recall, print_mAPs, print_multi_recall
)
from src.models.main import MMN
from src.utils import (
    nms, scores2ds_to_moments, moments_to_iou2ds, iou2ds_to_iou2d)


vgg = None


def update_vgg_features(video_feats, augmented_data, mixup_alpha):
    if augmented_data is None:
        return video_feats

    global vgg
    if vgg is None:
        vgg = vgg16(weights=VGG16_Weights.DEFAULT, progress=True)
        vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
        vgg.eval()
        vgg = vgg.to(dist.get_device())

    aug_frames = augmented_data['aug_frames']
    tgt_frames = augmented_data['tgt_frames']
    aug_frames_st_ed = augmented_data['aug_frames_st_ed']
    aug_num = augmented_data['aug_num']
    assert len(video_feats) == len(aug_num)

    aug_feats_all = []
    for batch in aug_frames.split(32, dim=0):
        batch = batch.to(dist.get_device())
        with torch.no_grad():
            feats = F.normalize(vgg(batch), dim=-1).cpu()
            aug_feats_all.append(feats)
    aug_feats_all = torch.cat(aug_feats_all, dim=0)

    tgt_feats_all = []
    for batch in tgt_frames.split(32, dim=0):
        batch = batch.to(dist.get_device())
        with torch.no_grad():
            feats = F.normalize(vgg(batch), dim=-1).cpu()
            tgt_feats_all.append(feats)
    tgt_feats_all = torch.cat(tgt_feats_all, dim=0)

    feats_shift = 0
    st_ed_shift = 0
    for i, num in enumerate(aug_num):
        for st, ed in aug_frames_st_ed[st_ed_shift: st_ed_shift + num]:
            length = ed.item() - st.item()
            aug_feats = aug_feats_all[feats_shift: feats_shift + length]
            tgt_feats = tgt_feats_all[feats_shift: feats_shift + length]
            new_feats = (1 - mixup_alpha) * aug_feats + mixup_alpha * tgt_feats
            video_feats[i, st: ed] = new_feats
            feats_shift += length
        st_ed_shift += num

    return video_feats


def append_to_json_file(path, data):
    if os.path.exists(path):
        history = json.load(open(path, 'r'))
    else:
        history = []
    history.append(data)
    json.dump(history, open(path, 'w'), indent=4)


def test_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    config: AttrDict,
):
    device = dist.get_device()
    model.eval()
    pred_moments = []
    true_moments = []

    for batch, info in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2ds, mask2d = model(**batch)
        # scores2ds:   [S, N, N]
        # out_moments: [S, P, 2]
        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)
        # use different nms_thres for different query sample?
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        # ground truth
        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

    return pred_moments, true_moments


# same as test epoch but no plotting
def val_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    config: AttrDict,
):
    device = dist.get_device()
    model.eval()
    pred_moments = []
    true_moments = []

    for batch, info in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
        batch = {key: value.to(device) for key, value in batch.items()}
        # prediciton
        with torch.no_grad():
            *_, scores2ds, mask2d = model(**batch)
        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)        # out_moments: [S, P, 2]
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        # ground truth
        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

    return pred_moments, true_moments


@torch.no_grad()
def find_false_negative(
    video_feats: torch.Tensor,          # [S, C, N, N]
    sents_feats: torch.Tensor,          # [S, C]
    num_sentences: torch.Tensor,        # [B]
    num_targets: torch.Tensor,          # [S]
    iou2d: torch.Tensor,                # [S, N, N]
    iou2ds: torch.Tensor,               # [M, N, N]
    mask2d: torch.Tensor,               # [N, N]
    config: AttrDict,
    epoch: int,
    batch_info: AttrDict,
) -> torch.Tensor:                      # [S, B * P]
    S, C, N, _ = video_feats.shape
    B = num_sentences.shape[0]
    M = num_targets.sum().cpu().item()
    P = mask2d.long().sum()
    K = config.pos_topk
    device = video_feats.device
    assert iou2d.shape == (S, N, N), f"{iou2d.shape} != {(S, N, N)}"
    assert iou2ds.shape == (M, N, N), f"{iou2ds.shape} != {(M, N, N)}"

    # Choose each sample's first sentence idx for mapping S to B
    scatter_b2s = []
    count = 0
    for num_sentence in num_sentences:
        scatter_b2s.append(count)
        count = count + num_sentence.item()
    scatter_b2s = torch.tensor(scatter_b2s, device=device)          # [B]

    scatter_s2b = torch.arange(B, device=device).long()
    scatter_s2b = scatter_s2b.repeat_interleave(num_sentences)

    # moment idx -> sentence idx
    scatter_m2s = torch.arange(S, device=device).long()
    scatter_m2s = scatter_m2s.repeat_interleave(num_targets)        # [M]

    video_feats = video_feats.masked_select(mask2d).view(S, C, -1)  # [S, C, P]
    video_feats = video_feats.permute(0, 2, 1)                      # [S, P, C]
    iou2d = iou2d.masked_select(mask2d).view(S, -1)                 # [S, P]
    iou2ds = iou2ds.masked_select(mask2d).view(M, -1)               # [M, P]

    # normalize for cosine similarity
    video_feats = F.normalize(video_feats.contiguous(), dim=-1)     # [S, P, C]
    sents_feats = F.normalize(sents_feats.contiguous(), dim=-1)     # [S, C]

    # === inter video (topk proposal -> all sentences)
    topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
    topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
    allm_video_feats = video_feats[scatter_m2s]             # [M, P, C]
    topk_video_feats = allm_video_feats.gather(
        dim=1, index=topk_idxs)                             # [M, K, C]

    # pos sample sim score
    inter_query_pos = torch.mul(
        topk_video_feats,                                   # [M, K, C]
        sents_feats[scatter_m2s].unsqueeze(1)               # [M, 1, C]
    ).sum(dim=-1)                                           # [M, K]
    inter_query_pos = (inter_query_pos + 1) / 2             # [M, K]

    # need to convert video_feats from [S, P, C] to [B, P, C]
    inter_query_all = torch.mm(
        sents_feats,                                        # [S, C]
        video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
    )                                                       # [S, B * P]

    # [-1, 1] -> [0, 1]
    inter_query_sim = (inter_query_all + 1) / 2             # [S, B * P]
    assert (inter_query_sim > 0 - 1e-3).all()
    assert (inter_query_sim < 1 + 1e-3).all()

    # pos mask for [S, B * P] video proposals
    pos_mask = torch.eye(B, device=device).bool()           # [B, B]
    pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
    pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
    pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
    pos_mask = pos_mask[scatter_s2b]                        # [S, B * P]
    assert pos_mask.long().sum(dim=-1).eq(P).all()
    s2v_pos_mask = iou2d > 0.5                              # [S, P]
    local_mask = pos_mask.clone()                           # [S, B * P]
    pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
    inter_query_neg_mask = ~pos_mask                        # [S, B * P]

    num_t = 0
    false_neg_thres = []            # [S]
    false_neg_accept_rate = []      # [S]
    for sent_idx, num_target in enumerate(num_targets):
        pos_sim_record = inter_query_pos[num_t:num_t + num_target].mean(dim=-1)  # top-k positive average
        neg_sim_record = inter_query_sim[sent_idx][inter_query_neg_mask[sent_idx]]

        # false neg threshold for each sample
        if config.thres_method == "mean":
            false_neg_thres.append(pos_sim_record.mean())
        elif config.thres_method == "max":
            false_neg_thres.append(pos_sim_record.max())
        elif config.thres_method == "fixed":
            false_neg_thres.append(config.false_neg_thres)

        # compute acceptance rate for each sample
        if config.accept_rate_method == "linear2x":  # y = 2x
            accept_rate = 2 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear1.5x":  # y = 1.5x
            accept_rate = 1.5 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear1.25x":  # y = 1.25x
            accept_rate = 1.25 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear":  # y = x
            accept_rate = (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear0.5x":  # y = 0.5x
            accept_rate = 0.5 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear0.25x":  # y = 0.25x
            accept_rate = 0.25 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear0.1x":  # y = 0.1x
            accept_rate = 0.1 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear0.05x":  # y = 0.05x
            accept_rate = 0.05 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear0.01x":  # y = 0.01x
            accept_rate = 0.01 * (pos_sim_record.mean() - neg_sim_record.mean())
        elif config.accept_rate_method == "linear_baseline":
            accept_rate = epoch / config.epochs
        false_neg_accept_rate.append(accept_rate)

        num_t += num_target

    # make false neg mask
    false_neg_mask = torch.zeros_like(inter_query_sim)        # [S, B * P]
    for sent_idx, (inter_query_sim, neg_mask, neg_thres, accept_rate) in enumerate(
        zip(inter_query_sim, inter_query_neg_mask, false_neg_thres, false_neg_accept_rate)
    ):
        neg_masked_inter_query_sim = inter_query_sim.masked_select(neg_mask)
        neg_thres_mask = torch.zeros_like(neg_masked_inter_query_sim)
        neg_thres_mask[neg_masked_inter_query_sim > neg_thres] = 1
        neg_thres_mask = neg_thres_mask.bool()
        # Only accept top-x% as false neg samples
        if neg_thres_mask.sum() > 0:
            K = round(int(min(max(accept_rate, 0), 1) * neg_thres_mask.sum().item()))
            topk_idx = neg_masked_inter_query_sim[neg_thres_mask].topk(K, dim=0)[1]
            keep_mask = torch.zeros_like(neg_masked_inter_query_sim[neg_thres_mask])
            keep_mask[topk_idx] = 1
            temp_mask = torch.zeros_like(false_neg_mask[sent_idx][neg_mask.clone()])
            temp_mask[neg_thres_mask.clone()] = keep_mask
            false_neg_mask[sent_idx][neg_mask.clone()] = temp_mask
        # no neg > thres
        else:
            false_neg_mask[sent_idx][neg_mask.clone()] = torch.zeros_like(neg_masked_inter_query_sim)

    # false neg mask for contrastive loss and BCE loss
    false_neg_mask_con = false_neg_mask.bool()                          # [S, B * P]
    false_neg_mask_iou = false_neg_mask_con[local_mask].reshape(S, P)   # [S, P]

    # To record sample learning progress, please comment train_sampler.set_epoch(epoch) in training loop
    # false_neg_vid_list = []
    # false_neg_time_list = []
    # moments = mask2d.nonzero()                                          # [P, 2]
    # moments[:, 1] += 1                                                  # [P, 2]
    # moments = moments / N                                               # [P, 2]
    # record_false_neg_mask = false_neg_mask_con[record_sent_idx].reshape(B, P)
    # record_false_neg_idx = record_false_neg_mask.nonzero()              # [num_false_neg, 2]
    # for idx, false_neg in enumerate(record_false_neg_idx):
    #     batch_idx, proposal_idx = false_neg
    #     false_neg_vid_list.append(batch_info['vid'][batch_idx])
    #     false_neg_time_list.append(
    #         (moments[proposal_idx].cpu().detach().numpy() * batch_info['duration'][batch_idx].item()).tolist()
    #     )

    # Record some samples
    # if dist.is_main():
    #     # record 1st sample of each batch
    #     mean_pos_sim = pos_sim_record.mean().item()
    #     max_neg_sim = neg_sim_record.max().item()
    #     mean_neg_sim = neg_sim_record.mean().item()
    #     min_neg_sim = neg_sim_record.min().item()

    #     append_to_json_file(
    #         os.path.join(config.logdir, "false_neg.json"),
    #         {
    #             'epoch': epoch,
    #             'vid': batch_info['vid'][record_batch_idx],
    #             'query': batch_info['sentences'][record_batch_idx][0],
    #             'sample_idx': batch_info['idx'][record_batch_idx].item(),
    #             'pos_sim': pos_sim_record.tolist(),           # list[float], sim of all pos samples
    #             'max_neg_sim': max_neg_sim,                   # float
    #             'mean_neg_sim': mean_neg_sim,                 # float
    #             'min_neg_sim': min_neg_sim,                   # float
    #             'mean_gap': mean_pos_sim - mean_neg_sim,      # float, mean_pos_sim - mean_neg_sim
    #             'false_neg_vid': false_neg_vid_list,          # list[str]
    #             'false_neg_timestamps': false_neg_time_list,  # list[[float, float]]
    #         }
    #     )

    return false_neg_mask_con, false_neg_mask_iou


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_iou_fn: torch.nn.Module,
    loss_inter_fn: torch.nn.Module,
    loss_intra_fn: torch.nn.Module,
    epoch: int,
    config: AttrDict,
):
    device = dist.get_device()
    model.train()
    pbar = tqdm(        # progress bar for each epoch
        loader,         # length is determined by the number of batches
        ncols=0,        # disable bar, only show percentage
        leave=False,    # when the loop is finished, the bar will be removed
        disable=not dist.is_main(),
        desc=f"Epoch {epoch}",
    )
    losses = defaultdict(list)
    pred_moments = []
    true_moments = []
    for batch, batch_info in pbar:
        batch['video_feats'] = update_vgg_features(
            batch['video_feats'], batch_info['augmented_data'], config['mixup_alpha'])
        batch = {key: value.to(device) for key, value in batch.items()}
        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)  # [M, N, N]
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])               # [S, N, N]
        # video_feats: [num_sents, seq_len, feat_dim]
        video_feats, sents_feats, logits2d, scores2ds, mask2d = model(**batch)

        if config.do_afnd:
            # Find false negative and return [S, B * P] mask
            false_neg_mask_con, false_neg_mask_iou = find_false_negative(
                video_feats=video_feats,
                sents_feats=sents_feats,
                num_sentences=batch['num_sentences'],
                num_targets=batch['num_targets'],
                iou2d=iou2d,
                iou2ds=iou2ds,
                mask2d=mask2d,
                config=config,
                epoch=epoch,
                batch_info=batch_info,
            )
        # Don't do AFND
        else:
            false_neg_mask_con = None
            false_neg_mask_iou = None

        loss_iou, loss_iou_metrics = loss_iou_fn(
            logits2d=logits2d,
            iou2d=iou2d,
            mask2d=mask2d,
            false_neg_mask=false_neg_mask_iou,
        )

        loss_inter, loss_inter_metrics = loss_inter_fn(
            video_feats=video_feats,
            sents_feats=sents_feats,
            num_sentences=batch['num_sentences'],
            num_targets=batch['num_targets'],
            iou2d=iou2d,
            iou2ds=iou2ds,
            mask2d=mask2d,
            epoch=epoch,
            false_neg_mask=false_neg_mask_con,
        )

        loss_intra, loss_intra_metrics = loss_intra_fn(
            video_feats=video_feats,
            sents_feats=sents_feats,
            num_sentences=batch['num_sentences'],
            num_targets=batch['num_targets'],
            iou2d=iou2d,
            iou2ds=iou2ds,
            mask2d=mask2d,
            epoch=epoch,
            false_neg_mask=false_neg_mask_con,
        )

        # Contrastive Decay
        if epoch < config.contrastive_decay_start:
            loss = \
                loss_iou + loss_inter + loss_intra
        else:
            loss = \
                loss_iou + (loss_inter + loss_intra) * config.contrastive_decay

        loss.backward()
        if config.grad_clip > 0:
            clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

        # save loss to tensorboard
        metrics = {
            'loss/total': loss,
            **loss_iou_metrics,
            **loss_inter_metrics,
            **loss_intra_metrics,
        }

        for key, value in metrics.items():
            losses[key].append(value.cpu())

        # update progress bar
        pbar.set_postfix_str(f"loss: {loss.item():.2f}")
    pbar.close()

    losses = {key: torch.stack(value).mean() for key, value in losses.items()}

    return pred_moments, true_moments, losses


def training_loop(config: AttrDict):
    set_seed(config.seed + dist.get_rank())
    device = dist.get_device()

    # train Dataset and DataLoader
    train_dataset = construct_class(
        config.TrainDataset,
        do_augmentation=config.do_augmentation,
        mixup_alpha=config.mixup_alpha,
        downsampling_method=config.downsampling_method,
        aug_prob=config.aug_prob,
        downsampling_prob=config.downsampling_prob,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=config.seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size // dist.get_world_size(),
        collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
        num_workers=min(torch.get_num_threads(), 2),
    )

    # if "activity" in config.TrainDataset:
    #     val_dataset = construct_class(config.ValDataset)
    #     val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=config.seed)
    #     val_loader = DataLoader(
    #         dataset=val_dataset,
    #         batch_size=config.test_batch_size // dist.get_world_size(),
    #         collate_fn=val_dataset.collate_fn,
    #         sampler=val_sampler,
    #         num_workers=min(torch.get_num_threads(), 8),
    #     )

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

    if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
        # re-annotated multi-test Dataset and DataLoader
        multi_test_dataset = construct_class(config.MultiTestDataset)
        multi_test_sampler = DistributedSampler(multi_test_dataset,
                                                shuffle=False,
                                                seed=config.seed)
        multi_test_loader = DataLoader(
            dataset=multi_test_dataset,
            batch_size=config.test_batch_size // dist.get_world_size(),
            collate_fn=multi_test_dataset.collate_fn,
            sampler=multi_test_sampler,
            num_workers=min(torch.get_num_threads(), 8),
        )

    # loss functions
    loss_iou_fn = construct_class(
        config.IoULoss,
        min_iou=config.min_iou,
        max_iou=config.max_iou,
        weight=config.iou_weight,
    )
    loss_inter_fn = construct_class(
        config.InterContrastiveLoss,
        t=config.inter_t,
        m=config.inter_m,
        neg_iou=config.neg_iou,
        pos_topk=config.pos_topk,
        weight=config.inter_weight,
    )
    loss_intra_fn = construct_class(
        config.IntraContrastiveLoss,
        t=config.intra_t,
        m=config.intra_m,
        neg_iou=config.neg_iou,
        pos_topk=config.pos_topk,
        weight=config.intra_weight,
    )

    # model
    model_local = MMN(
        backbone=config.backbone,
        num_init_clips=config.num_init_clips,
        feat1d_in_channel=train_dataset.get_feat_dim(),
        feat1d_out_channel=config.feat1d_out_channel,
        feat1d_pool_kernel_size=config.feat1d_pool_kernel_size,
        feat1d_pool_stride_size=config.num_init_clips // config.num_clips,
        feat2d_pool_counts=config.feat2d_pool_counts,
        conv2d_hidden_channel=config.conv2d_hidden_channel,
        conv2d_kernel_size=config.conv2d_kernel_size,
        conv2d_num_layers=config.conv2d_num_layers,
        joint_space_size=config.joint_space_size,
        resnet=config.resnet,
        dual_space=config.dual_space,
    ).to(device)

    model = SyncBatchNorm.convert_sync_batchnorm(model_local)
    model = DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True)
    bert_params = []
    base_params = []
    for name, params in model.named_parameters():
        if 'bert' in name:
            params.requires_grad_(False)
            bert_params.append(params)
        else:
            base_params.append(params)

    # optimizer
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': config.base_lr},
        {'params': bert_params, 'lr': config.bert_lr}
    ], betas=(0.9, 0.99), weight_decay=1e-5)
    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones, config.step_gamma)

    # val set testing
    # if "activity" in config.TrainDataset:
    #     val_pred_moments, val_true_moments = val_epoch(model, val_loader, 0, config)

    # evaluate test set
    test_pred_moments, test_true_moments = test_epoch(model, test_loader, 0, config)

    # multi testing
    if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
        multi_test_pred_moments, multi_test_true_moments = \
            test_epoch(model, multi_test_loader, 0, config)

    if dist.is_main():
        os.makedirs(config.logdir, exist_ok=True)
        os.makedirs(os.path.join(config.logdir, "plots"), exist_ok=True)
        json.dump(
            config,
            open(os.path.join(config.logdir, 'config.json'), "w"), indent=4)
        train_writer = SummaryWriter(os.path.join(config.logdir, "train"))
        test_writer = SummaryWriter(os.path.join(config.logdir, "test"))

        if "charades" in config.TrainDataset:
            # test set
            test_recall = calculate_recall(
                test_pred_moments, test_true_moments,
                config.recall_Ns, config.recall_IoUs
            )
            for name, value in test_recall.items():
                test_writer.add_scalar(f'recall/{name}', value, 0)

            #  multi test
            test_multi_recall = calculate_multi_recall(
                multi_test_pred_moments, multi_test_true_moments,
                [5,], config.recall_IoUs)
            for name, value in test_multi_recall.items():
                test_writer.add_scalar(f'recall/{name}', value, 0)

            # initialize best metrics
            best_recall = test_recall
            best_multi_recall = test_multi_recall

            # Print to terminal
            print("Epoch 0")
            print(f"Test set")
            print_recall(test_recall)
            print(f"Multi test set")
            print_multi_recall(test_multi_recall)

        elif "activity" in config.TrainDataset:
            # Val set
            # val_writer = SummaryWriter(os.path.join(config.logdir, "val"))
            # val_recall = calculate_recall(
            #     val_pred_moments, val_true_moments,
            #     config.recall_Ns, config.recall_IoUs
            # )
            # for name, value in val_recall.items():
            #     val_writer.add_scalar(f'recall/{name}', value, 0)

            # Test set
            test_recall = calculate_recall(
                test_pred_moments, test_true_moments,
                config.recall_Ns, config.recall_IoUs
            )
            for name, value in test_recall.items():
                test_writer.add_scalar(f'recall/{name}', value, 0)

            # Multi test set
            test_multi_recall = calculate_multi_recall(
                multi_test_pred_moments, multi_test_true_moments,
                [5,], config.recall_IoUs)
            for name, value in test_multi_recall.items():
                test_writer.add_scalar(f'recall/{name}', value, 0)

            # initialize best metrics
            # best_val_recall = val_recall
            best_recall = test_recall
            best_multi_recall = test_multi_recall

            # Print to terminal
            print("Epoch 0")
            # print(f"Val set")
            # print_recall(val_recall)
            print(f"Test set")
            print_recall(test_recall)
            print(f"Multi test set")
            print_multi_recall(test_multi_recall)

        elif "qv" in config.TrainDataset:
            test_mAPs = calculate_mAPs(test_pred_moments, test_true_moments)
            for name, value in test_mAPs.items():
                test_writer.add_scalar(f'mAP/{name}', value, 0)

            # initialize best metrics
            best_mAPs = test_mAPs

            # Print to terminal
            print("Epoch 0")
            print(f"Test set")
            print_mAPs(test_mAPs)

    dist.barrier()

    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)

        # freeze BERT parameters for the first few epochs
        if epoch == config.bert_fire_start:
            for param in bert_params:
                param.requires_grad_(True)
            model = SyncBatchNorm.convert_sync_batchnorm(model_local)
            model = DistributedDataParallel(model, device_ids=[device])

        # train
        train_pred_moments, train_true_moments, train_losses = train_epoch(
            model, train_loader, optimizer,
            loss_iou_fn, loss_inter_fn, loss_intra_fn, epoch, config)

        # val
        # if "activity" in config.TrainDataset:
        #     val_pred_moments, val_true_moments = val_epoch(
        #         model, val_loader, epoch, config)

        # test
        test_pred_moments, test_true_moments = test_epoch(
            model, test_loader, epoch, config)

        # multi-test
        if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
            multi_test_pred_moments, multi_test_true_moments = \
                test_epoch(model, multi_test_loader, epoch, config)

        scheduler.step()

        # Evaluate and print results, then save model
        if dist.is_main():
            # log learning rate and losses
            train_writer.add_scalar(
                "lr/base", optimizer.param_groups[0]["lr"], epoch)
            train_writer.add_scalar(
                "lr/bert", optimizer.param_groups[1]["lr"], epoch)
            for name, value in train_losses.items():
                train_writer.add_scalar(name, value, epoch)

            if "charades" in config.TrainDataset:
                # Train set
                train_recall = calculate_recall(
                    train_pred_moments, train_true_moments,
                    config.recall_Ns, config.recall_IoUs
                )
                for name, value in train_recall.items():
                    train_writer.add_scalar(f'recall/{name}', value, epoch)

                # Test set
                test_recall = calculate_recall(
                    test_pred_moments, test_true_moments,
                    config.recall_Ns, config.recall_IoUs
                )
                for name, value in test_recall.items():
                    test_writer.add_scalar(f'recall/{name}', value, epoch)

                # Multi test set
                test_multi_recall = calculate_multi_recall(
                    multi_test_pred_moments, multi_test_true_moments,
                    [5,], config.recall_IoUs
                )
                for name, value in test_multi_recall.items():
                    test_writer.add_scalar(f'recall/{name}', value, epoch)

                # show recall and mAPs in terminal
                print(f"Epoch {epoch}")
                print(f"test set")
                print_recall(test_recall)

                print(f"Multi test set")
                print_multi_recall(test_multi_recall)

                # save last checkpoint
                state = {
                    "model": model_local.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                # save last pth
                # path = os.path.join(config.logdir, f"last.pth")
                # torch.save(state, path)

                # periodically save checkpoint
                if epoch % config.save_freq == 0:
                    path = os.path.join(config.logdir, f"ckpt_{epoch}.pth")
                    torch.save(state, path)

                # save best checkpoint
                # Consider multi-target metric and config.best_metric at the same time
                if (test_multi_recall["R@(5/5),IoU=0.5"] + test_recall[config.best_metric]) > (best_multi_recall["R@(5/5),IoU=0.5"] + best_recall[config.best_metric]):
                    best_recall = test_recall
                    best_multi_recall = test_multi_recall
                    path = os.path.join(config.logdir, f"best.pth")
                    torch.save(state, path)

                # log best results
                for name, value in best_recall.items():
                    test_writer.add_scalar(f'best/{name}', value, epoch)
                for name, value in best_multi_recall.items():
                    test_writer.add_scalar(f'best/{name}', value, epoch)

                # flush to disk
                train_writer.flush()
                test_writer.flush()

                # Write json file
                append_to_json_file(
                    os.path.join(config.logdir, "metrics.json"),
                    {
                        'epoch': epoch,
                        'train': {
                            'recall': train_recall,
                        },
                        'test': {
                            'recall': test_recall,
                            'multi_recall': test_multi_recall,
                        },
                        'best_test': {
                            'recall': best_recall,
                            'multi_recall': best_multi_recall,
                        }
                    }
                )

            elif "activity" in config.TrainDataset:
                # Train set
                train_recall = calculate_recall(
                    train_pred_moments, train_true_moments,
                    config.recall_Ns, config.recall_IoUs
                )
                for name, value in train_recall.items():
                    train_writer.add_scalar(f'recall/{name}', value, epoch)

                # Val set
                # val_recall = calculate_recall(
                #     val_pred_moments, val_true_moments,
                #     config.recall_Ns, config.recall_IoUs
                # )
                # for name, value in val_recall.items():
                #     val_writer.add_scalar(f'recall/{name}', value, epoch)

                # Test set
                test_recall = calculate_recall(
                    test_pred_moments, test_true_moments,
                    config.recall_Ns, config.recall_IoUs
                )
                for name, value in test_recall.items():
                    test_writer.add_scalar(f'recall/{name}', value, epoch)

                # Multi test set
                test_multi_recall = calculate_multi_recall(
                    multi_test_pred_moments, multi_test_true_moments,
                    [5,], config.recall_IoUs
                )
                for name, value in test_multi_recall.items():
                    test_writer.add_scalar(f'recall/{name}', value, epoch)

                # show recall and mAPs in terminal
                print(f"Epoch {epoch}")
                # print(f"Val set")
                # print_recall(val_recall)

                print(f"test set")
                print_recall(test_recall)

                print(f"Multi test set")
                print_multi_recall(test_multi_recall)

                # save last checkpoint
                state = {
                    "model": model_local.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                path = os.path.join(config.logdir, f"last.pth")
                torch.save(state, path)

                # periodically save checkpoint
                if epoch % config.save_freq == 0:
                    path = os.path.join(config.logdir, f"ckpt_{epoch}.pth")
                    torch.save(state, path)

                # Save best checkpoint
                # if val_recall[config.best_metric] > best_val_recall[config.best_metric]:
                #     best_val_recall = val_recall

                # Consider multi-target metric and config.best_metric at the same time
                if (test_multi_recall["R@(5/5),IoU=0.5"] + test_recall[config.best_metric]) > (best_multi_recall["R@(5/5),IoU=0.5"] + best_recall[config.best_metric]):
                    best_recall = test_recall
                    best_multi_recall = test_multi_recall
                    path = os.path.join(config.logdir, f"best.pth")
                    torch.save(state, path)

                # log best results
                # for name, value in best_val_recall.items():
                #     val_writer.add_scalar(f'best/{name}', value, epoch)
                for name, value in best_recall.items():
                    test_writer.add_scalar(f'best/{name}', value, epoch)
                for name, value in best_multi_recall.items():
                    test_writer.add_scalar(f'best/{name}', value, epoch)

                # flush to disk
                train_writer.flush()
                # val_writer.flush()
                test_writer.flush()

                # Write json file
                append_to_json_file(
                    os.path.join(config.logdir, "metrics.json"),
                    {
                        'epoch': epoch,
                        'train': {
                            'recall': train_recall,
                        },
                        # 'val': {
                        #     'recall': val_recall,
                        # },
                        'test': {
                            'recall': test_recall,
                            'multi_recall': test_multi_recall,
                        },
                        'best_test': {
                            'recall': best_recall,
                            'multi_recall': best_multi_recall,
                        }
                    }
                )

            elif 'qv' in config.TrainDataset:
                # Train set
                train_mAPs = calculate_mAPs(train_pred_moments, train_true_moments)
                for name, value in train_mAPs.items():
                    train_writer.add_scalar(f'mAP/{name}', value, epoch)

                # Test set
                test_mAPs = calculate_mAPs(test_pred_moments, test_true_moments)
                for name, value in test_mAPs.items():
                    test_writer.add_scalar(f'mAP/{name}', value, epoch)

                # show recall and mAPs in terminal
                print(f"Epoch {epoch}")
                print(f"test set")
                print_mAPs(test_mAPs)

                # save last checkpoint
                state = {
                    "model": model_local.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                path = os.path.join(config.logdir, f"last.pth")
                torch.save(state, path)

                # periodically save checkpoint
                if epoch % config.save_freq == 0:
                    path = os.path.join(config.logdir, f"ckpt_{epoch}.pth")
                    torch.save(state, path)

                if test_mAPs[config.best_metric] > best_mAPs[config.best_metric]:
                    best_mAPs = test_mAPs
                    path = os.path.join(config.logdir, f"best.pth")
                    torch.save(state, path)

                # log best result
                for name, value in best_mAPs.items():
                    test_writer.add_scalar(f'best/{name}', value, epoch)

                # flush to disk
                train_writer.flush()
                test_writer.flush()

                # Write json file
                append_to_json_file(
                    os.path.join(config.logdir, "metrics.json"),
                    {
                        'epoch': epoch,
                        'train': {
                            'mAP': train_mAPs,
                        },
                        'test': {
                            'mAP': test_mAPs,
                        },
                        'best_test': {
                            'mAP': best_mAPs
                        }
                    }
                )

        dist.barrier()

    if dist.is_main():
        train_writer.close()
        # if "activity" in config.TrainDataset:
        #     val_writer.close()
        test_writer.close()
