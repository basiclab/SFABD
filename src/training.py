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
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_multi_recall, calculate_mAPs
from src.misc import (
    AttrDict, set_seed, construct_class,
    print_metrics, print_recall, print_mAPs, print_multi_recall)
from src.models.main import MMN
from src.utils import (
    nms, scores2ds_to_moments, moments_to_iou2ds, iou2ds_to_iou2d,
    plot_moments_on_iou2d)


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

        # plot prediction
        # batch = {key: value.cpu() for key, value in batch.items()}
        # iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)          # [M, N, N]
        # iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])                       # [S, N, N], separate to combined
        # if epoch > 0:
        #     # ploting batch
        #     shift = 0
        #     for batch_idx, scores2d in enumerate(scores2ds.cpu()):
        #         # nms(pred)
        #         num_proposals = pred_moments_batch['num_proposals'][batch_idx]
        #         nms_moments = pred_moments_batch["out_moments"][shift: shift + num_proposals]  # [num_proposals, 2]
        #         nms_moments = (nms_moments * config.num_clips).round().long()
        #         if info['idx'][batch_idx] % 200 == 0:
        #             plot_path = os.path.join(
        #                 config.logdir,
        #                 "plots",
        #                 f"{info['idx'][batch_idx]}",
        #                 f"epoch_{epoch:02d}.jpg")
        #             if epoch == 1:
        #                 os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        #             plot_moments_on_iou2d(
        #                 iou2d[batch_idx], scores2d, nms_moments, plot_path)
        #         shift = shift + num_proposals

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


def find_false_negative(
    video_feats: torch.Tensor,          # [S, C, N, N]
    sents_feats: torch.Tensor,          # [S, C]
    num_sentences: torch.Tensor,        # [B]
    num_targets: torch.Tensor,          # [S]
    iou2d: torch.Tensor,                # [S, N, N]
    iou2ds: torch.Tensor,               # [M, N, N]
    mask2d: torch.Tensor,               # [N, N]
    config: AttrDict,
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
    # neg mask for each target, for selecting neg proposals for each target
    target_inter_query_neg_mask = ~pos_mask[scatter_m2s]    # [M, B * P]

    # === inter video (topk proposal -> all sentences)
    topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
    topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
    allm_video_feats = video_feats[scatter_m2s]             # [M, P, C]
    topk_video_feats = allm_video_feats.gather(
        dim=1, index=topk_idxs)                             # [M, K, C]

    # need to convert video_feats from [S, P, C] to [B, P, C]
    inter_query_all = torch.mm(
        sents_feats,                                        # [S, C]
        video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
    )                                                       # [S, B * P]
    # compute cos_sim of query to all neg proposals
    inter_query_sim = inter_query_all[scatter_m2s]          # [M, B * P]
    inter_query_sim = inter_query_sim                       # [M, B * P]
    # [-1, 1] -> [0, 1]
    inter_query_sim = (inter_query_sim + 1) / 2             # [M, B * P]
    assert (inter_query_sim > 0 - 1e-3).all()
    assert (inter_query_sim < 1 + 1e-3).all()

    # compute cos_sim of topk video proposals and all neg proposals
    # intra_video_all  [M, B * P]
    intra_video_all = torch.matmul(
        topk_video_feats,                                   # [M, K, C]
        video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
    ).mean(dim=1)                                           # [M, B * P]
    # [-1, 1] -> [0, 1]
    intra_video_sim = (intra_video_all + 1) / 2             # [M, B * P]
    assert (intra_video_sim > 0 - 1e-3).all()
    assert (intra_video_sim < 1 + 1e-3).all()
    
    fused_neg_sim = config.fusion_ratio * inter_query_sim + \
        (1 - config.fusion_ratio) * intra_video_sim                # [M, B * P]
    
    # find the top-x% neg proposals for each target
    false_neg_mask = torch.zeros_like(fused_neg_sim)        # [M, B * P]
    for target_idx, (target_fused_neg_sim, neg_mask) in enumerate(zip(fused_neg_sim, target_inter_query_neg_mask)):
        neg_masked_fused_neg_sim = target_fused_neg_sim.masked_select(neg_mask)
        remove_mask = torch.zeros_like(neg_masked_fused_neg_sim)
        K = int(config.top_neg_removal_percent * int(neg_masked_fused_neg_sim.size(dim=0)))
        topk_idx = neg_masked_fused_neg_sim.topk(K, dim=0)[1]        
        remove_mask[topk_idx] = 1
        false_neg_mask[target_idx][neg_mask.clone()] = remove_mask
    
    # convert false_neg_mask from [M, B * P] to [S, B * P]
    num_t = 0
    final_false_neg_mask = torch.zeros(S, B * P, device=device)    
    for sent_idx, num_target in enumerate(num_targets):
        # if num_target == 1, find where > 0
        # if num_target > 1, find where > 1 (intersection of at least two targets' false neg)
        final_false_neg_mask[sent_idx] = \
            false_neg_mask[num_t: num_t + num_target].sum(dim=0) > min(num_target - 1, 1)
        num_t += num_target
    false_neg_mask_con = final_false_neg_mask.bool()        # [S, B * P]
    false_neg_mask_iou = false_neg_mask_con[local_mask].reshape(S, P)     # [S, P]

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
    for batch, _ in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)  # [M, N, N]
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])               # [S, N, N]
        # video_feats: [num_sents, seq_len, feat_dim]
        video_feats, sents_feats, logits2d, scores2ds, mask2d = model(**batch)

        if config.do_fnc:
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
            )
        # Don't do FNC
        else:
            false_neg_mask_con = None
            false_neg_mask_iou = None

        # FNC and DNS loss functions
        # Control do_dns by start_dns_epoch
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

        # # original loss functions        
        # loss_iou, loss_iou_metrics = loss_iou_fn(
        #     logits2d=logits2d,
        #     iou2d=iou2d,
        #     mask2d=mask2d,
        # )

        # loss_inter, loss_inter_metrics = loss_inter_fn(
        #     video_feats=video_feats,
        #     sents_feats=sents_feats,
        #     num_sentences=batch['num_sentences'],
        #     num_targets=batch['num_targets'],
        #     iou2d=iou2d,
        #     iou2ds=iou2ds,
        #     mask2d=mask2d,
        # )

        # loss_intra, loss_intra_metrics = loss_intra_fn(
        #     video_feats=video_feats,
        #     sents_feats=sents_feats,
        #     num_sentences=batch['num_sentences'],
        #     num_targets=batch['num_targets'],
        #     iou2d=iou2d,
        #     iou2ds=iou2ds,
        #     mask2d=mask2d,
        # )


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
        aug_expand_rate=config.aug_expand_rate,
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
        num_workers=min(torch.get_num_threads(), 8),
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
    # loss_iou_fn = construct_class(
    #     config.IoULoss,
    #     min_iou=config.min_iou,
    #     max_iou=config.max_iou,
    #     alpha=config.alpha,
    #     gamma=config.gamma,
    #     weight=config.iou_weight,
    # )
    # loss_inter_fn = construct_class(
    #     config.InterContrastiveLoss,
    #     t=config.inter_t,
    #     m=config.inter_m,
    #     neg_iou=config.neg_iou,
    #     pos_topk=config.pos_topk,
    #     weight=config.inter_weight,
    # )
    # loss_intra_fn = construct_class(
    #     config.IntraContrastiveLoss,
    #     t=config.intra_t,
    #     m=config.intra_m,
    #     neg_iou=config.neg_iou,
    #     pos_topk=config.pos_topk,
    #     weight=config.intra_weight,
    # )

    loss_iou_fn = construct_class(
        config.IoULoss + 'DNS',
        min_iou=config.min_iou,
        max_iou=config.max_iou,
        alpha=config.alpha,
        gamma=config.gamma,
        weight=config.iou_weight,
    )
    loss_inter_fn = construct_class(
        config.InterContrastiveLoss + 'DNS',
        t=config.inter_t,
        m=config.inter_m,
        neg_iou=config.neg_iou,
        pos_topk=config.pos_topk,
        weight=config.inter_weight,
        exponent=config.exponent,
        neg_samples_num=config.neg_samples_num,
        start_DNS_epoch=config.start_dns_epoch,
    )
    loss_intra_fn = construct_class(
        config.IntraContrastiveLoss + 'DNS',
        t=config.intra_t,
        m=config.intra_m,
        neg_iou=config.neg_iou,
        pos_topk=config.pos_topk,
        weight=config.intra_weight,
        exponent=config.exponent,
        neg_samples_num=config.neg_samples_num,
        start_DNS_epoch=config.start_dns_epoch,
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
                path = os.path.join(config.logdir, f"last.pth")
                torch.save(state, path)

                # periodically save checkpoint
                if epoch % config.save_freq == 0:
                    path = os.path.join(config.logdir, f"ckpt_{epoch}.pth")
                    torch.save(state, path)

                # save best checkpoint
                if test_recall[config.best_metric] > best_recall[config.best_metric]:
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

                if test_recall[config.best_metric] > best_recall[config.best_metric]:
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
