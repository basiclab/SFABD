import json
import os
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import SyncBatchNorm
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_multi_recall, calculate_mAPs
from src.misc import AttrDict, set_seed, construct_class, print_metrics, print_multi_recall
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

        batch = {key: value.cpu() for key, value in batch.items()}
        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)          # [M, N, N]
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])                       # [S, N, N], separate to combined

        # ploting batch
        shift = 0
        for batch_idx, scores2d in enumerate(scores2ds.cpu()):
            # nms(pred)
            num_proposals = pred_moments_batch['num_proposals'][batch_idx]
            nms_moments = pred_moments_batch["out_moments"][shift: shift + num_proposals]  # [num_proposals, 2]
            nms_moments = (nms_moments * config.num_clips).round().long()
            if info['idx'][batch_idx] % 200 == 0:
                plot_path = os.path.join(
                    config.logdir,
                    "plots",
                    f"{info['idx'][batch_idx]}",
                    f"epoch_{epoch:02d}.jpg")
                if epoch == 0:
                    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plot_moments_on_iou2d(
                    iou2d[batch_idx], scores2d, nms_moments, plot_path)
            shift = shift + num_proposals

    return pred_moments, true_moments


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
        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])
        video_feats, sents_feats, logits2d, scores2ds, mask2d = model(**batch)

        # loss_inter, loss_inter_metrics = loss_inter_fn(
        #     video_feats=video_feats,
        #     sents_feats=sents_feats,
        #     num_sentences=batch['num_sentences'],
        #     num_targets=batch['num_targets'],
        #     iou2d=iou2d,
        #     iou2ds=iou2ds,
        #     mask2d=mask2d,
        # )
        #  return neg mask
        (loss_inter,
         loss_inter_metrics,
         bce_sampled_neg_mask,
         intra_sampled_neg_mask) = loss_inter_fn(
            video_feats=video_feats,
            sents_feats=sents_feats,
            num_sentences=batch['num_sentences'],
            num_targets=batch['num_targets'],
            iou2d=iou2d,
            iou2ds=iou2ds,
            mask2d=mask2d,
            epoch=epoch,
        )
        loss_intra, loss_intra_metrics = loss_intra_fn(
            video_feats=video_feats,
            sents_feats=sents_feats,
            num_sentences=batch['num_sentences'],
            num_targets=batch['num_targets'],
            iou2d=iou2d,
            iou2ds=iou2ds,
            mask2d=mask2d,
            # sampled_neg_mask=intra_sampled_neg_mask,
        )
        # also do false negative removal
        loss_iou, loss_iou_metrics = loss_iou_fn(
            logits2d=logits2d,
            iou2d=iou2d,
            mask2d=mask2d,
            # sampled_neg_mask=bce_sampled_neg_mask,
        )

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


def train_epoch_mp_con(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_iou_fn: torch.nn.Module,
    loss_mpcon_fn: torch.nn.Module,
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
        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])

        video_feats, sents_feats, logits2d, scores2ds, mask2d = model(**batch)
        loss_iou, loss_iou_metrics = loss_iou_fn(
            logits2d=logits2d,
            iou2d=iou2d,
            mask2d=mask2d)

        # testing MultiPositiveContrastive
        loss_mpcon, loss_mpcon_metrics = loss_mpcon_fn(
            video_feats=video_feats,
            sents_feats=sents_feats,
            num_sentences=batch['num_sentences'],
            num_targets=batch['num_targets'],
            iou2d=iou2d,
            iou2ds=iou2ds,
            mask2d=mask2d,
        )

        if epoch < config.contrastive_decay_start:
            loss = \
                loss_iou + loss_mpcon
        else:
            loss = \
                loss_iou + loss_mpcon * config.contrastive_decay

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

        metrics = {
            'loss/total': loss,
            **loss_iou_metrics,
            **loss_mpcon_metrics,
        }
        for key, value in metrics.items():
            losses[key].append(value.cpu())

        # update progress bar
        pbar.set_postfix_str(f"loss: {loss.item():.2f}")
    pbar.close()

    losses = {key: torch.stack(value).mean() for key, value in losses.items()}

    return pred_moments, true_moments, losses


def training_loop(config: AttrDict):
    set_seed(config.seed)
    device = dist.get_device()

    # train Dataset and DataLoader
    train_dataset = construct_class(config.TrainDataset)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=config.seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size // dist.get_world_size(),
        collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

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
    #     top_neg_removal_percent=config.top_neg_removal_percent,
    #     weight=config.inter_weight,
    # )
    # loss_intra_fn = construct_class(
    #     config.IntraContrastiveLoss,
    #     t=config.intra_t,
    #     m=config.intra_m,
    #     neg_iou=config.neg_iou,
    #     pos_topk=config.pos_topk,
    #     top_neg_removal_percent=config.top_neg_removal_percent,
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
        top_neg_removal_percent=config.top_neg_removal_percent,
        weight=config.inter_weight,
        inter_query_threshold=config.inter_query_threshold,
        intra_video_threshold=config.intra_video_threshold,
        fusion_ratio=config.fusion_ratio,
        exponent=config.exponent,
        neg_samples_num=config.neg_samples_num,
        start_DNS_epoch=config.start_dns_epoch,
        rate_step_change=config.rate_step_change,
    )
    loss_intra_fn = construct_class(
        config.IntraContrastiveLoss + 'DNS',
        t=config.intra_t,
        m=config.intra_m,
        neg_iou=config.neg_iou,
        pos_topk=config.pos_topk,
        top_neg_removal_percent=config.top_neg_removal_percent,
        weight=config.intra_weight,
        mixup_alpha=config.mixup_alpha
    )

    # testing MultiPositiveContrastive
    # loss_mpcon_fn = construct_class(
    #     config.MultiPositiveContrastiveLoss,
    #     t=config.inter_t,
    #     inter_m=config.inter_m,
    #     intra_m=config.intra_m,
    #     neg_iou=config.neg_iou,
    #     pos_topk=config.pos_topk,
    #     inter_weight=config.inter_weight,
    #     intra_weight=config.intra_weight,
    # )

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

    # evaluate test set before the start of training
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

        test_recall = calculate_recall(
            test_pred_moments, test_true_moments,
            config.recall_Ns, config.recall_IoUs)
        test_mAPs = calculate_mAPs(test_pred_moments, test_true_moments)
        # multi testing
        if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
            test_multi_recall = calculate_multi_recall(
                multi_test_pred_moments, multi_test_true_moments,
                [5,], config.recall_IoUs)

        for name, value in test_recall.items():
            test_writer.add_scalar(f'recall/{name}', value, 0)
        for name, value in test_mAPs.items():
            test_writer.add_scalar(f'mAP/{name}', value, 0)
        if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
            for name, value in test_multi_recall.items():
                test_writer.add_scalar(f'recall/{name}', value, 0)

        # print to terminal
        print("Epoch 0")
        print_metrics(test_mAPs, test_recall)
        if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
            print_multi_recall(test_multi_recall)
        best_recall = test_recall
        best_mAPs = test_mAPs
        if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
            best_multi_recall = test_multi_recall
    dist.barrier()

    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)

        # freeze BERT parameters for the first few epochs
        if epoch == config.bert_fire_start:
            for param in bert_params:
                param.requires_grad_(True)
            model = SyncBatchNorm.convert_sync_batchnorm(model_local)
            model = DistributedDataParallel(model, device_ids=[device])

        # original inter, intra contrastive
        train_pred_moments, train_true_moments, train_losses = train_epoch(
            model, train_loader, optimizer,
            loss_iou_fn, loss_inter_fn, loss_intra_fn, epoch, config)

        # MultiPositive Contrastive
        # train_pred_moments, train_true_moments, train_losses = train_epoch_mp_con(
        #     model, train_loader, optimizer,
        #     loss_iou_fn, loss_mpcon_fn, epoch, config)

        test_pred_moments, test_true_moments = test_epoch(
            model, test_loader, epoch, config)

        if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
            multi_test_pred_moments, multi_test_true_moments = \
                test_epoch(model, multi_test_loader, epoch, config)

        scheduler.step()

        if dist.is_main():
            # log learning rate and losses
            train_writer.add_scalar(
                "lr/base", optimizer.param_groups[0]["lr"], epoch)
            train_writer.add_scalar(
                "lr/bert", optimizer.param_groups[1]["lr"], epoch)
            for name, value in train_losses.items():
                train_writer.add_scalar(name, value, epoch)

            # evaluate train set
            train_recall = calculate_recall(
                train_pred_moments, train_true_moments,
                config.recall_Ns, config.recall_IoUs)
            train_mAPs = calculate_mAPs(train_pred_moments, train_true_moments)
            for name, value in train_recall.items():
                train_writer.add_scalar(f'recall/{name}', value, epoch)
            for name, value in train_mAPs.items():
                train_writer.add_scalar(f'mAP/{name}', value, epoch)

            # evaluate test set
            test_recall = calculate_recall(
                test_pred_moments, test_true_moments,
                config.recall_Ns, config.recall_IoUs)
            test_mAPs = calculate_mAPs(test_pred_moments, test_true_moments)
            test_multi_recall = None    # for QVHighlights

            # multi testing
            if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
                test_multi_recall = calculate_multi_recall(
                    multi_test_pred_moments, multi_test_true_moments,
                    [5,], config.recall_IoUs)

            for name, value in test_recall.items():
                test_writer.add_scalar(f'recall/{name}', value, epoch)
            for name, value in test_mAPs.items():
                test_writer.add_scalar(f'mAP/{name}', value, epoch)
            if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
                for name, value in test_multi_recall.items():
                    test_writer.add_scalar(f'recall/{name}', value, epoch)

            # show recall and mAPs in terminal
            print(f"Epoch {epoch}")
            print_metrics(test_mAPs, test_recall)
            if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
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
            if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
                if test_recall[config.best_metric] > best_recall[config.best_metric]:
                    best_recall = test_recall
                    best_mAPs = test_mAPs
                    best_multi_recall = test_multi_recall
                    path = os.path.join(config.logdir, f"best.pth")
                    torch.save(state, path)
            elif "qvhighlights" in config.TrainDataset:
                if test_mAPs[config.best_metric] > best_mAPs[config.best_metric]:
                    best_recall = test_recall
                    best_mAPs = test_mAPs
                    path = os.path.join(config.logdir, f"best.pth")
                    torch.save(state, path)
                best_multi_recall = None    # for QVHighlights

            # log best results
            for name, value in best_recall.items():
                test_writer.add_scalar(f'best/{name}', value, epoch)
            for name, value in best_mAPs.items():
                test_writer.add_scalar(f'best/{name}', value, epoch)
            if "charades" in config.TrainDataset or "activity" in config.TrainDataset:
                for name, value in best_multi_recall.items():
                    test_writer.add_scalar(f'best/{name}', value, epoch)

            # flush to disk
            train_writer.flush()
            test_writer.flush()

            # save evaluation results to file
            append_to_json_file(
                os.path.join(config.logdir, "metrics.json"),
                {
                    'epoch': epoch,
                    'train': {
                        'recall': train_recall,
                        'mAP': train_mAPs,
                    },
                    'test': {
                        'recall': test_recall,
                        'multi_recall': test_multi_recall,
                        'mAP': test_mAPs,
                    },
                    'best_test': {
                        'recall': best_recall,
                        'multi_recall': best_multi_recall,
                        'mAP': best_mAPs,
                    }
                }
            )
        dist.barrier()

    if dist.is_main():
        train_writer.close()
        test_writer.close()
