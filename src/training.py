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
from src.evaluation import calculate_recall, calculate_mAPs
from src.losses.main import ScaledIoULoss, ContrastiveLoss
from src.misc import AttrDict, set_seed, print_table, construct_class
from src.models.model import MMN
from src.utils import (
    nms, scores2ds_to_moments, moments_to_iou2ds, iou2ds_to_iou2d)


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
    config: AttrDict,
):
    device = dist.get_device()
    model.eval()
    pred_moments = []
    true_moments = []
    for batch, _ in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
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
    return pred_moments, true_moments


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_iou_fn: torch.nn.Module,
    loss_con_fn: torch.nn.Module,
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

        video_feats, sents_feats, logits2d, scores2d, mask2d = model(**batch)
        loss_iou = loss_iou_fn(logits2d, iou2d, mask2d)
        if config.contrastive_weight != 0:
            loss_inter_video, loss_inter_query, loss_intra_video = loss_con_fn(
                video_feats=video_feats,
                sents_feats=sents_feats,
                num_sentences=batch['num_sentences'],
                num_targets=batch['num_targets'],
                iou2d=iou2d,
                iou2ds=iou2ds,
                mask2d=mask2d,
            )
            loss_contrastive = (
                loss_inter_video + loss_inter_query + loss_intra_video)
        else:
            loss_inter_video = torch.zeros((), device=device)
            loss_inter_query = torch.zeros((), device=device)
            loss_intra_video = torch.zeros((), device=device)
            loss_contrastive = torch.zeros((), device=device)

        loss = 0
        if epoch <= config.only_iou_epoch:
            loss += loss_iou * config.iou_weight
            loss += loss_contrastive * config.contrastive_weight
        else:
            loss += loss_iou * config.iou_weight
            loss += loss_contrastive * config.contrastive_weight * 0.01

        loss.backward()
        if config.grad_clip > 0:
            clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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

        # save loss to tensorboard
        losses['loss/total'].append(loss.cpu())
        losses['loss/iou'].append(loss_iou.cpu())
        losses['loss/contrastive'].append(loss_contrastive.cpu())
        losses['loss/inter_video'].append(loss_inter_video.cpu())
        losses['loss/inter_query'].append(loss_inter_query.cpu())
        losses['loss/intra_video'].append(loss_intra_video.cpu())

        # update progress bar
        pbar.set_postfix_str(", ".join([
            f"loss: {loss.item():.2f}",
            f"iou: {loss_iou.item():.2f}",
            "[inter]",
            f"video: {loss_inter_video.item():.2f}",
            f"query: {loss_inter_query.item():.2f}",
            "[intra]",
            f"video: {loss_intra_video.item():.2f}",
        ]))
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

    # loss functions
    loss_iou_fn = ScaledIoULoss(config.min_iou, config.max_iou)
    loss_con_fn = ContrastiveLoss(
        T_v=config.tau_video,
        T_q=config.tau_query,
        neg_iou=config.neg_iou,
        pos_topk=config.pos_topk,
        margin=config.margin,
        inter=config.inter,
        intra=config.intra,
    )

    # model
    model_local = MMN(
        num_init_clips=config.num_init_clips,
        feat1d_in_channel=train_dataset.get_feat_dim(),
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

    # evaluate test set before training to get initial recall
    test_pred_moments, test_true_moments = test_epoch(model, test_loader, config)
    if dist.is_main():
        os.makedirs(config.logdir, exist_ok=False)
        json.dump(
            config,
            open(os.path.join(config.logdir, 'config.json'), "w"), indent=4)
        train_writer = SummaryWriter(os.path.join(config.logdir, "train"))
        test_writer = SummaryWriter(os.path.join(config.logdir, "test"))

        test_recall = calculate_recall(
            test_pred_moments, test_true_moments,
            config.recall_Ns, config.recall_IoUs)
        test_mAPs = calculate_mAPs(test_pred_moments, test_true_moments)
        for name, value in test_recall.items():
            test_writer.add_scalar(f'recall/{name}', value, 0)
        for name, value in test_mAPs.items():
            test_writer.add_scalar(f'mAP/{name}', value, 0)
        print_table(epoch=0, rows={'test': test_recall})
        print_table(epoch=0, rows={'test': test_mAPs})

        best_recall = test_recall
        best_mAPs = test_mAPs
    dist.barrier()

    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)

        # freeze BERT parameters for the first few epochs
        if epoch == config.bert_freeze_epoch + 1:
            for param in bert_params:
                param.requires_grad_(True)
            model = SyncBatchNorm.convert_sync_batchnorm(model_local)
            model = DistributedDataParallel(model, device_ids=[device])

        train_pred_moments, train_true_moments, train_losses = train_epoch(
            model, train_loader, optimizer, loss_iou_fn, loss_con_fn, epoch,
            config)
        test_pred_moments, test_true_moments = test_epoch(
            model, test_loader, config)
        scheduler.step()

        if dist.is_main():
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
            for name, value in test_recall.items():
                test_writer.add_scalar(f'recall/{name}', value, epoch)
            for name, value in test_mAPs.items():
                test_writer.add_scalar(f'mAP/{name}', value, epoch)

            # save evaluation results to file
            append_to_json_file(
                os.path.join(config.logdir, "recall.json"),
                {
                    'epoch': epoch,
                    'train': {
                        'recall': train_recall,
                        'mAP': train_mAPs,
                    },
                    'test': {
                        'recall': test_recall,
                        'mAP': test_mAPs,
                    },
                }
            )
            # print to terminal
            print_table(epoch, {"train": train_recall, "test": test_recall})
            print_table(epoch, {"train": train_mAPs, "test": test_mAPs})

            state = {
                "model": model_local.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            path = os.path.join(config.logdir, f"last.pth")
            torch.save(state, path)
            if epoch % config.save_freq == 0:
                path = os.path.join(config.logdir, f"ckpt_{epoch}.pth")
                torch.save(state, path)
            
            #if test_recall[config.best_metric] > best_recall[config.best_metric]: 
            if test_mAPs[config.best_metric] > best_mAPs[config.best_metric]:
                best_recall = test_recall
                best_mAPs = test_mAPs
                path = os.path.join(config.logdir, f"best.pth")
                torch.save(state, path)

            for name, value in best_recall.items():
                test_writer.add_scalar(f'best/{name}', value, epoch)
            for name, value in best_mAPs.items():
                test_writer.add_scalar(f'best/{name}', value, epoch)

            train_writer.flush()
            test_writer.flush()
        dist.barrier()

    if dist.is_main():
        train_writer.close()
        test_writer.close()
