import json
import os
from collections import defaultdict
from typing import List, Dict

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, evaluate
from src.losses import ContrastiveLoss, ScaledIoULoss
from src.misc import set_seed, print_table, construct_class
from src.models.model import MMN
from src.utils import moment_to_iou2d


def write_recall_to_file(path, train_recall, test_recall, epoch):
    if os.path.exists(path):
        history = json.load(open(path, 'r'))
    else:
        history = []
    history.append({
        'epoch': epoch,
        'train': train_recall,
        'test': test_recall,
    })
    json.dump(history, open(path, 'w'), indent=4)


def test_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    # config parameters
    rec_metrics: List[int],
    nms_threshold: float,
    **dummy,
) -> List[Dict]:
    device = dist.get_device()
    model.eval()
    results = []
    for batch in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2d, _ = model(**batch)
        result = evaluate(
            scores2d,
            batch['moments'],
            batch['idxs'],
            rec_metrics,
            nms_threshold,
        )
        result = dist.all_gather_dict(result)
        results.append(result)
    return results


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_iou_fn: torch.nn.Module,
    loss_con_fn: torch.nn.Module,
    epoch: int,
    # config parameters
    num_clips: int,
    iou_weight: float,
    contrastive_weight: float,
    only_iou_epoch: int,
    clip_grad_norm: float,
    rec_metrics: List[int],
    nms_threshold: float,
    **dummy,
):
    device = dist.get_device()
    pbar = tqdm(        # progress bar for each epoch
        loader,         # length is determined by the number of batches
        ncols=0,        # disable bar, only show percentage
        leave=False,    # when the loop is finished, the bar will be removed
        disable=not dist.is_main(),
        desc=f"Epoch {epoch}",
    )
    losses = defaultdict(list)
    results = []  # for recall calculation
    for batch in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        video_feats, sents_feats, scores2d, mask2d = model(**batch)
        iou2ds = moment_to_iou2d(batch['moments'], num_clips)

        loss_iou = loss_iou_fn(scores2d, iou2ds, mask2d)
        if contrastive_weight != 0:
            (
                loss_inter_video,
                loss_inter_query,
            ) = loss_con_fn(
                video_feats,
                sents_feats,
                num_targets=batch["num_targets"],
                iou2ds=iou2ds,
                mask2d=mask2d,
            )

            loss_con = torch.stack([
                loss_inter_video,
                loss_inter_query,
            ]).sum()
        else:
            loss_inter_video = torch.zeros((), device=device)
            loss_inter_query = torch.zeros((), device=device)
            loss_con = torch.zeros((), device=device)

        loss = 0
        if epoch < only_iou_epoch:
            loss += loss_iou * iou_weight
            loss += loss_con * contrastive_weight
        else:
            loss += loss_iou * iou_weight
            loss += loss_con * contrastive_weight * 0.1

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm > 0:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        # save results for recall calculation
        result = evaluate(
            scores2d.detach(),
            batch['moments'],
            batch['idxs'],
            rec_metrics,
            nms_threshold,
        )
        result = dist.all_gather_dict(result)
        results.append(result)

        # save loss to tensorboard
        losses['loss/total'].append(loss.cpu())
        losses['loss/iou'].append(loss_iou.cpu())
        losses['loss/inter_video'].append(loss_inter_video.cpu())
        losses['loss/inter_query'].append(loss_inter_query.cpu())

        # update progress bar
        pbar.set_postfix_str(", ".join([
            f"loss: {loss.item():.2f}",
            f"iou: {loss_iou.item():.2f}",
            f"video: {loss_inter_video.item():.2f}",
            f"query: {loss_inter_query.item():.2f}",
        ]))
    pbar.close()

    losses = {key: torch.stack(value).mean() for key, value in losses.items()}
    return results, losses


def training_loop(
    seed: int,
    TrainDataset: str,      # Train dataset class name
    train_ann_file: str,    # Train annotation file path
    TestDataset: str,       # Test dataset class name
    test_ann_file: str,     # Test annotation file path
    feat_file: str,         # feature file path
    feat_channel: int,      # feature channel size
    num_init_clips: int,    # Number of initial clips
    num_clips: int,         # Number of clips
    # model
    feat1d_out_channel: int,
    feat1d_pool_kerenl_size: int,
    feat2d_pool_counts: List[int],
    conv2d_hidden_channel: int,
    conv2d_kernel_size: int,
    conv2d_num_layers: int,
    joint_space_size: int,
    # iou loss
    min_iou: float,
    max_iou: float,
    iou_weight: float,
    # contrastive loss
    tau_video: float,
    tau_query: float,
    neg_video_iou: float,
    pos_video_topk: int,
    margin: float,
    contrastive_weight: float,
    # optimizer
    base_lr: float,
    bert_lr: float,
    milestones: List[int],
    batch_size: int,
    epochs: int,
    bert_freeze_epoch: int,
    only_iou_epoch: int,
    clip_grad_norm: float,
    # test
    test_batch_size: int,
    nms_threshold: float,
    rec_metrics: List[float],
    iou_metrics: List[float],
    # logging
    logdir: str,
    kwargs: Dict,
):
    set_seed(seed)
    device = dist.get_device()

    train_dataset = construct_class(
        TrainDataset,
        ann_file=train_ann_file,
        num_clips=num_clips,
        feat_file=feat_file,
        num_init_clips=num_init_clips,
        seed=seed,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size // dist.get_world_size(),
        collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

    test_dataset = construct_class(
        TestDataset,
        ann_file=test_ann_file,
        num_clips=num_clips,
        feat_file=feat_file,
        num_init_clips=num_init_clips,
        seed=seed,
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=seed)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size // dist.get_world_size(),
        collate_fn=test_dataset.collate_fn,
        sampler=test_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

    loss_iou_fn = ScaledIoULoss(min_iou, max_iou)
    loss_con_fn = ContrastiveLoss(
        T_v=tau_video,
        T_q=tau_query,
        neg_video_iou=neg_video_iou,
        pos_video_topk=pos_video_topk,
        margin=margin,
    )

    model = MMN(
        feat1d_in_channel=feat_channel,
        feat1d_out_channel=feat1d_out_channel,
        feat1d_pool_kerenl_size=feat1d_pool_kerenl_size,
        feat1d_pool_stride_size=num_init_clips // num_clips,
        feat2d_pool_counts=feat2d_pool_counts,
        conv2d_hidden_channel=conv2d_hidden_channel,
        conv2d_kernel_size=conv2d_kernel_size,
        conv2d_num_layers=conv2d_num_layers,
        joint_space_size=joint_space_size,
    ).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True)

    bert_params = []
    base_params = []
    for name, params in model.named_parameters():
        if 'bert' in name:
            bert_params.append(params)
        else:
            base_params.append(params)

    # optimizer
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': base_lr},
        {'params': bert_params, 'lr': bert_lr}
    ], betas=(0.9, 0.99), weight_decay=1e-5)
    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)

    # evaluate test set before training to get initial recall
    test_results = test_epoch(model, test_loader, **kwargs)
    if dist.is_main():
        os.makedirs(logdir, exist_ok=False)
        json.dump(
            kwargs, open(os.path.join(logdir, 'config.json'), "w"), indent=4)
        train_writer = SummaryWriter(os.path.join(logdir, "train"))
        test_writer = SummaryWriter(os.path.join(logdir, "test"))

        test_recall = calculate_recall(
            test_results,
            rec_metrics,
            iou_metrics,
        )
        for metric_name, value in test_recall.items():
            test_writer.add_scalar(f'recall/{metric_name}', value, 0)
        print_table(epoch=0, recalls_dict={'test': test_recall})
    dist.barrier()

    for epoch in range(1, epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        # freeze BERT parameters for the first few epochs
        if epoch < bert_freeze_epoch:
            bert_requires_grad = False
        else:
            bert_requires_grad = True
        for param in bert_params:
            param.requires_grad_(bert_requires_grad)

        train_results, train_losses = train_epoch(
            model, train_loader, optimizer, loss_iou_fn, loss_con_fn, epoch,
            **kwargs)
        test_results = test_epoch(model, test_loader, **kwargs)
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
                train_results, rec_metrics, iou_metrics)
            for metric_name, value in train_recall.items():
                train_writer.add_scalar(f'recall/{metric_name}', value, epoch)

            # evaluate test set
            test_recall = calculate_recall(
                test_results, rec_metrics, iou_metrics)
            for metric_name, value in test_recall.items():
                test_writer.add_scalar(f'recall/{metric_name}', value, epoch)

            # save evaluation results to file
            write_recall_to_file(
                os.path.join(logdir, "recall.json"),
                train_recall,
                test_recall,
                epoch,
            )
            # print to terminal
            print_table(epoch, {"train": train_recall, "test": test_recall})

            # save model every epoch
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            path = os.path.join(logdir, f"epoch_{epoch}.pth")
            torch.save(state, path)

            train_writer.flush()
            test_writer.flush()
        dist.barrier()

    if dist.is_main():
        train_writer.close()
        test_writer.close()
