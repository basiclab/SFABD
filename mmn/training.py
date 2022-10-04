import json
import os
from typing import List, Dict

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mmn.datasets import MultiTargetCharadesDataset
from mmn.evaluation import inference_loop, evaluate
from mmn.losses import ContrastiveLoss, ScaledIoULoss
from mmn.misc import set_seed, print_table
from mmn.models.main import MMN


device = torch.device('cuda:0')


def write_recall_to_file(path, train_recalls, test_recalls, step, epoch):
    if os.path.exists(path):
        history = json.load(open(path, 'r'))
    else:
        history = []
    history.append({
        'epoch': epoch + 1,
        'step': step,
        'train': train_recalls,
        'test': test_recalls,
    })
    json.dump(history, open(path, 'w'), indent=4)


def write_recall_to_tensorboard(writer, recalls, step):
    for target_name, recall in recalls.items():
        for metric_name, value in recall.items():
            writer.add_scalar(
                # ex. "recall/1-target/R@1,IoU0.5"
                f'recall/{target_name}/{metric_name}', value, step)


def training_loop(
    seed: int,
    train_ann_file: str,
    test_ann_file: str,
    vgg_feat_file: str,
    num_init_clips: int,
    num_clips: int,
    conv1d_in_channel: int,
    conv1d_out_channel: int,
    conv1d_pool_kernel_size: int,
    conv1d_pool_kernel_stride: int,
    conv2d_in_channel: int,
    conv2d_hidden_channel: int,
    conv2d_kernel_size: int,
    conv2d_num_layers: int,
    joint_space_size: int,
    min_iou: float,
    max_iou: float,
    iou_weight: float,
    tau_video: float,
    tau_query: float,
    neg_video_iou: float,
    pos_video_topk: int,
    margin: float,
    contrastive_weight: float,
    base_lr: float,
    bert_lr: float,
    batch_size: int,
    epochs: int,
    bert_freeze_epoch: int,
    clip_grad_norm: float,
    test_batch_size: int,
    nms_threshold: float,
    rec_metrics: List[float],
    iou_metrics: List[float],
    logdir: str,
    kwargs: Dict,
):
    set_seed(seed)
    os.makedirs(logdir, exist_ok=False)
    json.dump(kwargs, open(os.path.join(logdir, 'config.json'), "w"), indent=4)
    writer = SummaryWriter(os.path.join(logdir, "train"))
    test_writer = SummaryWriter(os.path.join(logdir, "test"))

    train_dataset = MultiTargetCharadesDataset(
        ann_file=train_ann_file,
        vgg_feat_file=vgg_feat_file,
        num_init_clips=num_init_clips,
        num_clips=num_clips,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=min(torch.get_num_threads(), 8),
        drop_last=True,
    )

    test_dataset = MultiTargetCharadesDataset(
        ann_file=test_ann_file,
        vgg_feat_file=vgg_feat_file,
        num_init_clips=num_init_clips,
        num_clips=num_clips,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=min(torch.get_num_threads(), 8),
    )

    loss_bce_fn = ScaledIoULoss(min_iou, max_iou)
    loss_con_fn = ContrastiveLoss(
        T_v=tau_video,
        T_q=tau_query,
        neg_video_iou=neg_video_iou,
        pos_video_topk=pos_video_topk,
    )

    model = MMN(
        conv1d_in_channel=conv1d_in_channel,
        conv1d_out_channel=conv1d_out_channel,
        conv1d_pool_kerenl_size=conv1d_pool_kernel_size,
        conv1d_pool_stride_size=conv1d_pool_kernel_stride,
        conv2d_in_dim=num_clips,
        conv2d_in_channel=conv2d_in_channel,
        conv2d_hidden_channel=conv2d_hidden_channel,
        conv2d_kernel_size=conv2d_kernel_size,
        conv2d_num_layers=conv2d_num_layers,
        joint_space_size=joint_space_size,
    ).to(device)

    # TODO: DDP
    bert_params = []
    base_params = []
    for name, params in model.named_parameters():
        if 'bert' in name:
            bert_params.append(params)
        else:
            base_params.append(params)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': base_lr},
        {'params': bert_params, 'lr': bert_lr}
    ], betas=(0.9, 0.99), weight_decay=1e-5)

    # evaluate test set before training to get initial recall
    test_results = inference_loop(model, test_loader)
    test_recalls, _ = evaluate(
        test_results,
        nms_threshold,
        rec_metrics,
        iou_metrics,
    )
    write_recall_to_tensorboard(test_writer, test_recalls, step=0)
    print_table(epoch=0, recalls_dict={'test': test_recalls})

    step = 0
    for epoch in range(epochs):
        model.train()

        # freeze BERT parameters for the first few epochs
        if epoch < bert_freeze_epoch:
            ber_requires_grad = False
        else:
            ber_requires_grad = True
        for param in bert_params:
            param.requires_grad_(ber_requires_grad)

        writer.add_scalar("lr/base", optimizer.param_groups[0]["lr"], step)
        writer.add_scalar("lr/bert", optimizer.param_groups[1]["lr"], step)

        pbar = tqdm(        # progress bar for each epoch
            train_loader,   # length is determined by the number of batches
            ncols=0,        # disable bar, only show percentage
            leave=False,    # when the loop is finished, the bar will be removed
            desc=f"Epoch {epoch + 1}",
        )
        train_results = []  # for recall calculation
        for batch, info in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            video_feats, query_feats, sents_feats, scores2d = model(**batch)

            loss_iou = loss_bce_fn(scores2d, batch["iou2d"])
            if contrastive_weight != 0:
                (
                    loss_inter_video,
                    loss_inter_query,
                    loss_intra_video,
                    loss_intra_query,
                ) = loss_con_fn(
                    video_feats,
                    query_feats,
                    sents_feats,
                    iou2d=batch["iou2d"],
                    iou2ds=batch["iou2ds"],
                    num_targets=batch["num_targets"],
                )

                loss_con = torch.stack([
                    loss_inter_video,
                    loss_inter_query,
                    loss_intra_video,
                    loss_intra_query,
                ]).sum()
            else:
                loss_inter_video = torch.zeros((), device=device)
                loss_inter_query = torch.zeros((), device=device)
                loss_intra_video = torch.zeros((), device=device)
                loss_intra_query = torch.zeros((), device=device)
                loss_con = torch.zeros((), device=device)

            loss = loss_iou * iou_weight + loss_con * contrastive_weight

            optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm > 0:
                clip_grad_norm_(
                    model.parameters(), clip_grad_norm)
            optimizer.step()

            # save results for recall calculation
            train_results.append({
                'scores2d': scores2d.detach().cpu(),
                **info,
            })

            # save loss to tensorboard
            step += 1
            writer.add_scalar('loss/total', loss.cpu(), step)
            writer.add_scalar('loss/iou', loss_iou.cpu(), step)
            writer.add_scalar('loss/inter_video', loss_inter_video.cpu(), step)
            writer.add_scalar('loss/inter_query', loss_inter_query.cpu(), step)
            writer.add_scalar('loss/intra_video', loss_intra_video.cpu(), step)
            writer.add_scalar('loss/intra_query', loss_intra_query.cpu(), step)
            # update progress bar
            pbar.set_postfix_str(", ".join([
                f"loss: {loss.item():.2f}",
                f"iou: {loss_iou.item():.2f}",
                f"[Intra] v: {loss_intra_video.item():.2f}",
                f"q: {loss_intra_query.item():.2f}",
                f"[Inter] v: {loss_inter_video.item():.2f}",
                f"q: {loss_inter_query.item():.2f}",
            ]))
        pbar.close()

        # evaluate train set
        train_recalls, _ = evaluate(
            train_results,
            nms_threshold,
            rec_metrics,
            iou_metrics,
        )
        # save training recall to tensorboard
        write_recall_to_tensorboard(writer, train_recalls, step)

        # evaluate test set
        test_results = inference_loop(model, test_loader)
        test_recalls, _ = evaluate(
            test_results,
            nms_threshold,
            rec_metrics,
            iou_metrics,
        )
        # save testing recall to tensorboard
        write_recall_to_tensorboard(test_writer, test_recalls, step)

        # save evaluation results to file
        write_recall_to_file(
            os.path.join(logdir, "recall.json"),
            train_recalls,
            test_recalls,
            step,
            epoch,
        )

        # save model every epoch
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        path = os.path.join(logdir, f"epoch_{epoch + 1}.pth")
        torch.save(state, path)

        # print to terminal
        print_table(epoch + 1, {"train": train_recalls, "test": test_recalls})

        writer.flush()
        test_writer.flush()
