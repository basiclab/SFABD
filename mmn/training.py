from copy import deepcopy
import json
import os

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mmn.datasets import MultiTargetCharadesDataset
from mmn.evaluation import inference_loop, evaluate
from mmn.losses import ContrastiveLoss, ScaledBCELoss
from mmn.misc import AttrDict, set_seed
from mmn.models.main import MMN


device = torch.device('cuda:0')


def write_recall_to_file(path, train_recalls, test_recalls, step, epoch):
    if os.path.exists(path):
        history = json.load(open(path, 'r'))
    else:
        history = []
    history.append({
        'epoch': epoch,
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


def training_loop(config: AttrDict):
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=False)
    json.dump(
        config,
        open(os.path.join(config.output_dir, 'config.json'), "w"),
        indent=4)
    writer = SummaryWriter(os.path.join(config.output_dir, "train"))
    test_writer = SummaryWriter(os.path.join(config.output_dir, "test"))

    train_dataset = MultiTargetCharadesDataset(
        ann_file=config.datasets.train.ann_file,
        vgg_feat_file=config.datasets.train.vgg_feat_file,
        c3d_feat_folder=config.datasets.train.c3d_feat_folder,
        num_init_clips=config.datasets.num_init_clips,
        num_clips=config.model.num_clips,
        feat_type=config.datasets.feat_type,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.optimizer.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=min(torch.get_num_threads(), 8),
        drop_last=True,
    )

    test_dataset = MultiTargetCharadesDataset(
        ann_file=config.datasets.test.ann_file,
        vgg_feat_file=config.datasets.test.vgg_feat_file,
        c3d_feat_folder=config.datasets.test.c3d_feat_folder,
        num_init_clips=config.datasets.num_init_clips,
        num_clips=config.model.num_clips,
        feat_type=config.datasets.feat_type,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test.batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=min(torch.get_num_threads(), 8),
    )

    loss_bce_fn = ScaledBCELoss(
        min_iou=config.loss.iou.min_iou,
        max_iou=config.loss.iou.max_iou,
    )
    loss_con_fn = ContrastiveLoss(
        T_v=config.loss.contrastive.tau_video,
        T_q=config.loss.contrastive.tau_query,
        neg_video_iou=config.loss.contrastive.neg_video_iou,
        pos_video_topk=config.loss.contrastive.pos_video_topk,
    )

    model = MMN(
        conv1d_in_channel=config.model.conv1d.in_channel,
        conv1d_out_channel=config.model.conv1d.out_channel,
        conv1d_pool_kerenl_size=config.model.conv1d.pool_kernel_size,
        conv1d_pool_stride_size=config.model.conv1d.pool_kernel_stride,
        conv2d_in_dim=config.model.num_clips,
        conv2d_in_channel=config.model.conv2d.in_channel,
        conv2d_hidden_channel=config.model.conv2d.hidden_channel,
        conv2d_kernel_size=config.model.conv2d.kernel_size,
        conv2d_num_layers=config.model.conv2d.num_layers,
        joint_space_size=config.model.joint_space_size,
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
        {'params': base_params, 'lr': config.optimizer.lr},
        {'params': bert_params, 'lr': config.optimizer.lr * 0.1}
    ], betas=(0.9, 0.99), weight_decay=1e-5)

    model.train()
    step = 0
    for epoch in range(config.optimizer.epochs):
        # freeze BERT parameters for the first few epochs
        if epoch < config.optimizer.bert_freeze_epoch:  # TODO: breaking change
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
            model.train()
            batch = {key: value.to(device) for key, value in batch.items()}
            video_feats, query_feats, sents_feats, scores2d = model(**batch)

            loss_iou = loss_bce_fn(scores2d, batch["iou2d"])
            if epoch < config.optimizer.only_iou_epoch:
                loss_inter_video = torch.zeros((), device=device)
                loss_inter_query = torch.zeros((), device=device)
                loss_intra_video = torch.zeros((), device=device)
                loss_intra_query = torch.zeros((), device=device)
                loss_con = torch.zeros((), device=device)
            else:
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
                    scatter_idx=batch["scatter_idx"],
                )

                loss_con = torch.stack([
                    loss_inter_video,
                    loss_inter_query,
                    loss_intra_video,
                    loss_intra_query,
                ]).sum()

            loss = (
                loss_iou * config.loss.iou.weight +
                loss_con * config.loss.contrastive.weight   # TODO: check weight
            )

            optimizer.zero_grad()
            loss.backward()
            if config.optimizer.clip_grad_norm > 0:
                clip_grad_norm_(
                    model.parameters(), config.optimizer.clip_grad_norm)
            optimizer.step()

            # save results for recall calculation
            train_results.append({
                'scores2d': scores2d.detach().cpu(),
                **deepcopy(info),
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
            config.test.nms_threshold,
            config.test.rec_metrics,
            config.test.iou_metrics,
        )
        # save training recall to tensorboard
        write_recall_to_tensorboard(writer, train_recalls, step)

        # evaluate test set
        test_results = inference_loop(model, test_loader)
        test_recalls, _ = evaluate(
            test_results,
            config.test.nms_threshold,
            config.test.rec_metrics,
            config.test.iou_metrics,
        )
        # save testing recall to tensorboard
        write_recall_to_tensorboard(test_writer, test_recalls, step)

        # save evaluation results to file
        write_recall_to_file(
            os.path.join(config.output_dir, "recall.json"),
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
        path = os.path.join(config.output_dir, f"epoch_{epoch + 1}.pth")
        torch.save(state, path)
