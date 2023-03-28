import json
import os
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch import optim
from torch.nn import SyncBatchNorm
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_mAPs
from src.losses.main import (
        ScaledIoULoss, ScaledIoUFocalLoss, ContrastiveLoss, ConfidenceLoss, 
        BboxRegressionLoss
)
from src.misc import AttrDict, set_seed, print_table, construct_class
from src.models.model import MMN, MMN_bbox_reg
from src.utils import (
    nms, scores2ds_to_moments, moments_to_iou2ds, moments_to_rescaled_iou2ds, 
    iou2ds_to_iou2d, plot_moments_on_iou2d
)


def append_to_json_file(path, data):
    if os.path.exists(path):
        history = json.load(open(path, 'r'))
    else:
        history = []
    history.append(data)
    json.dump(history, open(path, 'w'), indent=4)


## compute topk similarity score
@torch.no_grad()
def compute_topk_sim_score(
    video_feats: torch.Tensor,      # [B, C, N, N]
    sents_feats: torch.Tensor,      # [S, C]
    num_sentences: torch.Tensor,    # [B]           number of sentences for each video
    num_targets: torch.Tensor,      # [S]           number of targets for each sentence
    iou2d: torch.Tensor,            # [S, N, N]
    iou2ds: torch.Tensor,           # [M, N, N]
    mask2d: torch.Tensor,           # [N, N]
    topk: int=1,
    inter: bool = True,
    intra: bool = True,
):
    device = video_feats.device
    B, C, N, _ = video_feats.shape
    S = num_sentences.sum().cpu().item()
    M = num_targets.sum().cpu().item()
    P = mask2d.long().sum()
    K = topk

    assert iou2d.shape == (S, N, N), f"{iou2d.shape} != {(S, N, N)}"
    assert iou2ds.shape == (M, N, N), f"{iou2ds.shape} != {(M, N, N)}"

    # sentence idx -> video idx
    scatter_s2v = torch.arange(B, device=device).long()
    scatter_s2v = scatter_s2v.repeat_interleave(num_sentences)      # [S]
    # moment idx -> sentence idx
    scatter_m2s = torch.arange(S, device=device).long()
    scatter_m2s = scatter_m2s.repeat_interleave(num_targets)        # [M]
    # moment idx -> video idx
    scatter_m2v = scatter_s2v[scatter_m2s]

    video_feats = video_feats.masked_select(mask2d).view(B, C, -1)  # [B, C, P]
    video_feats = video_feats.permute(0, 2, 1)                      # [B, P, C]
    iou2d = iou2d.masked_select(mask2d).view(S, -1)                 # [S, P]
    iou2ds = iou2ds.masked_select(mask2d).view(M, -1)               # [M, P]

    # normalize for cosine similarity
    video_feats = F.normalize(video_feats.contiguous(), dim=-1)     # [B, P, C]
    sents_feats = F.normalize(sents_feats.contiguous(), dim=-1)     # [S, C]

    inter_topk_sim_single = []
    inter_topk_sim_multi = []
    inter_neg_sim_single = []
    inter_neg_sim_multi = []
    if inter:
        # === inter video
        topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
        topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
        allm_video_feats = video_feats[scatter_m2v]             # [M, P, C]
        topk_video_feats = allm_video_feats.gather(
            dim=1, index=topk_idxs)                             # [M, K, C]
        
        ## inter topk similarity score
        inter_video_pos = torch.mul(
            topk_video_feats,                                   # [M, K, C]
            sents_feats[scatter_m2s].unsqueeze(1)               # [M, 1, C]
        ).sum(dim=-1)                                           # [M, K]
        #inter_topk_sim = torch.sigmoid(10 * inter_video_pos)    # [M, K]
        inter_topk_sim = inter_video_pos
        inter_topk_sim = torch.mean(
            inter_topk_sim, dim=-1, keepdim=False).cpu()        # [M]
        
        ## inter neg similarity score
        inter_video_all = torch.matmul(
                topk_video_feats,                                   # [M, K, C]
                sents_feats.t(),                                    # [C, S]
            )                                                       # [M, K, S]
        mask = ~torch.eye(S, device=device).bool()                  # [S, S]
        inter_video_neg_mask = mask[scatter_m2s].unsqueeze(1)       # [M, 1, S]
        # mean neg sim for each moment
        #inter_neg_sim = torch.sigmoid(10 * inter_video_all)         # [M, K, S]
        inter_neg_sim = inter_video_all
        inter_neg_sim = inter_neg_sim.mul(inter_video_neg_mask)
        sample_neg_num = inter_video_neg_mask.sum(dim=[1, 2]) * K   # [M]
        inter_neg_sim = inter_neg_sim.sum(dim=[1, 2])               # [M] sum all neg
        inter_neg_sim = inter_neg_sim.div(
            sample_neg_num
        ).cpu()                                                     # [M] mean neg sim score

        ## statistics
        shift_t = 0
        for num_t in num_targets:
            if num_t == 1:  ## single-target
                inter_topk_sim_single.append(inter_topk_sim[shift_t: shift_t + num_t]) 
                inter_neg_sim_single.append(inter_neg_sim[shift_t: shift_t + num_t])   
            else:
                inter_topk_sim_multi.append(inter_topk_sim[shift_t: shift_t + num_t])  
                inter_neg_sim_multi.append(inter_neg_sim[shift_t: shift_t + num_t])       
            shift_t += num_t

        ## sometimes a batch may only contains single-target samples, so
        ## inter_topk_sim_multi will be empty list
        if len(inter_topk_sim_single) == 0:
            inter_topk_sim_single.append(0)
        if len(inter_neg_sim_single) == 0:
            inter_neg_sim_single.append(0)
        if len(inter_topk_sim_multi) == 0:
            inter_topk_sim_multi.append(0)
        if len(inter_neg_sim_multi) == 0:
            inter_neg_sim_multi.append(0)    

        inter_topk_sim_single = torch.cat(inter_topk_sim_single)
        inter_neg_sim_single = torch.cat(inter_neg_sim_single)
        inter_topk_sim_multi = torch.cat(inter_topk_sim_multi)
        inter_neg_sim_multi = torch.cat(inter_neg_sim_multi)
    else:
        inter_topk_sim = torch.zeros(1)
        inter_neg_sim = torch.zeros(1)
            

    ## always record intra sim 
    shift = 0
    combinations = []
    scatter_e2s = []
    for i, num in enumerate(num_targets):
        if num > 1: ## multi-target
            pairs = torch.ones(
                num * K, num * K, device=device).fill_diagonal_(0).nonzero()      # [num * K * num * K, 2]
            
            combinations.append(pairs + shift)
            scatter_e2s.append(torch.ones(len(pairs), device=device) * i)
        shift += num * K
        
    # E: number of (E)numerated positive pairs
    ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
    scatter_e2s = torch.cat(scatter_e2s, dim=0).long()      # [E]  ex.[0, 0, 0, 1, 1, 1...]
    assert (ref_idx < M * K).all()
    assert (pos_idx < M * K).all()

    ## intra topk sim
    pos_video_feats = topk_video_feats.reshape(M * K, C)    # [M * K, C]
    intra_video_pos = torch.mul(
        pos_video_feats[ref_idx],                           # [E, C]
        pos_video_feats[pos_idx],                           # [E, C]
    ).sum(dim=1)                                            # [E]
    #intra_topk_sim = torch.sigmoid(
    #    10 * intra_video_pos).cpu()                         # [E]
    intra_topk_sim = intra_video_pos.cpu()

    ## intra neg sim
    intra_video_all = torch.mul(
            topk_video_feats.unsqueeze(2),                      # [M, K, 1, C]
            video_feats[scatter_m2v].unsqueeze(1),              # [M, 1, P, C]
        ).sum(dim=-1).reshape(M * K, -1)                        # [M * K, P]
    intra_video_all = intra_video_all[ref_idx]                  # [E, P]
    intra_video_neg_mask = iou2d <= 0.5                         # [S, P]
    intra_video_neg_mask = intra_video_neg_mask[scatter_e2s]    # [E, P]

    ## mean neg sim for each intra pair
    #intra_neg_sim = torch.sigmoid(10 * intra_video_all)         # [E, P]
    ## cos sim [-1, 1]
    intra_neg_sim = intra_video_all
    intra_neg_sim = intra_neg_sim.mul(intra_video_neg_mask)     # [E, P]
    
    sample_neg_num = intra_video_neg_mask.sum(dim=-1).squeeze() # [E] for computing mean
    intra_neg_sim = intra_neg_sim.sum(dim=-1).squeeze()         # [E]
    intra_neg_sim = intra_neg_sim.div(
        sample_neg_num
    ).cpu()                                                     # [E]



    return inter_topk_sim, inter_topk_sim_single, inter_topk_sim_multi, \
           inter_neg_sim, inter_neg_sim_single, inter_neg_sim_multi, \
           intra_topk_sim, intra_neg_sim


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
    #sim_dict = defaultdict(list)
    ## sample idx count
    sample_idx_count = 0
    for batch, _ in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            #*_, scores2ds, mask2d = model(**batch)
            video_feats, sents_feats, logits2d, scores2ds, mask2d = model(**batch)
        
        ## pred
        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d) ## out_moments: [S, P, 2]
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)
        
        ## GT
        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)       # [M, N, N]
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])                    # [S, N, N], separate to combined  

        ## ploting batch
        result_path = os.path.join(config.logdir, config.result_plot_path)       # logs/xxx/result_plot/
        shift_gt = 0
        shift_pred = 0
        ## scores2ds: [S, N, N]
        for batch_idx, (scores2d, gt_iou2d) in enumerate(zip(scores2ds.cpu(), iou2d.cpu())):
            ## Gt
            num_gt_targets = batch['num_targets'][batch_idx]
            moments = batch['tgt_moments'][shift_gt: shift_gt + num_gt_targets]
            moments = (moments * config.num_clips).round().long()           
            ## nms(pred)
            num_proposals = pred_moments_batch['num_proposals'][batch_idx]
            nms_moments = pred_moments_batch["out_moments"][shift_pred: shift_pred + num_proposals] ## [num_props, 2]
            nms_moments = (nms_moments * config.num_clips).round().long()   # Pred

            if dist.is_main() and sample_idx_count % 100 == 0 and epoch > 0:
                plot_path = os.path.join(result_path, f"sample_{sample_idx_count}_epoch_{epoch}.jpg")
                plot_moments_on_iou2d(
                     gt_iou2d, scores2d, nms_moments, plot_path)

            shift_gt = shift_gt + num_gt_targets
            shift_pred = shift_pred + num_proposals
            sample_idx_count += dist.get_world_size()

        '''
        ## record topk similarity score
        (
            inter_topk_sim, 
            inter_topk_sim_single, 
            inter_topk_sim_multi, 
            inter_neg_sim,
            inter_neg_sim_single, 
            inter_neg_sim_multi,
            intra_topk_sim,
            intra_neg_sim,
        ) = compute_topk_sim_score(
            video_feats=video_feats,
            sents_feats=sents_feats,
            num_sentences=batch['num_sentences'],
            num_targets=batch['num_targets'],
            iou2d=iou2d,
            iou2ds=iou2ds,
            mask2d=mask2d,
            topk=config.pos_topk,
            inter=config.inter,
            intra=config.intra,  
        )
        ## inter topk sim
        sim_dict['topk_sim/inter_all'].append(inter_topk_sim)
        sim_dict['topk_sim/inter_single'].append(inter_topk_sim_single)
        sim_dict['topk_sim/inter_multi'].append(inter_topk_sim_multi)
        ## inter neg sim
        sim_dict['neg_sim/inter_all'].append(inter_neg_sim)
        sim_dict['neg_sim/inter_single'].append(inter_neg_sim_single)
        sim_dict['neg_sim/inter_multi'].append(inter_neg_sim_multi)
        ## intra
        sim_dict['topk_sim/intra'].append(intra_topk_sim)
        sim_dict['neg_sim/intra'].append(intra_neg_sim)
        '''
        
    #sim_dict = {key: torch.cat(value).mean() for key, value in sim_dict.items()}

    #return pred_moments, true_moments, sim_dict
    return pred_moments, true_moments


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_conf_fn: torch.nn.Module,
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
    #sim_dict = defaultdict(list)
    for batch, _ in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])

        video_feats, sents_feats, logits2d, scores2ds, mask2d = model(**batch)
        loss_iou = loss_conf_fn(logits2d, iou2d, mask2d)
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
            
            if epoch < config.intra_start_epoch:
                loss_contrastive = (loss_inter_video * config.inter_weight + 
                                    loss_inter_query * config.inter_weight + 
                                    loss_intra_video * 0)    
            else:
                loss_contrastive = (loss_inter_video * config.inter_weight + 
                                    loss_inter_query * config.inter_weight + 
                                    loss_intra_video * config.intra_weight)
            
            
        else:
            loss_inter_video = torch.zeros((), device=device)
            loss_inter_query = torch.zeros((), device=device)
            loss_intra_video = torch.zeros((), device=device)
            loss_contrastive = torch.zeros((), device=device)


        loss = 0
        loss += loss_iou * config.iou_weight
        if epoch <= config.only_iou_epoch:            
            loss += loss_contrastive * config.contrastive_weight
        else:
            loss += loss_contrastive * config.contrastive_weight * config.cont_weight_step ## scale down cont loss

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
        losses['loss/total'].append(loss.cpu())
        losses['loss/iou'].append(loss_iou.cpu())
        losses['loss/contrastive'].append(loss_contrastive.cpu())
        losses['loss/inter_video'].append(loss_inter_video.cpu())
        losses['loss/inter_query'].append(loss_inter_query.cpu())
        losses['loss/intra_video'].append(loss_intra_video.cpu())
        
        '''
        ## record topk similarity score
        (
            inter_topk_sim, 
            inter_topk_sim_single, 
            inter_topk_sim_multi, 
            inter_neg_sim,
            inter_neg_sim_single, 
            inter_neg_sim_multi,
            intra_topk_sim,
            intra_neg_sim,
        ) = compute_topk_sim_score(
            video_feats=video_feats,
            sents_feats=sents_feats,
            num_sentences=batch['num_sentences'],
            num_targets=batch['num_targets'],
            iou2d=iou2d,
            iou2ds=iou2ds,
            mask2d=mask2d,
            topk=config.pos_topk,
            inter=config.inter,
            intra=config.intra,  
        )
        ## inter topk sim
        sim_dict['topk_sim/inter_all'].append(inter_topk_sim)
        sim_dict['topk_sim/inter_single'].append(inter_topk_sim_single)
        sim_dict['topk_sim/inter_multi'].append(inter_topk_sim_multi)
        ## inter neg sim
        sim_dict['neg_sim/inter_all'].append(inter_neg_sim)
        sim_dict['neg_sim/inter_single'].append(inter_neg_sim_single)
        sim_dict['neg_sim/inter_multi'].append(inter_neg_sim_multi)
        ## intra
        sim_dict['topk_sim/intra'].append(intra_topk_sim)
        sim_dict['neg_sim/intra'].append(intra_neg_sim)
        '''
        
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
    #sim_dict = {key: torch.cat(value).mean() for key, value in sim_dict.items()}

    #return pred_moments, true_moments, losses, sim_dict
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
    loss_conf_fn = ScaledIoULoss(config.min_iou, config.max_iou)
    ## testing focal loss
    '''
    loss_conf_fn = ScaledIoUFocalLoss(
                        min_iou=config.min_iou,
                        max_iou=config.max_iou,
                        scale=10,
                        alpha=config.alpha,
                        gamma=config.gamma,
                    )
    '''
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

    ## mAP metric groups
    mAP_keys_group_1 = ["avg_mAP", "mAP@0.50", "mAP@0.75", 
                        "single_avg_mAP", "single_mAP@0.50", "single_mAP@0.75",
                        "multi_avg_mAP", "multi_mAP@0.50", "multi_mAP@0.75",]
    mAP_keys_group_2 = ["short_avg_mAP", "short_mAP@0.50", "short_mAP@0.75",
                        "medium_avg_mAP", "medium_mAP@0.50", "medium_mAP@0.75",
                        "long_avg_mAP", "long_mAP@0.50", "long_mAP@0.75"]

    # evaluate test set before training to get initial recall
    #os.makedirs(config.logdir, exist_ok=False)
    os.makedirs(config.logdir, exist_ok=True)
    result_path = os.path.join(config.logdir, config.result_plot_path)
    #os.makedirs(result_path, exist_ok=False)
    os.makedirs(result_path, exist_ok=True)

    test_pred_moments, test_true_moments = test_epoch(model, test_loader, 0, config)

    if dist.is_main():
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

        ## split mAPs table, too long to print on terminal
        test_mAPs_group_1 = {key: test_mAPs[key] for key in mAP_keys_group_1}
        test_mAPs_group_2 = {key: test_mAPs[key] for key in mAP_keys_group_2}
        # print to terminal
        print_table(epoch=0, rows={'test': test_recall})
        print_table(epoch=0, rows={"test": test_mAPs_group_1})
        print_table(epoch=0, rows={"test": test_mAPs_group_2})
        best_recall = test_recall
        best_mAPs = test_mAPs
        '''
        ## record topk_sim
        for name, value in test_sim.items():
            test_writer.add_scalar(name, value, 0)
        print_table(epoch=0, rows={'test': test_sim})
        '''
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
            model, train_loader, optimizer, loss_conf_fn, loss_con_fn, epoch,
            config)
        test_pred_moments, test_true_moments = test_epoch(
            model, test_loader, epoch, config)
        scheduler.step()

        if dist.is_main():
            train_writer.add_scalar(
                "lr/base", optimizer.param_groups[0]["lr"], epoch)
            train_writer.add_scalar(
                "lr/bert", optimizer.param_groups[1]["lr"], epoch)

            for name, value in train_losses.items():
                train_writer.add_scalar(name, value, epoch)
            
            # evaluate train set
            '''
            ## record topk_sim
            for name, value in train_sim.items():
                train_writer.add_scalar(name, value, epoch)
            '''
            train_recall = calculate_recall(
                train_pred_moments, train_true_moments,
                config.recall_Ns, config.recall_IoUs)
            train_mAPs = calculate_mAPs(train_pred_moments, train_true_moments)
            for name, value in train_recall.items():
                train_writer.add_scalar(f'recall/{name}', value, epoch)
            for name, value in train_mAPs.items():
                train_writer.add_scalar(f'mAP/{name}', value, epoch)

            # evaluate test set
            '''
            ## record topk_sim
            for name, value in test_sim.items():
                test_writer.add_scalar(name, value, epoch)
            '''
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

            ## split mAPs table, too long to print on terminal
            train_mAPs_group_1 = {key: train_mAPs[key] for key in mAP_keys_group_1}
            train_mAPs_group_2 = {key: train_mAPs[key] for key in mAP_keys_group_2}
            test_mAPs_group_1 = {key: test_mAPs[key] for key in mAP_keys_group_1}
            test_mAPs_group_2 = {key: test_mAPs[key] for key in mAP_keys_group_2}
            
            # print to terminal
            print_table(epoch, {"train": train_recall, "test": test_recall})
            print_table(epoch, {"train": train_mAPs_group_1, "test": test_mAPs_group_1})
            print_table(epoch, {"train": train_mAPs_group_2, "test": test_mAPs_group_2})
            #print_table(epoch, {"train": train_sim, "test": test_sim})

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


def test_epoch_bbox_reg(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    config: AttrDict,
):
    device = dist.get_device()
    model.eval()
    pred_moments = []
    true_moments = []
    ## sample idx count
    sample_idx_count = 0
    for batch, _ in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            (video_feats, sents_feats, logits2d, 
            scores2ds, mask2d, bbox, 
            start_offset, end_offset, scores1ds) = model(**batch)
            
        # ## out_moments is default proposal moments
        # out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d) ## out_moments: [S, P, 2]
        
        # ## add bbox_offset: [S, 2, N, N] to out_moments: [S, P, 2]
        # S, N, _ = scores2ds.shape
        # bbox_offset_1ds = bbox_offset.masked_select(mask2d).view(S, 2, -1)  # [S, 2, P]
        # bbox_offset_1ds = bbox_offset_1ds.permute(0, 2, 1)                  # [S, P, 2]
        # out_moments = out_moments + bbox_offset_1ds.tanh() * 1/N            # [S, P, 2]
        # out_moments = out_moments + bbox_offset_1ds.tanh()                  # [S, P, 2]

        # ## clamp start and end
        # out_moments = torch.clamp(out_moments, min=0, max=1)                # [S, P, 2]
        
        pred_moments_batch = nms(bbox, scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)
        
        ## GT
        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

        iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)       # [M, N, N]
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])                    # [S, N, N], separate to combined  

        ## ploting batch
        result_path = os.path.join(config.logdir, config.result_plot_path)       # logs/xxx/result_plot/
        shift_gt = 0
        shift_pred = 0
        ## scores2ds: [S, N, N]
        for batch_idx, (scores2d, gt_iou2d) in enumerate(zip(scores2ds.cpu(), iou2d.cpu())):
            ## Gt
            num_gt_targets = batch['num_targets'][batch_idx]
            ## nms(pred)
            num_proposals = pred_moments_batch['num_proposals'][batch_idx]
            nms_moments = pred_moments_batch["out_moments"][shift_pred: shift_pred + num_proposals] ## [num_props, 2]
            nms_moments = (nms_moments * config.num_clips).round().long()   # Pred

            if dist.is_main() and sample_idx_count % 200 == 0 and epoch > 0:
                plot_path = os.path.join(result_path, f"sample_{sample_idx_count}_epoch_{epoch}.jpg")
                plot_moments_on_iou2d(gt_iou2d, scores2d, nms_moments, plot_path)

            shift_gt = shift_gt + num_gt_targets
            shift_pred = shift_pred + num_proposals
            sample_idx_count += dist.get_world_size()


    return pred_moments, true_moments


def train_epoch_bbox_reg(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_con_fn: torch.nn.Module,
    loss_conf_fn: torch.nn.Module,
    loss_bbox_reg_fn: torch.nn.Module,
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
        #iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips) # [M, N, N] for each target
        iou2ds = moments_to_rescaled_iou2ds(batch['tgt_moments'], config.num_clips) # [M, N, N] for each target
        iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets']) # [S, N, N] for each sentence

        ## error
        (video_feats, sents_feats, logits2d, 
         scores2ds, mask2d, bbox, 
         start_offset, end_offset, scores1ds) = model(**batch)
        
        ## nms
        ## out_moments.clone().detach() create a copy of out_moments that doesn't require grads
        pred_moments_batch = nms(bbox.clone().detach(), scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

        ## testing ScaledIoUFocaLoss with Rescaled IoU
        ## pass in epoch for curriculum learning
        if epoch < 6:
            loss_conf = loss_conf_fn(logits2d, iou2d, mask2d, rescale=False)
        else:
            loss_conf = loss_conf_fn(logits2d, iou2d, mask2d, rescale=True)
            
        ## bbox regression score
        loss_bbox_reg = loss_bbox_reg_fn(
                            start_offset=start_offset,
                            end_offset=end_offset, 
                            tgt_moments=batch['tgt_moments'], 
                            num_targets=batch['num_targets'],
                            iou2ds=iou2ds,
                            mask2d=mask2d,
                        )
      
        
        ## contrastive loss
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
            
            if epoch < config.intra_start_epoch:
                loss_contrastive = (loss_inter_video * config.inter_weight + 
                                    loss_inter_query * config.inter_weight + 
                                    loss_intra_video * 0)    
            else:
                loss_contrastive = (loss_inter_video * config.inter_weight + 
                                    loss_inter_query * config.inter_weight + 
                                    loss_intra_video * config.intra_weight)
                  
        else:
            loss_inter_video = torch.zeros((), device=device)
            loss_inter_query = torch.zeros((), device=device)
            loss_intra_video = torch.zeros((), device=device)
            loss_contrastive = torch.zeros((), device=device)

        loss = 0
        ## confidence loss
        loss += loss_conf * config.iou_weight
        ## bbox regression loss
        loss += loss_bbox_reg * config.bbox_reg_weight
        ## contrastive loss
        if epoch <= config.only_iou_epoch:            
            loss += loss_contrastive * config.contrastive_weight
        else:
            ## scale down cont loss
            loss += loss_contrastive * config.contrastive_weight * config.cont_weight_step 

        loss.backward()
        if config.grad_clip > 0:
            clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # save loss to tensorboard
        losses['loss/total'].append(loss.cpu())
        losses['loss/conf'].append(loss_conf.cpu())
        losses['loss/bbox_reg'].append(loss_bbox_reg.cpu())
        losses['loss/contrastive'].append(loss_contrastive.cpu())
        losses['loss/inter_video'].append(loss_inter_video.cpu())
        losses['loss/inter_query'].append(loss_inter_query.cpu())
        losses['loss/intra_video'].append(loss_intra_video.cpu())
        
        # update progress bar
        pbar.set_postfix_str(", ".join([
            f"loss: {loss.item():.2f}",
            f"conf: {loss_conf.item():.2f}",
            f"bbox_reg: {loss_bbox_reg.item():.2f}",
            "[inter]",
            f"video: {loss_inter_video.item():.2f}",
            f"query: {loss_inter_query.item():.2f}",
            "[intra]",
            f"video: {loss_intra_video.item():.2f}",
        ]))
    pbar.close()

    losses = {key: torch.stack(value).mean() for key, value in losses.items()}

    return pred_moments, true_moments, losses

## Bbox regression test
def training_loop_bbox_reg(config: AttrDict):
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
    loss_con_fn = ContrastiveLoss(
        T_v=config.tau_video,
        T_q=config.tau_query,
        neg_iou=config.neg_iou,
        pos_topk=config.pos_topk,
        margin=config.margin,
        inter=config.inter,
        intra=config.intra,
    )
    
    ## iou_threshold is for selecting foreground
    #loss_conf_fn = ConfidenceLoss(iou_threshold=config.iou_threshold)
    loss_conf_fn = ScaledIoULoss(config.min_iou, config.max_iou)
    ## testing Focal loss
    # loss_conf_fn = ScaledIoUFocalLoss(
    #                     min_iou=config.min_iou,
    #                     max_iou=config.max_iou,
    #                     scale=10,
    #                     alpha=config.alpha,
    #                     gamma=config.gamma,
    #                 )
    loss_bbox_reg_fn = BboxRegressionLoss(iou_threshold=config.iou_threshold)

    # model
    model_local = MMN_bbox_reg(
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

    ## mAP metric groups
    mAP_keys_group_1 = ["avg_mAP", "mAP@0.50", "mAP@0.75", 
                        "single_avg_mAP", "single_mAP@0.50", "single_mAP@0.75",
                        "multi_avg_mAP", "multi_mAP@0.50", "multi_mAP@0.75",]
    mAP_keys_group_2 = ["short_avg_mAP", "short_mAP@0.50", "short_mAP@0.75",
                        "medium_avg_mAP", "medium_mAP@0.50", "medium_mAP@0.75",
                        "long_avg_mAP", "long_mAP@0.50", "long_mAP@0.75"]
    
    ## create result_plot folder
    #os.makedirs(config.logdir, exist_ok=False)
    os.makedirs(config.logdir, exist_ok=True)
    result_path = os.path.join(config.logdir, config.result_plot_path)
    #os.makedirs(result_path, exist_ok=False)
    os.makedirs(result_path, exist_ok=True)
    test_pred_moments, test_true_moments = test_epoch_bbox_reg(
                                                model, test_loader, 0, config
                                            )
    

    # evaluate test set before training to get initial recall
    if dist.is_main():
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

        ## split mAPs table, too long to print on terminal
        test_mAPs_group_1 = {key: test_mAPs[key] for key in mAP_keys_group_1}
        test_mAPs_group_2 = {key: test_mAPs[key] for key in mAP_keys_group_2}
        
        # print to terminal
        print_table(epoch=0, rows={'test': test_recall})
        print_table(epoch=0, rows={"test": test_mAPs_group_1})
        print_table(epoch=0, rows={"test": test_mAPs_group_2})
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

        train_pred_moments, train_true_moments, train_losses = train_epoch_bbox_reg(
                model, train_loader, optimizer, loss_con_fn, 
                loss_conf_fn, loss_bbox_reg_fn, epoch, config
            )
        test_pred_moments, test_true_moments = test_epoch_bbox_reg(
            model, test_loader, epoch, config)
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
            
            ## split mAPs table, too long to print on terminal
            train_mAPs_group_1 = {key: train_mAPs[key] for key in mAP_keys_group_1}
            train_mAPs_group_2 = {key: train_mAPs[key] for key in mAP_keys_group_2}
            test_mAPs_group_1 = {key: test_mAPs[key] for key in mAP_keys_group_1}
            test_mAPs_group_2 = {key: test_mAPs[key] for key in mAP_keys_group_2}
            
            # print to terminal
            print_table(epoch, {"train": train_recall, "test": test_recall})
            print_table(epoch, {"train": train_mAPs_group_1, "test": test_mAPs_group_1})
            print_table(epoch, {"train": train_mAPs_group_2, "test": test_mAPs_group_2})


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
