import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import (
    batch_iou, batch_diou, plot_mask_and_gt, scores2ds_to_moments
)

class ScaledIoULoss(nn.Module):
    def __init__(self, min_iou, max_iou):
        super().__init__()
        self.min_iou = min_iou
        self.max_iou = max_iou

    def linear_scale(self, iou: torch.Tensor):
        return iou.sub(self.min_iou).div(self.max_iou - self.min_iou).clamp(0, 1)

    def forward(
        self,
        logits2d: torch.Tensor,     # [S, N, N]
        iou2d: torch.Tensor,        # [S, N, N]
        mask2d: torch.Tensor,       # [N, N]
        rescale: bool,
    ):
        """
            B: (B)atch size
            N: (N)um clips
            S: number of (S)entences
            P: number of (P)roposals = number of 1 in mask2d
        """
        S, _, _ = logits2d.shape
        assert logits2d.shape == iou2d.shape, f"{logits2d.shape} != {iou2d.shape}"
        logits1d = logits2d.masked_select(mask2d).view(S, -1)   # [S, P]
        iou1d = iou2d.masked_select(mask2d).view(S, -1)         # [S, P]
        iou1d = self.linear_scale(iou1d)                        # [S, P]
        #loss = F.binary_cross_entropy_with_logits(logits1d, iou1d)
        loss = F.binary_cross_entropy_with_logits(logits1d, iou1d, reduction='none')    # [S, P]
        
        if rescale:
            ## different loss weight for different proposal length 
            moments, _ = scores2ds_to_moments(logits2d, mask2d)     # [S, P, 2]
            moments_len = moments[:, :, 1] - moments[:, :, 0]       # [S, P]
            weight_scale = 1 / moments_len                          # [S, P]
            loss = weight_scale * loss                              # [S, P]

        return loss.mean()
    
class ScaledIoUFocalLoss(nn.Module):
    def __init__(
        self, 
        min_iou: float = 0.5, 
        max_iou: float = 1, 
        scale: float = 10,
        alpha: float = 0.25,
        gamma: float = 2,
    ):
        super().__init__()
        self.min_iou = min_iou
        self.max_iou = max_iou
        self.scale = scale
        self.alpha = alpha
        self.gamma = gamma

    def linear_scale(self, iou: torch.Tensor):
        return iou.sub(self.min_iou).div(self.max_iou - self.min_iou).clamp(0, 1)

    def forward(
        self,
        logits2d: torch.Tensor,     # [S, N, N]
        iou2d: torch.Tensor,        # [S, N, N]
        mask2d: torch.Tensor,       # [N, N]
        rescale: bool,  
    ):
        """
            B: (B)atch size
            N: (N)um clips
            S: number of (S)entences
            P: number of (P)roposals = number of 1 in mask2d
            x: logits
            p: predicted probability
            y: label
        """
        S, _, _ = logits2d.shape
        assert logits2d.shape == iou2d.shape, f"{logits2d.shape} != {iou2d.shape}"
        x = logits2d.masked_select(mask2d).view(S, -1)          # [S, P]
        ## sigmoid is not numerically stable, sometimes get nan
        p = torch.sigmoid(x)                       # [S, P]    
        iou1d = iou2d.masked_select(mask2d).view(S, -1)         # [S, P]
        y = self.linear_scale(iou1d)                            # [S, P]
        
        ## with alpha
        #pos_weight = self.alpha * torch.pow((1 - p), self.gamma) * y
        #neg_weight = (1 - self.alpha) * torch.pow(p, self.gamma) * (1 - y)

        # no alpha
        pos_weight = torch.pow((1 - p), self.gamma) * y
        neg_weight = torch.pow(p, self.gamma) * (1 - y)

        loss = neg_weight * x + \
               (pos_weight + neg_weight) * (torch.log1p(torch.exp(-torch.abs(x))) + F.relu(-x))   
        
        if rescale:
            ## different loss weight for different proposal length 
            moments, _ = scores2ds_to_moments(logits2d, mask2d)     # [S, P, 2]
            moments_len = moments[:, :, 1] - moments[:, :, 0]       # [S, P]
            #weight_scale = torch.exp( torch.pow(1-moments_len, 2) ) # [S, P]
            weight_scale = 1 / moments_len                          # [S, P]
            loss = weight_scale * loss
            
        return loss.mean()

## fore/background binary classification
class ConfidenceLoss(nn.Module):
    def __init__(
        self, 
        iou_threshold: float = 0.75
    ):
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(
            self,
            logits2d: torch.Tensor,     # [S, N, N]
            iou2d: torch.Tensor,        # [S, N, N]
            iou2ds: torch.Tensor,       # [M, N, N]
            mask2d: torch.Tensor,       # [N, N]
            num_targets: torch.Tensor,  # [S] number of targets for each sentence
    ):
        """
            B: (B)atch size
            N: (N)um clips
            S: number of (S)entences
            P: number of (P)roposals = number of 1 in mask2d
        """
        S, _, _ = logits2d.shape
        M = iou2ds.shape[0]
        assert logits2d.shape == iou2d.shape, f"{logits2d.shape} != {iou2d.shape}"
        device = logits2d.device
        logits1d = logits2d.masked_select(mask2d).view(S, -1)   # [S, P]
        iou1d = iou2d.masked_select(mask2d).view(S, -1)         # [S, P]
        P = iou1d.shape[1]
        
        ## select top1 for each targets
        iou1ds = iou2ds.masked_select(mask2d).view(M, -1)       # [M, P]
        top1_idxs = iou1ds.topk(1, dim=1)[1]                    # [M, 1]
        target = torch.zeros(S, P, device=device)               # [S, P], backgrounds are 0
        target_count = 0
        for sent_idx, num_target in enumerate(num_targets):
            for i in range(num_target):
                top1_prop_idx = int(top1_idxs[target_count + i])
                target[sent_idx, top1_prop_idx] = 1
            target_count += num_target
        
        ## select proposals with > IoU threshold as foreground
        target[iou1d > self.iou_threshold] = 1                  # foregrounds are 1
        
        loss = F.binary_cross_entropy_with_logits(logits1d, target)
        
        return loss

## Bbox IoU loss between pred bbox and gt bbox
class BboxIoULoss(nn.Module):
    def __init__(
        self,
        topk: int = 3,
        iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.topk = topk
        self.iou_threshold = iou_threshold
    
    def forward(
        self,
        out_moments: torch.Tensor,  # [S, P, 2]
        tgt_moments: torch.Tensor,  # [M, 2]
        num_targets: torch.Tensor,  # [S]
        iou2ds: torch.Tensor,       # [M, N, N]
        mask2d: torch.Tensor,       # [N, N]
    ):
        device = iou2ds.device
        M, N, _ = iou2ds.shape
        S, P, _ = out_moments.shape
        # moment idx -> sentence idx
        scatter_m2s = torch.arange(S, device=device).long()
        scatter_m2s = scatter_m2s.repeat_interleave(num_targets)    # [M]

        out_moments = out_moments[scatter_m2s]                      # [M, P, 2]
        ## compute iou with tgt_moments
        ## [M, P, 2],  [M, 1, 2] -> [M, P, 1]
        bbox_diou = batch_diou(out_moments, 
                               tgt_moments.unsqueeze(1))            # [M, P, 1]
        bbox_diou = bbox_diou.squeeze()                             # [M, P]
        
        ## create mask to find responsible anchors for each target
        ## TODO choose top-k and GT IoU > threshold proposals
        iou1ds = iou2ds.masked_select(mask2d).view(M, -1)           # [M, P]
        topk_idxs = iou1ds.topk(self.topk, dim=1)[1]                # [M, topk]
        target_mask = torch.zeros(M, P, device=device)              # [M, P]
        for target_idx, sample_topk_idx in enumerate(topk_idxs):
            for idx in sample_topk_idx:
                target_mask[target_idx, idx] = 1
        ## select anchors with GT_IoU > IoU_threshold
        target_mask[iou1ds > self.iou_threshold] = 1                # foregrounds are 1
        
        loss = 1 - bbox_diou                                         # [M, P]
        ## only compute anchor loss whose target_mask = 1
        loss = (loss * target_mask).sum() / target_mask.sum()                        

        return loss


## for regression offset
class BboxRegressionLoss(nn.Module):
    def __init__(
        self,
        topk: int = 3,
        iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.topk = topk
        self.iou_threshold = iou_threshold
    
    def forward(
        self,
        start_offset: torch.Tensor, # [S, P]
        end_offset: torch.Tensor,   # [S, P]        
        tgt_moments: torch.Tensor,  # [M, 2]
        num_targets: torch.Tensor,  # [S]
        iou2ds: torch.Tensor,       # [M, N, N]
        mask2d: torch.Tensor,       # [N, N]
    ):
        device = iou2ds.device
        M, N, _ = iou2ds.shape
        S, P = start_offset.shape
        # moment idx -> sentence idx
        scatter_m2s = torch.arange(S, device=device).long()
        scatter_m2s = scatter_m2s.repeat_interleave(num_targets)    # [M]

        start_offset = start_offset[scatter_m2s]                    # [M, P]
        end_offset = end_offset[scatter_m2s]                        # [M, P]

        ## create mask to find responsible anchors for each target
        ## TODO choose top-k and GT IoU > threshold proposals
        iou1ds = iou2ds.masked_select(mask2d).view(M, -1)           # [M, P]
        topk_idxs = iou1ds.topk(self.topk, dim=1)[1]                # [M, topk]
        target_mask = torch.zeros(M, P, device=device)              # [M, P]
        for target_idx, sample_topk_idx in enumerate(topk_idxs):
            for idx in sample_topk_idx:
                target_mask[target_idx, idx] = 1
        ## select anchors with GT_IoU > IoU_threshold
        target_mask[iou1ds > self.iou_threshold] = 1                # foregrounds are 1
        
        ## compute the GT for start/end offsets
        moments, _ = scores2ds_to_moments(iou2ds, mask2d)           # [S, P, 2]
        ## [M, 1] - [M, P]
        start_offset_target = tgt_moments[:, 0].unsqueeze(-1) - moments[:, :, 0]    # [M, P]
        end_offset_target = tgt_moments[:, 1].unsqueeze(-1) - moments[:, :, 1]      # [M, P]
        loss = torch.abs(start_offset - start_offset_target) + \
                torch.abs(end_offset - end_offset_target)

        ## only compute anchor loss whose target_mask = 1
        loss = (loss * target_mask).sum() / target_mask.sum()                        
        
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        T_v: float = 0.1,
        T_q: float = 0.1,
        neg_iou: float = 0.5,
        pos_topk: int = 1,
        margin: float = 0,
        inter: bool = True,
        intra: bool = False,
    ):
        super().__init__()
        self.T_v = T_v              # 0.1
        self.T_q = T_q              # 0.1
        self.neg_iou = neg_iou      # 0.5
        self.pos_topk = pos_topk    # 1
        self.margin = margin
        self.inter = inter
        self.intra = intra

    def log_cross_entropy(
        self,
        pos_score: torch.Tensor,                # [...]
        all_score: torch.Tensor,                # [..., Number_of_samples]
        neg_mask: torch.Tensor,                 # [..., Number_of_samples] 1 -> neg, 0 -> pos
        t: float,
        m: float = 0,
    ):
        pos_exp = (pos_score - m).div(t).exp()                          # [...]
        neg_exp_sum = all_score.div(t).exp().mul(neg_mask).sum(dim=-1)  # [...]
        all_exp_sum = pos_exp + neg_exp_sum                             # [...]
        loss = -((pos_score - m).div(t) - torch.log(all_exp_sum))
        return loss.mean()

    def forward(
        self,
        video_feats: torch.Tensor,      # [B, C, N, N]
        sents_feats: torch.Tensor,      # [S, C]
        num_sentences: torch.Tensor,    # [B]           number of sentences for each video
        num_targets: torch.Tensor,      # [S]           number of targets for each sentence
        iou2d: torch.Tensor,            # [S, N, N]
        iou2ds: torch.Tensor,           # [M, N, N]
        mask2d: torch.Tensor,           # [N, N]
    ):
        """
            B: (B)atch size
            C: (C)hannel
            N: (N)um clips
            S: number of (S)entences
            M: number of (M)oments
            P: number of (P)roposals = the number 1 in mask2d
        """
        device = video_feats.device
        B, C, N, _ = video_feats.shape
        S = num_sentences.sum().cpu().item()
        M = num_targets.sum().cpu().item()
        P = mask2d.long().sum()
        K = self.pos_topk

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

        if self.inter:
            # === inter video (topk proposal -> all sentences)
            topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
            topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
            allm_video_feats = video_feats[scatter_m2v]             # [M, P, C]
            topk_video_feats = allm_video_feats.gather(
                dim=1, index=topk_idxs)                             # [M, K, C]

            inter_video_pos = torch.mul(
                topk_video_feats,                                   # [M, K, C]
                sents_feats[scatter_m2s].unsqueeze(1)               # [M, 1, C]
            ).sum(dim=-1)                                           # [M, K]

            inter_video_all = torch.matmul(
                topk_video_feats,                                   # [M, K, C]
                sents_feats.t(),                                    # [C, S]
            )                                                       # [M, K, S]
            mask = ~torch.eye(S, device=device).bool()              # [S, S]
            inter_video_neg_mask = mask[scatter_m2s].unsqueeze(1)   # [M, 1, S]

            loss_inter_video = self.log_cross_entropy(
                inter_video_pos,                                    # [M, K]
                inter_video_all,                                    # [M, K, S]
                inter_video_neg_mask,                               # [M, 1, S]
                self.T_v,
                self.margin,
            )
            
        else:
            loss_inter_video = torch.tensor(0., device=device)


        if self.inter:
            # === inter query (sentence -> all video proposals)
            inter_query_pos = inter_video_pos                       # [M, K]

            inter_query_all = torch.mm(
                sents_feats,                                        # [S, C]
                video_feats.view(-1, C).t(),                        # [C, B * P]
            ).unsqueeze(1)                                          # [S, 1, B * P]

            pos_mask = torch.eye(B, device=device).bool()           # [B, B]
            pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
            pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
            pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
            pos_mask = pos_mask[scatter_s2v]                        # [S, B * P]
            assert pos_mask.long().sum(dim=-1).eq(P).all()

            s2v_pos_mask = iou2d > self.neg_iou                     # [S, P]
            
            '''
            ###### for top1 pos sample selection
            ## to do: at least choose top 1 for each target as pos
            ##        otherwise tiny clip (iou < 0.5) will be considered neg
            top1_idxs = iou2ds.topk(1, dim=1)[1]                    # [M, 1]
            top1_pos_mask = torch.zeros(S, P, device=device)
            target_count = 0
            for idx, num_target in enumerate(num_targets):          # [S]
                for i in range(num_target):
                    top1_prop_idx = int(top1_idxs[target_count + i])
                    top1_pos_mask[idx, top1_prop_idx] = 1             
                target_count += num_target
            ## top1_pos_mask:   [S, P], top1 proposal is 1, others are 0
            ## element-wise or, make sure that top1 proposal with IoU < 0.5 is also pos sample
            s2v_pos_mask = torch.logical_or(s2v_pos_mask, top1_pos_mask)    # [S, P]
            ######
            '''
            
            ## for indexing
            backup_pos_mask = pos_mask.clone()                      # [S, B * P]
            pos_mask[backup_pos_mask] = s2v_pos_mask.view(-1)       # [S, B * P] 
            inter_query_neg_mask = ~pos_mask.unsqueeze(1)           # [S, 1, B * P]
            
            
            ###### for neg sample selection
            '''
            ## don't choose lower-left corner as neg samples
            top1_position = torch.zeros(M, P, device=device)
            top1_position[range(M), top1_idxs.squeeze()] = 1        # [M, P] ## top1 = 1,  others = 0
            top1_map = torch.zeros(M, N * N, device=device)
            top1_map[mask2d.view(1, -1).repeat(M, 1)] = top1_position.view(-1)
            top1_map = top1_map.reshape(M, N, N)                    # [M, N, N]
            top1_position_on_map = top1_map.nonzero()[:, 1:]        # [M, 2]

            ## build lower-left mask [S, N, N]
            lower_left_mask = torch.ones(S, N, N, device=device)
            target_count = 0
            for idx, num_target in enumerate(num_targets):  # [S]
                for i in range(num_target):
                    start, end = top1_position_on_map[target_count + i]
                    gt_len = end + 1 - start
                    for s in range(start, end+1):
                        for e in range(start, end+1):
                            ## proposals that has len < xx% of gt_len are still
                            ## neg samples
                            if s <= e and (e+1-s) > int(gt_len*0.0):
                                ## lower-left corner set to 0
                                lower_left_mask[idx, s, e] = 0
                    
                target_count += num_target

            ## lower-left: 0, others: 1
            lower_left_mask = lower_left_mask.masked_select(mask2d).view(S, -1)     # [S, P]
            s2v_neg_mask = ~s2v_pos_mask                                            # [S, P]
            ## for plotting
            original_s2v_neg_mask = s2v_neg_mask.clone()
            ## don't choose the lower-left corner as negative samples
            s2v_neg_mask = torch.logical_and(s2v_neg_mask, lower_left_mask)         # [S, P]
            inter_query_neg_mask[backup_pos_mask.unsqueeze(1)] = s2v_neg_mask.view(-1)
            

            ## build lower-left mask [S, N, N]
            lower_left_mask = torch.ones(S, N, N, device=device)
            target_count = 0
            for idx, num_target in enumerate(num_targets):  # [S]
                for i in range(num_target):
                    start, end = top1_position_on_map[target_count + i]
                    gt_len = end + 1 - start
                    for s in range(start, end+1):
                        for e in range(start, end+1):
                            ## proposals that has len < xx% of gt_len are still
                            ## neg samples
                            if s <= e and (e+1-s) > int(gt_len*0.0):
                                ## lower-left corner set to 0
                                lower_left_mask[idx, s, e] = 0
                    
                target_count += num_target

            ## lower-left: 0, others: 1
            lower_left_mask = lower_left_mask.masked_select(mask2d).view(S, -1)     # [S, P]
            s2v_neg_mask = ~s2v_pos_mask                                            # [S, P]
            ## for plotting
            original_s2v_neg_mask = s2v_neg_mask.clone()
            ## don't choose the lower-left corner as negative samples
            s2v_neg_mask = torch.logical_and(s2v_neg_mask, lower_left_mask)         # [S, P]
            inter_query_neg_mask[backup_pos_mask.unsqueeze(1)] = s2v_neg_mask.view(-1)
            '''
            
            loss_inter_query = self.log_cross_entropy(
                inter_query_pos,                                    # [M, K]
                inter_query_all[scatter_m2s],                       # [M, 1, B * P]
                inter_query_neg_mask[scatter_m2s],                  # [M, 1, B * P]
                self.T_q,
                self.margin,
            )

            ###### for plotting mask
            '''
            ## save gt_iou, pos_mask, neg_mask to check
            ## convert iou2d, pos_mask, neg_mask back to 2d map
            #iou2d = iou2d.masked_select(mask2d).view(S, -1)                 # [S, P]
            gt_iou2d = torch.zeros(N, N)
            gt_iou2d[mask2d] = iou2d[0].cpu()
            pos_mask2d = torch.zeros(N, N, dtype=torch.long)
            pos_mask2d[mask2d] = s2v_pos_mask.long()[0].cpu()
            neg_mask2d = torch.zeros(N, N, dtype=torch.long)
            neg_mask2d[mask2d] = s2v_neg_mask.long()[0].cpu()
            original_neg_mask2d = torch.zeros(N, N, dtype=torch.long)
            original_neg_mask2d[mask2d] = original_s2v_neg_mask.long()[0].cpu()
            
            plot_mask_and_gt(gt_iou2d, pos_mask2d, path="./logs/pos.jpg")
            plot_mask_and_gt(gt_iou2d, neg_mask2d, path="./logs/neg.jpg")
            plot_mask_and_gt(gt_iou2d, original_neg_mask2d, path="./logs/original_neg.jpg")
            '''

        else:
            loss_inter_query = torch.tensor(0., device=device)
     
        #### Compute whole batch 
        if self.intra:
            # === intra video
            shift = 0
            combinations = []
            scatter_e2s = []
            for i, num in enumerate(num_targets):
                ## only for multi-target samples
                if num > 1: 
                    pairs = torch.ones(
                        num * K, num * K, device=device).nonzero()      # [num * K * num * K, 2]
                    combinations.append(pairs + shift)
                    scatter_e2s.append(torch.ones(len(pairs), device=device) * i)
                    shift += num * K

            ## RuntimeError: torch.cat(): expected a non-empty list of Tensors
            ## sometimes a batch only contains single-target samples, so combination is empty
            
            if len(combinations) == 0:
                loss_intra_video = torch.tensor(0., device=device)
                return loss_inter_video, loss_inter_query, loss_intra_video
            
            
            # E: number of (E)numerated positive pairs
            ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
            scatter_e2s = torch.cat(scatter_e2s, dim=0).long()      # [E]  ex.[0, 0, 0, 1, 1, 1...]
            assert (ref_idx < M * K).all()
            assert (pos_idx < M * K).all()

            pos_video_feats = topk_video_feats.reshape(M * K, C)    # [M * K, C]
            intra_video_pos = torch.mul(
                pos_video_feats[ref_idx],                           # [E, C]
                pos_video_feats[pos_idx],                           # [E, C]
            ).sum(dim=1)                                            # [E]

            ## all moment M x all proposals B * P in the batch
            intra_video_all = torch.mm(
                pos_video_feats,                                    # [M * K, C]
                video_feats.view(-1, C).t(),                        # [C, B * P]
            )                                                       # [M * K, B * P]
            
            ## same mask as inter query (sentence -> proposals)
            intra_video_neg_mask = inter_query_neg_mask.clone().squeeze() # [S, B * P]
            
            loss_intra_video = self.log_cross_entropy(
                intra_video_pos,                                    # [E]
                intra_video_all[ref_idx],                           # [E, B * P]
                intra_video_neg_mask[scatter_e2s],                  # [E, B * P] 
                self.T_v,
            )

        else:
            loss_intra_video = torch.tensor(0., device=device)

        return loss_inter_video, loss_inter_query, loss_intra_video


if __name__ == '__main__':
    B = 4
    C = 256
    N = 64
    S = 8

    video_feats = torch.randn(B, C, N, N)
    sents_feats = torch.randn(S, C)
    num_sentences = torch.Tensor([1, 2, 3, 2]).long()
    num_targets = torch.Tensor([1, 2, 2, 1, 7, 1, 3, 2]).long()
    M = num_targets.sum().cpu().item()
    iou2d = torch.rand(S, N, N)
    iou2ds = torch.rand(M, N, N)
    mask2d = (torch.rand(N, N) > 0.5).triu().bool()

    ## test ContrastiveLoss
    '''
    loss_fn = ContrastiveLoss(pos_topk=1, intra=True)
    (
        loss_inter_video,
        loss_inter_query,
        loss_intra_video,
    ) = loss_fn(
        video_feats,
        sents_feats,
        num_sentences,
        num_targets,
        iou2d,
        iou2ds,
        mask2d,
    )
    print(
        loss_inter_video,
        loss_inter_query,
        loss_intra_video,
    )
    '''

    ## test ScaledIoUFocalLoss
    loss_fn = ScaledIoUFocalLoss(
                    min_iou=0.5, 
                    max_iou=1.0, 
                    scale=10,
                    alpha=0.25,
                    gamma=2
                )
    scores2d = torch.rand(S, N, N)
    iou2d = torch.rand(S, N, N)
    loss = loss_fn(scores2d, iou2d, mask2d)
    print(f"Focal loss:{loss.item()}")

    ## test iou loss
    loss_fn = ScaledIoULoss(min_iou=0.1, max_iou=1)
    #scores2d = torch.rand(B, N, N)
    #iou2d = torch.rand(B, N, N)
    loss = loss_fn(scores2d, iou2d, mask2d)
    print(f"BCE loss:{loss.item()}")