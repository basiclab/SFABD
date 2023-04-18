import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCrossEntropy(nn.Module):
    def forward(
        self,
        pos_scores: torch.Tensor,       # [...]  ex. [M, K]
        all_scores: torch.Tensor,       # [..., Number_of_samples]
        neg_mask: torch.Tensor,         # [..., Number_of_samples] 1 -> neg, 0 -> pos
        t: float = 1,                   # temperature
        m: float = 0,                   # margin
    ):
        pos_exp = (pos_scores - m).div(t).exp()                             # [...]
        neg_exp_sum = all_scores.div(t).exp().mul(neg_mask).sum(dim=-1)     # [...]
        all_exp_sum = pos_exp + neg_exp_sum                                 # [...]
        loss = -((pos_scores - m).div(t) - torch.log(all_exp_sum))
        return loss.mean()


class InterContrastiveLoss(LogCrossEntropy):
    def __init__(
        self,
        t: float = 0.1,                 # temperature
        m: float = 0,                   # margin
        neg_iou: float = 0.5,           # negative iou threshold
        pos_topk: int = 1,              # positive topk
        top_neg_removal_percent: float = 0.01,
        weight: float = 1.0,            # weight
    ):
        super().__init__()
        self.t = t
        self.m = m
        self.neg_iou = neg_iou
        self.pos_topk = pos_topk
        self.weight = weight
        self.top_neg_removal_percent = top_neg_removal_percent

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
        if self.weight == 0:
            zero = torch.tensor(0., device=device)
            return (
                zero,
                {
                    'loss/inter_video': zero,
                    'loss/inter_query': zero,
                }
            )

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

        loss_inter_video = super().forward(
            inter_video_pos,                                    # [M, K]
            inter_video_all,                                    # [M, K, S]
            inter_video_neg_mask,                               # [M, 1, S]
            self.t,
            self.m,
        )

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
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
        inter_query_neg_mask = ~pos_mask.unsqueeze(1)           # [S, 1, B * P]

        # Removing top-x% false negatives
        # num_t = 0
        # for sent_idx, (num_target, neg_mask) in enumerate(zip(num_targets, inter_query_neg_mask)):
        #     sent_targets_intra_sim = intra_video_all[num_t:num_t + num_target].mean(dim=0)  # [B * P]
        #     neg_mask_intra_sim = sent_targets_intra_sim.masked_select(neg_mask.squeeze())
        #     remove_mask = torch.ones_like(neg_mask_intra_sim).bool()
        #     # sort by sim score and remove top-x% from neg_mask
        #     K = int(self.top_neg_removal_percent * int(neg_mask_intra_sim.size(dim=0)))
        #     topk_idxs = neg_mask_intra_sim.topk(K, dim=0)[1]             # [K]
        #     # neg samples to be removed
        #     remove_mask[topk_idxs] = 0
        #     inter_query_neg_mask[sent_idx][neg_mask.clone()] = remove_mask
        #     num_t += num_target

        # TODO Try negative sampling strategy
        # neg_samples_num = 512
        # sampled_negative = torch.multinomial(
        #     inter_query_neg_mask.squeeze().float(),
        #     neg_samples_num, replacement=False
        # )                                                       # [S, neg_samples_num]
        # Sorted sampling, high sim -> low sim
        # [-1, 1] -> [0, 1] to avoid negative numbers
        # sampled_negative = torch.multinomial(
        #     (((inter_query_neg_mask * inter_query_all) + 1) / 2).squeeze().float(),
        #     neg_samples_num, replacement=False
        # )                                                     # [S, neg_samples_num]

        # sampled_inter_query_neg_mask = torch.zeros_like(inter_query_neg_mask.squeeze())
        # # from https://discuss.pytorch.org/t/setting-the-values-at-specific-indices-of-a-2d-tensor/168564
        # sampled_inter_query_neg_mask[range(sampled_negative.shape[0]),
        #                              sampled_negative.t()] = 1  # [S, B * P]
        # sampled_inter_query_neg_mask = sampled_inter_query_neg_mask.unsqueeze(1)
        # assert sampled_inter_query_neg_mask.sum(dim=-1).eq(neg_samples_num).all()

        loss_inter_query = super().forward(
            inter_query_pos,                                    # [M, K]
            inter_query_all[scatter_m2s],                       # [M, 1, B * P]
            inter_query_neg_mask[scatter_m2s],                  # [M, 1, B * P]
            self.t,
            self.m,
        )

        return (
            (loss_inter_video + loss_inter_query) * self.weight,
            {
                "loss/inter_video": loss_inter_video,
                "loss/inter_query": loss_inter_query,
            },
        )


class IntraContrastiveLoss(LogCrossEntropy):
    def __init__(
        self,
        t: float = 0.1,                 # temperature
        m: float = 0,                   # margin
        neg_iou: float = 0.5,           # negative iou threshold
        pos_topk: int = 1,              # positive topk
        top_neg_removal_percent: float = 0.01,
        weight: float = 1.0,            # weight
    ):
        super().__init__()
        self.t = t
        self.m = m
        self.neg_iou = neg_iou
        self.pos_topk = pos_topk
        self.weight = weight
        self.top_neg_removal_percent = top_neg_removal_percent

    def forward(
        self,
        video_feats: torch.Tensor,      # [B, C, N, N]
        sents_feats: torch.Tensor,      # [S, C]
        num_sentences: torch.Tensor,    # [B]           number of sentences for each video
        num_targets: torch.Tensor,      # [S]           number of targets for each sentence
        iou2d: torch.Tensor,            # [S, N, N]
        iou2ds: torch.Tensor,           # [M, N, N]
        mask2d: torch.Tensor,           # [N, N]
        sampled_neg_mask: torch.Tensor = None,  # [S, B * P]
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
        if self.weight == 0:
            zero = torch.tensor(0., device=device)
            return (
                zero,
                {
                    'loss/intra_video': zero,
                }
            )

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

        # Enumerate positive pairs
        shift = 0
        combinations = []
        scatter_e2s = []
        for i, num in enumerate(num_targets):
            # only for multi-target samples
            if num > 0:
                pairs = torch.ones(
                    num * K, num * K, device=device).nonzero()  # [num * K * num * K, 2]
                combinations.append(pairs + shift)
                scatter_e2s.append(torch.ones(len(pairs), device=device) * i)
            shift += num * K

        # E: number of (E)numerated positive pairs
        ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
        # S -> E
        scatter_e2s = torch.cat(scatter_e2s, dim=0).long()      # [E]  ex.[0, 0, 0, 1, 1, 1...]

        assert (ref_idx < M * K).all()
        assert (pos_idx < M * K).all()

        # top-K positive proposals
        topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
        topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
        allm_video_feats = video_feats[scatter_m2v]             # [M, P, C]
        topk_video_feats = allm_video_feats.gather(
            dim=1, index=topk_idxs)                             # [M, K, C]

        # positive scores
        pos_video_feats = topk_video_feats.reshape(M * K, C)    # [M * K, C]
        intra_video_pos = torch.mul(
            pos_video_feats[ref_idx],                           # [E, C]
            pos_video_feats[pos_idx],                           # [E, C]
        ).sum(dim=1)                                            # [E]

        # all scores
        intra_video_all = torch.mm(
            pos_video_feats,                                    # [M * K, C]
            video_feats.view(-1, C).t(),                        # [C, B * P]
        )                                                       # [M * K, B * P]

        # negative mask
        pos_mask = torch.eye(B, device=device).bool()           # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
        pos_mask = pos_mask[scatter_s2v]                        # [S, B * P]
        assert pos_mask.long().sum(dim=-1).eq(P).all()
        s2v_pos_mask = iou2d > self.neg_iou                     # [S, P]
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
        intra_video_neg_mask = ~pos_mask                        # [S, B * P]

        # intra_video_temp = intra_video_all.reshape(M, K, -1).mean(dim=1)  # [M, B * P]
        # num_t = 0
        # for sent_idx, (num_target, neg_mask) in enumerate(zip(num_targets, intra_video_neg_mask)):
        #     sent_targets_intra_sim = intra_video_temp[num_t:num_t + num_target].mean(dim=0)  # [B * P]
        #     neg_mask_intra_sim = sent_targets_intra_sim.masked_select(neg_mask.squeeze())
        #     remove_mask = torch.ones_like(neg_mask_intra_sim).bool()
        #     # sort by sim score and remove top-x% from neg_mask
        #     K = int(self.top_neg_removal_percent * int(neg_mask_intra_sim.size(dim=0)))
        #     topk_idxs = neg_mask_intra_sim.topk(K, dim=0)[1]             # [K]
        #     # neg samples to be removed
        #     remove_mask[topk_idxs] = 0
        #     intra_video_neg_mask[sent_idx][neg_mask.clone()] = remove_mask
        #     num_t += num_target

        # TODO Try negative sampling strategy
        # neg_samples_num = 512
        # sampled_negative = torch.multinomial(
        #     intra_video_neg_mask.float(),
        #     neg_samples_num, replacement=False
        # )                                                       # [S, neg_samples_num]

        # Sorted sampling, high sim -> low sim
        # sampled_negative = torch.multinomial(
        #     (((intra_video_neg_mask[scatter_e2s] * intra_video_all[ref_idx]) + 1) / 2).float(),
        #     neg_samples_num, replacement=False
        # )                                                       # [E, neg_samples_num]

        # for uniform sampling
        # sampled_intra_video_neg_mask = torch.zeros_like(intra_video_neg_mask)
        # for sorted sampling
        # sampled_intra_video_neg_mask = torch.zeros((sampled_negative.shape[0], B * P),
        #                                            device=device,
        #                                            )            # [E, B * P]
        # from https://discuss.pytorch.org/t/setting-the-values-at-specific-indices-of-a-2d-tensor/168564
        # sampled_intra_video_neg_mask[range(sampled_negative.shape[0]),
        #                              sampled_negative.t()] = 1  # [S or E, B * P]

        # assert sampled_intra_video_neg_mask.sum(dim=-1).eq(neg_samples_num).all()

        # uniformly sampled negative mask
        # loss_intra_video = super().forward(
        #     intra_video_pos,                                    # [E]
        #     intra_video_all[ref_idx],                           # [E, B * P]
        #     sampled_neg_mask[scatter_e2s],                      # [E, B * P]
        #     self.t,
        #     self.m
        # )

        # sorted sampled negative mask
        # loss_intra_video = super().forward(
        #     intra_video_pos,                                    # [E]
        #     intra_video_all[ref_idx],                           # [E, B * P]
        #     sampled_intra_video_neg_mask,                       # [E, B * P]
        #     self.t,
        #     self.m
        # )

        loss_intra_video = super().forward(
            intra_video_pos,                                    # [E]
            intra_video_all[ref_idx],                           # [E, B * P]
            intra_video_neg_mask[scatter_e2s],                  # [E, B * P]
            self.t,
            self.m
        )

        return (
            loss_intra_video * self.weight,
            {
                'loss/intra_video': loss_intra_video,
            }
        )



# Dynamic negative sampling version
class InterContrastiveLossDNS(InterContrastiveLoss):
    def __init__(
        self,
        t: float = 0.1,                 # temperature
        m: float = 0,                   # margin
        neg_iou: float = 0.5,           # negative iou threshold
        pos_topk: int = 1,              # positive topk
        top_neg_removal_percent: float = 0.01,
        weight: float = 1.0,            # weight
        inter_query_threshold: float = 0.7,
        intra_video_threshold: float = 0.9,
        fusion_ratio: float = 0.5,
        exponent: float = 2,
        neg_samples_num: int = 512,
        start_DNS_epoch: int = 1,
        rate_step_change: float = 0.05,
    ):
        super().__init__(
            t,
            m,
            neg_iou,
            pos_topk,
            top_neg_removal_percent,
            weight,
        )
        self.top_neg_removal_percent = top_neg_removal_percent
        self.inter_query_threshold = inter_query_threshold
        self.intra_video_threshold = intra_video_threshold
        self.fusion_ratio = fusion_ratio
        self.exponent = exponent
        self.neg_samples_num = neg_samples_num
        self.start_DNS_epoch = start_DNS_epoch
        self.rate_step_change = rate_step_change

    def forward(
        self,
        video_feats: torch.Tensor,      # [B, C, N, N]
        sents_feats: torch.Tensor,      # [S, C]
        num_sentences: torch.Tensor,    # [B]           number of sentences for each video
        num_targets: torch.Tensor,      # [S]           number of targets for each sentence
        iou2d: torch.Tensor,            # [S, N, N]
        iou2ds: torch.Tensor,           # [M, N, N]
        mask2d: torch.Tensor,           # [N, N]
        epoch: int
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
        if self.weight == 0:
            zero = torch.tensor(0., device=device)
            return (
                zero,
                {
                    'loss/inter_video': zero,
                    'loss/inter_query': zero,
                }
            )

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

        loss_inter_video = super(InterContrastiveLoss, self).forward(
            inter_video_pos,                                    # [M, K]
            inter_video_all,                                    # [M, K, S]
            inter_video_neg_mask,                               # [M, 1, S]
            self.t,
            self.m,
        )

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
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
        inter_query_neg_mask = ~pos_mask.unsqueeze(1)           # [S, 1, B * P]
        target_inter_query_neg_mask = ~pos_mask[scatter_m2s]    # [M, B * P]

        if epoch >= self.start_DNS_epoch:
            # compute query to all neg proposals sim s_inter
            # inter_query_all  [S, 1, B * P]
            inter_query_sim = inter_query_all[scatter_m2s]          # [M, 1, B * P]
            inter_query_sim = inter_query_sim.squeeze()             # [M, B * P]
            # [-1, 1] -> [0, 1]
            inter_query_sim = (inter_query_sim + 1) / 2             # [M, B * P]
            assert inter_query_sim.sum() > 0
            # intra_video_all  [M, B * P]
            intra_video_all = torch.matmul(
                topk_video_feats,                                   # [M, K, C]
                video_feats.view(-1, C).t(),                        # [C, B * P]
            ).mean(dim=1)                                           # [M, B * P]
            intra_video_sim = intra_video_all.clone()               # [M, B * P]
            # [-1, 1] -> [0, 1]
            intra_video_sim = (intra_video_sim + 1) / 2             # [M, B * P]
            assert intra_video_sim.sum() > 0
            fused_neg_sim = self.fusion_ratio * intra_video_sim + (1 - self.fusion_ratio) * inter_query_sim

            # # Threshold based method
            # inter_query_sim_mask = inter_query_sim > self.inter_query_threshold
            # intra_video_sim_mask = intra_video_sim > self.intra_video_threshold
            # false_neg_mask = inter_query_sim_mask | intra_video_sim_mask   # [M, B * P]
            # accept_rate = min(self.rate_step_change * (epoch - self.start_DNS_epoch), 1.0)
            # # target_false_neg_mask  [B * P]
            # for target_idx, target_false_neg_mask in enumerate(false_neg_mask):
            #     masked_fused_neg_sim = fused_neg_sim[target_idx].masked_select(target_false_neg_mask)
            #     K = int(accept_rate * int(masked_fused_neg_sim.size(dim=0)))
            #     topk_idx = masked_fused_neg_sim.topk(K, dim=0)[1]
            #     accept_false_neg_mask = torch.zeros_like(masked_fused_neg_sim)  # [B * P]
            #     accept_false_neg_mask[topk_idx] = 1
            #     # only accept top-x% as false neg, then remove from neg set later
            #     false_neg_mask[target_idx][target_false_neg_mask.clone()] = accept_false_neg_mask.bool()

            # top-x% based method
            false_neg_mask = torch.zeros_like(fused_neg_sim)        # [M, B * P]
            for target_idx, (target_fused_neg_sim, neg_mask) in enumerate(zip(fused_neg_sim, target_inter_query_neg_mask)):
                # masked selected proposals
                neg_masked_fused_neg_sim = target_fused_neg_sim.masked_select(neg_mask)
                remove_mask = torch.zeros_like(neg_masked_fused_neg_sim)
                K = int(self.top_neg_removal_percent * int(neg_masked_fused_neg_sim.size(dim=0)))
                topk_idx = neg_masked_fused_neg_sim.topk(K, dim=0)[1]
                remove_mask[topk_idx] = 1
                false_neg_mask[target_idx][neg_mask.clone()] = remove_mask

            false_neg_mask = false_neg_mask.bool()                  # [M, B * P]

            fused_neg_sim = torch.pow(fused_neg_sim, self.exponent)
            # not in neg mask, ignore. Won't be sampled as neg later
            fused_neg_sim[~target_inter_query_neg_mask] = 0
            # false neg, ignore. Won't be sampled as neg later
            fused_neg_sim[false_neg_mask] = 0
            # sometimes every neg > threshold, so they are all false neg
            assert fused_neg_sim.sum() > 0
            sampled_negative = torch.multinomial(
                fused_neg_sim,                                      # [M, B * P]
                self.neg_samples_num,
                replacement=False,
            )                                                       # [M, neg_samples_num]
            sampled_inter_query_neg_mask = torch.zeros_like(fused_neg_sim)  # [M, B * P]
            # from https://discuss.pytorch.org/t/setting-the-values-at-specific-indices-of-a-2d-tensor/168564
            sampled_inter_query_neg_mask[range(sampled_negative.shape[0]),
                                         sampled_negative.t()] = 1  # [M, B * P]
            assert sampled_inter_query_neg_mask.sum(dim=-1).eq(self.neg_samples_num).all()

            num_t = 0
            bce_sampled_neg_mask = torch.zeros(S, B * P, device=device)
            intra_sampled_neg_mask = torch.zeros(S, B * P, device=device)
            # convert [M, B * P] to [S, B * P]
            for sent_idx, num_target in enumerate(num_targets):
                bce_sampled_neg_mask[sent_idx] = sampled_inter_query_neg_mask[num_t:num_t + num_target].mean(dim=0) > 0
                intra_sampled_neg_mask[sent_idx] = sampled_inter_query_neg_mask[num_t:num_t + num_target].mean(dim=0) > 0
                num_t += num_target
            bce_sampled_neg_mask = bce_sampled_neg_mask[local_mask].reshape(S, P)

            # sampled negative mask
            loss_inter_query = super(InterContrastiveLoss, self).forward(
                inter_query_pos,                                    # [M, K]
                inter_query_all[scatter_m2s],                       # [M, 1, B * P]
                sampled_inter_query_neg_mask.unsqueeze(1),          # [M, 1, B * P]
                self.t,
                self.m,
            )
            return (
                (loss_inter_video + loss_inter_query) * self.weight,
                {
                    "loss/inter_video": loss_inter_video,
                    "loss/inter_query": loss_inter_query,
                },
                bce_sampled_neg_mask,                               # [S, P]
                intra_sampled_neg_mask,                             # [S, B * P]
            )

        else:
            loss_inter_query = super(InterContrastiveLoss, self).forward(
                inter_query_pos,                                    # [M, K]
                inter_query_all[scatter_m2s],                       # [M, 1, B * P]
                inter_query_neg_mask[scatter_m2s],                  # [M, 1, B * P]
                self.t,
                self.m,
            )

            return (
                (loss_inter_video + loss_inter_query) * self.weight,
                {
                    "loss/inter_video": loss_inter_video,
                    "loss/inter_query": loss_inter_query,
                },
                None,
                None,
            )


class IntraContrastiveLossDNS(IntraContrastiveLoss):
    def __init__(
        self,
        t: float = 0.1,                 # temperature
        m: float = 0,                   # margin
        neg_iou: float = 0.5,           # negative iou threshold
        pos_topk: int = 1,              # positive topk
        top_neg_removal_percent: float = 0.01,
        weight: float = 1.0,            # weight
        mixup_alpha: float = 0.9,
    ):
        super().__init__(
            t,
            m,
            neg_iou,
            pos_topk,
            top_neg_removal_percent,
            weight,
        )
        self.mixup_alpha = mixup_alpha

    def forward(
        self,
        video_feats: torch.Tensor,      # [B, C, N, N]
        sents_feats: torch.Tensor,      # [S, C]
        num_sentences: torch.Tensor,    # [B]           number of sentences for each video
        num_targets: torch.Tensor,      # [S]           number of targets for each sentence
        iou2d: torch.Tensor,            # [S, N, N]
        iou2ds: torch.Tensor,           # [M, N, N]
        mask2d: torch.Tensor,           # [N, N]
        sampled_neg_mask: torch.Tensor = None,  # [S, B * P]
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
        if self.weight == 0:
            zero = torch.tensor(0., device=device)
            return (
                zero,
                {
                    'loss/intra_video': zero,
                }
            )

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

        # Re-write this part (assume that each sample only have 1 augmented feat)
        # shift = 0
        # combinations = []
        # scatter_e2s = []
        # for i, num in enumerate(num_targets):
        #     # only for multi-target samples
        #     pairs = torch.ones(
        #         num * K * 2, num * K * 2, device=device).nonzero()  # [num * K * 2 * num * K * 2, 2]
        #     # mask trivial pairs to save memory cost
        #     # pairs = torch.ones(
        #     #     num * K * 2, num * K * 2, device=device)
        #     # diagonal_mask = torch.eye(num * K * 2, device=device)
        #     # pairs = pairs * ~diagonal_mask

        #     combinations.append(pairs + shift)
        #     scatter_e2s.append(torch.ones(len(pairs), device=device) * i)
        #     shift += num * K * 2

        # # E: number of (E)numerated positive pairs
        # ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
        # # S -> E
        # scatter_e2s = torch.cat(scatter_e2s, dim=0).long()      # [E]  ex.[0, 0, 0, 1, 1, 1...]

        # assert (ref_idx < M * K * 2).all()
        # assert (pos_idx < M * K * 2).all()

        # Enumerate positive pairs (original)
        shift = 0
        combinations = []
        scatter_e2s = []
        for i, num in enumerate(num_targets):
            # only for multi-target samples
            if num > 0:
                pairs = torch.ones(
                    num * K, num * K, device=device).nonzero()  # [num * K * num * K, 2]
                combinations.append(pairs + shift)
                scatter_e2s.append(torch.ones(len(pairs), device=device) * i)
            shift += num * K

        # E: number of (E)numerated positive pairs
        ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
        # S -> E
        scatter_e2s = torch.cat(scatter_e2s, dim=0).long()      # [E]  ex.[0, 0, 0, 1, 1, 1...]

        assert (ref_idx < M * K).all()
        assert (pos_idx < M * K).all()

        # top-K positive proposals
        topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
        topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
        allm_video_feats = video_feats[scatter_m2v]             # [M, P, C]
        topk_video_feats = allm_video_feats.gather(
            dim=1, index=topk_idxs)                             # [M, K, C]

        # positive scores
        pos_video_feats = topk_video_feats.reshape(M * K, C)    # [M * K, C]

        # Do feature space augmentation with query
        # a * v_feat + (1 - a) * q_feat
        # TODO, sample alpha from a distribution
        # temp_sent_feats = sents_feats[scatter_m2s]              # [M, C]
        # temp_sent_feats = torch.repeat_interleave(temp_sent_feats, K, dim=0)    # [M * K, C]
        # aug_video_feats = self.mixup_alpha * pos_video_feats + (1 - self.mixup_alpha) * temp_sent_feats
        # aug_video_feats = F.normalize(aug_video_feats.contiguous(), dim=-1)  # [M * K, C]
        # # concate this with original pos_v_feats [M * K, 2, C]
        # new_video_feats = torch.stack([pos_video_feats, aug_video_feats], dim=1)  # [M * K, 2, C]
        # # then reshape to [M * K * 2, C]
        # new_video_feats = new_video_feats.reshape(M * K * 2, C)

        # intra_video_pos = torch.mul(
        #     new_video_feats[ref_idx],                           # [E, C]
        #     new_video_feats[pos_idx],                           # [E, C]
        # ).sum(dim=1)                                            # [E]

        # # all scores
        # intra_video_all = torch.mm(
        #     new_video_feats,                                    # [M * K * 2, C]
        #     video_feats.view(-1, C).t(),                        # [C, B * P]
        # )                                                       # [M * K * 2, B * P]

        intra_video_pos = torch.mul(
            pos_video_feats[ref_idx],                           # [E, C]
            pos_video_feats[pos_idx],                           # [E, C]
        ).sum(dim=1)                                            # [E]

        # all scores
        intra_video_all = torch.mm(
            pos_video_feats,                                    # [M * K, C]
            video_feats.view(-1, C).t(),                        # [C, B * P]
        )                                                       # [M * K, B * P]

        # negative mask
        pos_mask = torch.eye(B, device=device).bool()           # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
        pos_mask = pos_mask[scatter_s2v]                        # [S, B * P]
        assert pos_mask.long().sum(dim=-1).eq(P).all()
        s2v_pos_mask = iou2d > self.neg_iou                     # [S, P]
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
        intra_video_neg_mask = ~pos_mask                        # [S, B * P]

        if sampled_neg_mask is not None:
            loss_intra_video = super(IntraContrastiveLoss, self).forward(
                intra_video_pos,                                    # [E]
                intra_video_all[ref_idx],                           # [E, B * P]
                sampled_neg_mask[scatter_e2s],                      # [E, B * P]
                self.t,
                self.m
            )
        else:
            loss_intra_video = super(IntraContrastiveLoss, self).forward(
                intra_video_pos,                                    # [E]
                intra_video_all[ref_idx],                           # [E, B * P]
                intra_video_neg_mask[scatter_e2s],                  # [E, B * P]
                self.t,
                self.m
            )

        return (
            loss_intra_video * self.weight,
            {
                'loss/intra_video': loss_intra_video,
            }
        )



class LogCrossEntropyMP(nn.Module):
    def forward(
        self,
        pos_scores: torch.Tensor,       # [...]  ex. [M, K]
        all_scores: torch.Tensor,       # [..., Number_of_samples]
        neg_mask: torch.Tensor,         # [..., Number_of_samples] 1 -> neg, 0 -> pos
        t: float = 1,                   # temperature
        m: float = 0,                   # margin
    ):
        pos_exp = (pos_scores - m).div(t).exp()                             # [...]
        neg_exp_sum = all_scores.exp().mul(neg_mask).sum(dim=-1)     # [...]
        all_exp_sum = pos_exp + neg_exp_sum                                 # [...]
        loss = -((pos_scores - m).div(t) - torch.log(all_exp_sum))
        return loss.mean()


class MultiPositiveContrastiveLoss(LogCrossEntropyMP):
    def __init__(
        self,
        t: float = 0.1,                 # temperature
        inter_m: float = 0.3,           # margin for inter
        intra_m: float = 0.0,           # margin for intra
        neg_iou: float = 0.5,           # negative iou threshold
        pos_topk: int = 1,              # positive topk
        inter_weight: float = 1.0,      # inter domain
        intra_weight: float = 0.1,      # intra domain
    ):
        super().__init__()
        self.t = t
        self.inter_m = inter_m
        self.intra_m = intra_m
        self.neg_iou = neg_iou
        self.pos_topk = pos_topk
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight

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
        if self.inter_weight == 0 and self.intra_weight == 0:
            zero = torch.tensor(0., device=device)
            return (
                zero,
                {
                    'loss/inter_video': zero,
                    'loss/inter_query': zero,
                    'loss/intra_video': zero,
                }
            )

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
        ########################################################################

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
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
        inter_query_neg_mask = ~pos_mask.unsqueeze(1)           # [S, 1, B * P]

        # TODO Try negative sampling strategy
        neg_samples_num = 4096
        sampled_negative = torch.multinomial(
            inter_query_neg_mask.squeeze().float(),
            neg_samples_num, replacement=False
        )                                                       # [S, neg_samples_num]

        sampled_inter_query_neg_mask = torch.zeros_like(inter_query_neg_mask.squeeze())
        # from https://discuss.pytorch.org/t/setting-the-values-at-specific-indices-of-a-2d-tensor/168564
        sampled_inter_query_neg_mask[range(sampled_negative.shape[0]),
                                     sampled_negative.t()] = 1  # [S, B * P]
        sampled_inter_query_neg_mask = sampled_inter_query_neg_mask.unsqueeze(1)
        assert sampled_inter_query_neg_mask.sum(dim=-1).eq(neg_samples_num).all()

        ########################################################################

        # === intra video (video proposals -> all video proposals)
        # Enumerate positive pairs
        shift = 0
        combinations = []
        scatter_e2s = []
        for i, num in enumerate(num_targets):
            pairs = torch.ones(
                num * K, num * K, device=device).nonzero()  # [num * K * num * K, 2]
            combinations.append(pairs + shift)
            scatter_e2s.append(torch.ones(len(pairs), device=device) * i)
            shift += num * K

        # E: number of (E)numerated positive pairs
        ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
        scatter_e2s = torch.cat(scatter_e2s, dim=0).long()      # [E]  ex.[0, 0, 0, 1, 1, 1...]
        assert (ref_idx < M * K).all()
        assert (pos_idx < M * K).all()

        # positive scores
        pos_video_feats = topk_video_feats.reshape(M * K, C)    # [M * K, C]
        intra_video_pos = torch.mul(
            pos_video_feats[ref_idx],                           # [E, C]
            pos_video_feats[pos_idx],                           # [E, C]
        ).sum(dim=1)                                            # [E]

        # all scores
        intra_video_all = torch.mm(
            pos_video_feats,                                    # [M * K, C]
            video_feats.view(-1, C).t(),                        # [C, B * P]
        )                                                       # [M * K, B * P]

        intra_video_neg_mask = ~pos_mask                        # [S, B * P]

        # TODO Try negative sampling strategy
        sampled_negative = torch.multinomial(
            intra_video_neg_mask.float(),
            neg_samples_num, replacement=False
        )                                                       # [S, neg_samples_num]

        # for uniform sampling
        sampled_intra_video_neg_mask = torch.zeros_like(intra_video_neg_mask)
        # from https://discuss.pytorch.org/t/setting-the-values-at-specific-indices-of-a-2d-tensor/168564
        sampled_intra_video_neg_mask[range(sampled_negative.shape[0]),
                                     sampled_negative.t()] = 1  # [S, B * P]
        assert sampled_intra_video_neg_mask.sum(dim=-1).eq(neg_samples_num).all()
        ########################################################################

        # concat the neg pairs and neg_mask of (vid, vid) pairs to
        # inter_video_all and inter_video_neg_mask
        # loss_inter_video = super().forward(
        #     inter_video_pos,                                       # [M, K]
        #     torch.cat(
        #         (inter_video_all,                                  # [M, K, S]
        #          intra_video_all.reshape(M, K, -1)),               # [M, K, B * P]
        #         dim=-1,
        #     ),                                                     # [M, K, S + B * P]
        #     torch.cat(
        #         (inter_video_neg_mask,                             # [M, 1, S]
        #          intra_video_neg_mask[scatter_m2s].unsqueeze(1)),  # [M, 1, B * P]
        #         dim=-1,
        #     ),                                                     # [M, 1, S + B * P]
        #     self.t,
        #     self.inter_m,
        # )

        # # no (query, query) pairs so no changes
        # loss_inter_query = super().forward(
        #     inter_query_pos,                                    # [M, K]
        #     inter_query_all[scatter_m2s],                       # [M, 1, B * P]
        #     inter_query_neg_mask[scatter_m2s],                  # [M, 1, B * P]
        #     self.t,
        #     self.inter_m,
        # )

        # # concat the neg pairs and neg_mask of (vid, text) pairs to
        # # intra_video_all[ref_idx] and intra_video_neg_mask[scatter_e2s]
        # # intra_video_all # [M * K, B * P]
        # # inter_video_all # [M, K, S]
        # # inter_video_neg_mask  # [M, 1, S] -> [M, K, S] -> [E, S]
        # loss_intra_video = super().forward(
        #     intra_video_pos,                                    # [E]
        #     torch.cat(
        #         (intra_video_all[ref_idx],                      # [E, B * P]
        #          inter_video_all.reshape(M * K, -1)[ref_idx]),  # [E, S]
        #         dim=-1
        #     ),                                                  # [E, B * P + S]
        #     torch.cat(
        #         (intra_video_neg_mask[scatter_e2s],             # [E, B * P]
        #          # [M, K, S] -> [E, S]
        #          inter_video_neg_mask.repeat(1, K, 1).reshape(M * K, -1)[ref_idx]),
        #         dim=-1
        #     ),                                                  # [E, B * P + S]
        #     self.t,
        #     self.intra_m
        # )

        # #### Sampled Version #################################################
        intra_t = 1.0
        loss_inter_video = super().forward(
            inter_video_pos,                                        # [M, K]
            torch.cat(
                (inter_video_all.div(self.t),                                   # [M, K, S]
                 intra_video_all.reshape(M, K, -1).div(intra_t)),                # [M, K, B * P]
                dim=-1,
            ),                                                      # [M, K, S + B * P]
            torch.cat(
                (inter_video_neg_mask,                              # [M, 1, S]
                 sampled_intra_video_neg_mask[scatter_m2s]\
                 .unsqueeze(1)),                                    # [M, 1, B * P]
                dim=-1,
            ),                                                      # [M, 1, S + B * P]
            self.t,
            self.inter_m,
        )

        loss_inter_query = super().forward(
            inter_query_pos,                                        # [M, K]
            inter_query_all[scatter_m2s].div(self.t),                           # [M, 1, B * P]
            sampled_inter_query_neg_mask[scatter_m2s],              # [M, 1, B * P]
            self.t,
            self.inter_m,
        )

        loss_intra_video = super().forward(
            intra_video_pos,                                        # [E]
            torch.cat(
                (intra_video_all[ref_idx].div(intra_t),                          # [E, B * P]
                 inter_video_all.reshape(M * K, -1)[ref_idx].div(self.t)),      # [E, S]
                dim=-1
            ),                                                      # [E, B * P + S]
            torch.cat(
                (sampled_intra_video_neg_mask[scatter_e2s],         # [E, B * P]
                 # [M, K, S] -> [E, S]
                 inter_video_neg_mask.repeat(1, K, 1).reshape(M * K, -1)[ref_idx]),
                dim=-1
            ),                                                  # [E, B * P + S]
            # self.t,
            intra_t,
            self.intra_m
        )

        return \
            loss_inter_video * self.inter_weight + \
            loss_inter_query * self.inter_weight + \
            loss_intra_video * self.intra_weight, \
            {
                "loss/inter_video": loss_inter_video,
                "loss/inter_query": loss_inter_query,
                "loss/intra_video": loss_intra_video,
            }


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

    # test InterContrastiveLoss
    loss_fn = InterContrastiveLoss(t=0.1, m=0.3, neg_iou=0.5, pos_topk=1,
                                   top_neg_removal_percent=0.01)
    loss_inter, loss_inter_metrics = loss_fn(
        video_feats,
        sents_feats,
        num_sentences,
        num_targets,
        iou2d,
        iou2ds,
        mask2d,
    )
    print("InterContrastiveLoss:", loss_inter, loss_inter_metrics)

    # # test IntraContrastiveLoss
    loss_fn = IntraContrastiveLoss(t=0.1, m=0.0, neg_iou=0.5, pos_topk=1,
                                   top_neg_removal_percent=0.01)
    loss_intra, loss_intra_metrics = loss_fn(
        video_feats,
        sents_feats,
        num_sentences,
        num_targets,
        iou2d,
        iou2ds,
        mask2d,
    )
    print("IntraContrastiveLoss:", loss_intra, loss_intra_metrics)

    # loss_fn = MultiPositiveContrastiveLoss(t=0.1, inter_m=0.3, intra_m=0.0, neg_iou=0.5, pos_topk=1)
    # loss_all, loss_all_metrics = loss_fn(
    #     video_feats,
    #     sents_feats,
    #     num_sentences,
    #     num_targets,
    #     iou2d,
    #     iou2ds,
    #     mask2d,
    # )
    # print("MultiPositiveContrastiveLoss:", loss_all, loss_all_metrics)
