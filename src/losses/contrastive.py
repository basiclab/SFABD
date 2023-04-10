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
        # TODO find easy neg mask
        # neg_margin = 0.3
        # zero_tensor = torch.zeros_like(inter_video_all)  # [M, K, S]
        # easy_neg_mask = torch.maximum(zero_tensor,
        #                               inter_video_all + neg_margin)
        # easy_neg_mask[easy_neg_mask > 0] = 1             # keep where mask = 1
        # easy_neg_mask = easy_neg_mask[:, 0, :].bool()    # [M, 1, S]

        mask = ~torch.eye(S, device=device).bool()              # [S, S]
        inter_video_neg_mask = mask[scatter_m2s].unsqueeze(1)   # [M, 1, S]

        loss_inter_video = super().forward(
            inter_video_pos,                                    # [M, K]
            inter_video_all,                                    # [M, K, S]
            inter_video_neg_mask,                               # [M, 1, S]
            # inter_video_neg_mask * easy_neg_mask,               # [M, 1, S]
            self.t,
            self.m,
        )

        # === inter query (sentence -> all video proposals)
        inter_query_pos = inter_video_pos                       # [M, K]

        inter_query_all = torch.mm(
            sents_feats,                                        # [S, C]
            video_feats.view(-1, C).t(),                        # [C, B * P]
        ).unsqueeze(1)                                          # [S, 1, B * P]
        # # TODO find easy neg mask
        # zero_tensor = torch.zeros_like(inter_query_all)  # [S, 1, B * P]
        # easy_neg_mask = torch.maximum(zero_tensor,
        #                               inter_query_all + neg_margin)
        # easy_neg_mask[easy_neg_mask > 0] = 1             # keep where mask = 1
        # easy_neg_mask = easy_neg_mask.bool()    # [S, 1, B * P]

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
        # intra_video_all = torch.matmul(
        #     topk_video_feats,                                   # [M, K, C]
        #     video_feats.view(-1, C).t(),                        # [C, B * P]
        # ).mean(dim=1)                                           # [M, B * P]
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
            # inter_query_neg_mask[scatter_m2s] * easy_neg_mask[scatter_m2s],
            self.t,
            self.m,
        )

        # sampled negative mask
        # loss_inter_query = super().forward(
        #     inter_query_pos,                                    # [M, K]
        #     inter_query_all[scatter_m2s],                       # [M, 1, B * P]
        #     sampled_inter_query_neg_mask[scatter_m2s],          # [M, 1, B * P]
        #     self.t,
        #     self.m,
        # )

        return (
            (loss_inter_video + loss_inter_query) * self.weight,
            {
                "loss/inter_video": loss_inter_video,
                "loss/inter_query": loss_inter_query,
            },
            # inter_query_neg_mask.squeeze()[local_mask].reshape(S, P)  # [S, P]
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
        # TODO find easy neg mask
        # neg_margin = 0.3
        # zero_tensor = torch.zeros_like(intra_video_all)         # [M * K, B * P]
        # easy_neg_mask = torch.maximum(zero_tensor,
        #                               intra_video_all + neg_margin)
        # easy_neg_mask[easy_neg_mask > 0] = 1                    # keep where mask = 1
        # easy_neg_mask = easy_neg_mask[ref_idx]                  # [E, B * P]
        # easy_neg_mask = easy_neg_mask.bool()                    # [E, B * P]

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

        loss_intra_video = super().forward(
            intra_video_pos,                                    # [E]
            intra_video_all[ref_idx],                           # [E, B * P]
            intra_video_neg_mask[scatter_e2s],                  # [E, B * P]
            # intra_video_neg_mask[scatter_e2s] * easy_neg_mask,  # [E, B * P]
            self.t,
            self.m
        )

        # uniformly sampled negative mask
        # loss_intra_video = super().forward(
        #     intra_video_pos,                                    # [E]
        #     intra_video_all[ref_idx],                           # [E, B * P]
        #     sampled_intra_video_neg_mask[scatter_e2s],          # [E, B * P]
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