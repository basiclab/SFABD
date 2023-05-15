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
        weight: float = 1.0,            # weight
    ):
        super().__init__()
        self.t = t
        self.m = m
        self.neg_iou = neg_iou
        self.pos_topk = pos_topk
        self.weight = weight

    def forward(
        self,
        video_feats: torch.Tensor,      # [S, C, N, N]
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

        S, C, N, _ = video_feats.shape
        B = num_sentences.shape[0]
        M = num_targets.sum().cpu().item()
        P = mask2d.long().sum()
        K = self.pos_topk

        assert iou2d.shape == (S, N, N), f"{iou2d.shape} != {(S, N, N)}"
        assert iou2ds.shape == (M, N, N), f"{iou2ds.shape} != {(M, N, N)}"

        scatter_b2s = []
        count = 0
        for num_sentence in num_sentences:
            scatter_b2s.append(count)
            count = count + num_sentence.item()
        scatter_b2s = torch.tensor(scatter_b2s, device=device)            # [B]

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

        # === inter video (topk proposal -> all sentences)
        topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
        topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
        allm_video_feats = video_feats[scatter_m2s]             # [M, P, C]
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
        # need to convert video_feats from [S, P, C] to [B, P, C]
        inter_query_all = torch.mm(
            sents_feats,                                        # [S, C]
            video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
        ).unsqueeze(1)                                          # [S, 1, B * P]

        pos_mask = torch.eye(B, device=device).bool()           # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
        pos_mask = pos_mask[scatter_s2b]                        # [S, B * P]
        assert pos_mask.long().sum(dim=-1).eq(P).all()
        s2v_pos_mask = iou2d > self.neg_iou                     # [S, P]
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
        inter_query_neg_mask = ~pos_mask.unsqueeze(1)           # [S, 1, B * P]

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
        weight: float = 1.0,            # weight
    ):
        super().__init__()
        self.t = t
        self.m = m
        self.neg_iou = neg_iou
        self.pos_topk = pos_topk
        self.weight = weight

    def forward(
        self,
        video_feats: torch.Tensor,      # [S, C, N, N]
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
        S, C, N, _ = video_feats.shape
        B = num_sentences.shape[0]
        M = num_targets.sum().cpu().item()
        P = mask2d.long().sum()
        K = self.pos_topk

        assert iou2d.shape == (S, N, N), f"{iou2d.shape} != {(S, N, N)}"
        assert iou2ds.shape == (M, N, N), f"{iou2ds.shape} != {(M, N, N)}"

        scatter_m2s = torch.arange(S, device=device).long()
        scatter_m2s = scatter_m2s.repeat_interleave(num_targets)        # [M]

        scatter_b2s = []
        count = 0
        for num_sentence in num_sentences:
            scatter_b2s.append(count)
            count = count + num_sentence.item()
        scatter_b2s = torch.tensor(scatter_b2s, device=device)            # [B]

        scatter_s2b = torch.arange(B, device=device).long()
        scatter_s2b = scatter_s2b.repeat_interleave(num_sentences)

        video_feats = video_feats.masked_select(mask2d).view(S, C, -1)  # [S, C, P]
        video_feats = video_feats.permute(0, 2, 1)                      # [S, P, C]
        iou2d = iou2d.masked_select(mask2d).view(S, -1)                 # [S, P]
        iou2ds = iou2ds.masked_select(mask2d).view(M, -1)               # [M, P]

        # normalize for cosine similarity
        video_feats = F.normalize(video_feats.contiguous(), dim=-1)     # [S, P, C]
        sents_feats = F.normalize(sents_feats.contiguous(), dim=-1)     # [S, C]

        # Enumerate positive pairs
        shift = 0
        combinations = []
        scatter_e2s = []
        for i, num in enumerate(num_targets):
            if num > 0:
                # use trivial pos pairs
                pairs = torch.ones(
                    num * K, num * K, device=device).nonzero()  # [num * K * num * K, 2]
                combinations.append(pairs + shift)
                scatter_e2s.append(torch.ones(len(pairs), device=device) * i)

            # only for multi-target samples if num > 1:
            #  num > 0 because activity and charades may have whole batch without augmentation
            # if num > 0:
            #     pairs = torch.ones(num * K, num * K, device=device)   # [num * K, num * K]
            #     trivial_pair_mask = 1 - torch.eye(num * K, device=device)
            #     pairs = (pairs * trivial_pair_mask).nonzero()         # [num * K * num * K, 2]
            #     combinations.append(pairs + shift)
            #     scatter_e2s.append(torch.ones(len(pairs), device=device) * i)

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
        allm_video_feats = video_feats[scatter_m2s]             # [M, P, C]
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
            video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
        )                                                       # [M * K, B * P]

        # negative mask
        pos_mask = torch.eye(B, device=device).bool()           # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
        pos_mask = pos_mask[scatter_s2b]                        # [S, B * P]
        assert pos_mask.long().sum(dim=-1).eq(P).all()
        s2v_pos_mask = iou2d > self.neg_iou                     # [S, P]
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]
        intra_video_neg_mask = ~pos_mask                        # [S, B * P]

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

# Augmentation version DNS loss
class InterContrastiveLossDNS(InterContrastiveLoss):
    def __init__(
        self,
        t: float = 0.1,                 # temperature
        m: float = 0,                   # margin
        neg_iou: float = 0.5,           # negative iou threshold
        pos_topk: int = 1,              # positive topk
        weight: float = 1.0,            # weight
        exponent: float = 2,
        neg_samples_num: int = 512,
        start_DNS_epoch: int = 1,
    ):
        super().__init__(
            t,
            m,
            neg_iou,
            pos_topk,
            weight,
        )
        self.exponent = exponent
        self.neg_samples_num = neg_samples_num
        self.start_DNS_epoch = start_DNS_epoch

    def forward(
        self,
        video_feats: torch.Tensor,              # [S, C, N, N]
        sents_feats: torch.Tensor,              # [S, C]
        num_sentences: torch.Tensor,            # [B] number of sentences for each video
        num_targets: torch.Tensor,              # [S] number of targets for each sentence
        iou2d: torch.Tensor,                    # [S, N, N]
        iou2ds: torch.Tensor,                   # [M, N, N]
        mask2d: torch.Tensor,                   # [N, N]
        epoch: int,
        false_neg_mask: torch.Tensor = None,    # [S, B * P]
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

        S, C, N, _ = video_feats.shape
        B = num_sentences.shape[0]
        M = num_targets.sum().cpu().item()
        P = mask2d.long().sum()
        K = self.pos_topk

        assert iou2d.shape == (S, N, N), f"{iou2d.shape} != {(S, N, N)}"
        assert iou2ds.shape == (M, N, N), f"{iou2ds.shape} != {(M, N, N)}"

        # Choose each sample's first sentence idx to get B
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

        # === inter video (topk proposal -> all sentences)
        topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [M, K]
        topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [M, K, C]
        allm_video_feats = video_feats[scatter_m2s]             # [M, P, C]
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
        # need to convert video_feats from [S, P, C] to [B, P, C]
        inter_query_all = torch.mm(
            sents_feats,                                        # [S, C]
            video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
        ).unsqueeze(1)                                          # [S, 1, B * P]

        pos_mask = torch.eye(B, device=device).bool()           # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
        pos_mask = pos_mask[scatter_s2b]                        # [S, B * P]
        assert pos_mask.long().sum(dim=-1).eq(P).all()
        s2v_pos_mask = iou2d > self.neg_iou                     # [S, P]
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]

        # Do DNS sampling
        if epoch >= self.start_DNS_epoch:
            # compute cos_sim of query to all neg proposals
            # inter_query_all  [S, 1, B * P]
            inter_query_sim = inter_query_all[scatter_m2s]          # [M, 1, B * P]
            inter_query_sim = inter_query_sim.squeeze()             # [M, B * P]
            # [-1, 1] -> [0, 1]
            inter_query_sim = (inter_query_sim + 1) / 2             # [M, B * P]
            assert (inter_query_sim > 0 - 1e-3).all()
            assert (inter_query_sim < 1 + 1e-3).all()

            # fused_neg_sim is used to do dynamic negative sampling
            fused_neg_sim = inter_query_sim                         # [M, B * P]
            # x^exponent so that hard neg samples will be sampled more
            fused_neg_sim = torch.pow(fused_neg_sim, self.exponent)
            fused_neg_sim[pos_mask[scatter_m2s]] = 0             # ignore pos samples
            if false_neg_mask != None:
                fused_neg_sim[false_neg_mask[scatter_m2s]] = 0   # ignore false neg samples
            # make sure the amount of negative samples is enough for sampling
            assert (fused_neg_sim > 0).sum() >= self.neg_samples_num

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
            )

        # No dynamic negative sampling
        else:
            inter_query_neg_mask = ~pos_mask.unsqueeze(1)           # [S, 1, B * P]
            if false_neg_mask != None:
                inter_query_neg_mask[false_neg_mask.unsqueeze(dim=1)] = 0    # remove false neg from neg mask

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
            )


class IntraContrastiveLossDNS(IntraContrastiveLoss):
    def __init__(
        self,
        t: float = 0.1,                 # temperature
        m: float = 0,                   # margin
        neg_iou: float = 0.5,           # negative iou threshold
        pos_topk: int = 1,              # positive topk
        weight: float = 1.0,            # weight
        exponent: float = 2,
        neg_samples_num: int = 512,
        start_DNS_epoch: int = 1,
    ):
        super().__init__(
            t,
            m,
            neg_iou,
            pos_topk,
            weight,
        )
        self.exponent = exponent
        self.neg_samples_num = neg_samples_num
        self.start_DNS_epoch = start_DNS_epoch

    def forward(
        self,
        video_feats: torch.Tensor,              # [S, C, N, N]
        sents_feats: torch.Tensor,              # [S, C]
        num_sentences: torch.Tensor,            # [B]           number of sentences for each video
        num_targets: torch.Tensor,              # [S]           number of targets for each sentence
        iou2d: torch.Tensor,                    # [S, N, N]
        iou2ds: torch.Tensor,                   # [M, N, N]
        mask2d: torch.Tensor,                   # [N, N]
        epoch: int,
        false_neg_mask: torch.Tensor = None,    # [S, B * P]
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

        S, C, N, _ = video_feats.shape
        B = num_sentences.shape[0]
        M = num_targets.sum().cpu().item()
        P = mask2d.long().sum()
        K = self.pos_topk

        assert iou2d.shape == (S, N, N), f"{iou2d.shape} != {(S, N, N)}"
        assert iou2ds.shape == (M, N, N), f"{iou2ds.shape} != {(M, N, N)}"

        # Choose each sample's first sentence idx to get B
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

        # Enumerate positive pairs (original)
        shift = 0
        combinations = []
        scatter_e2s = []
        for i, num in enumerate(num_targets):
            # use trivial pos pairs
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
        allm_video_feats = video_feats[scatter_m2s]             # [M, P, C]
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
            video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
        )                                                       # [M * K, B * P]

        # negative mask
        pos_mask = torch.eye(B, device=device).bool()           # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)                       # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)                   # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)                      # [B, B * P]
        pos_mask = pos_mask[scatter_s2b]                        # [S, B * P]
        assert pos_mask.long().sum(dim=-1).eq(P).all()
        s2v_pos_mask = iou2d > self.neg_iou                     # [S, P]
        local_mask = pos_mask.clone()                           # [S, B * P]
        pos_mask[local_mask] = s2v_pos_mask.view(-1)            # [S, B * P]

        if epoch >= self.start_DNS_epoch:
            # compute cos_sim of topk video proposals and all neg proposals
            # intra_video_all  [M, B * P]
            intra_video_all_topk_mean = torch.matmul(
                topk_video_feats,                                   # [M, K, C]
                video_feats[scatter_b2s].view(-1, C).t(),           # [C, B * P]
            ).mean(dim=1)                                           # [M, B * P]
            # [-1, 1] -> [0, 1]
            intra_video_sim = (intra_video_all_topk_mean + 1) / 2             # [M, B * P]            
            assert (intra_video_sim > 0 - 1e-3).all()
            assert (intra_video_sim < 1 + 1e-3).all()

            # convert fused_neg_sim from [M, B * P] to [S, B * P]
            num_t = 0
            fused_neg_sim = torch.zeros(S, B * P, device=device)
            for sent_idx, num_target in enumerate(num_targets):
                fused_neg_sim[sent_idx] = intra_video_sim[num_t:num_t + num_target].mean(dim=0)
                num_t += num_target

            # x^exponent so that hard negative samples will be sampled more
            fused_neg_sim = torch.pow(fused_neg_sim, self.exponent)  # [S, B * P]
            fused_neg_sim[pos_mask] = 0             # ignore pos samples
            if false_neg_mask != None:
                fused_neg_sim[false_neg_mask] = 0       # ignore false neg samples
            # make sure the amount of negative samples is enough for sampling
            assert (fused_neg_sim > 0).sum() >= self.neg_samples_num

            sampled_negative = torch.multinomial(
                fused_neg_sim,                                      # [S, B * P]
                self.neg_samples_num,
                replacement=False,
            )                                                       # [S, neg_samples_num]
            sampled_intra_video_neg_mask = torch.zeros_like(fused_neg_sim)  # [S, B * P]
            sampled_intra_video_neg_mask[range(sampled_negative.shape[0]),
                                         sampled_negative.t()] = 1  # [S, B * P]
            assert sampled_intra_video_neg_mask.sum(dim=-1).eq(self.neg_samples_num).all()

            loss_intra_video = super(IntraContrastiveLoss, self).forward(
                intra_video_pos,                                    # [E]
                intra_video_all[ref_idx],                           # [E, B * P]
                sampled_intra_video_neg_mask[scatter_e2s],          # [E, B * P]
                self.t,
                self.m
            )

        # No dynamic negative sampling
        else:
            intra_video_neg_mask = ~pos_mask                        # [S, B * P]
            if false_neg_mask != None:
                intra_video_neg_mask[false_neg_mask] = 0                # remove false neg from neg mask

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
