import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledBCELoss(nn.Module):
    def __init__(self, min_iou, max_iou):
        super().__init__()
        self.min_out = min_iou
        self.max_out = max_iou

    def linear_scale(self, iou):
        return (iou - self.min_out) / (self.max_out - self.min_out)

    def forward(
        self,
        scores2d: torch.Tensor,   # [B, D, D]
        iou2d: torch.Tensor,    # [B, D, D]
    ):
        """
            B: (B)atch size
            D: (D)imension of prorposal matrix
            P: number of (P)roposals = the number of upper triangular elements
            P = D * (D + 1) / 2
        """
        assert scores2d.shape == iou2d.shape, f"{scores2d.shape} != {iou2d.shape}"
        B, D, _ = scores2d.shape
        device = scores2d.device
        ones = torch.ones(D, D, device=device).bool()           # [D, D]
        mask = torch.triu(ones, diagonal=0)                     # [D, D]

        inputs = scores2d.masked_select(mask).view(B, -1)       # [B, P]
        target = iou2d.masked_select(mask).view(B, -1)          # [B, P]
        target = self.linear_scale(target)                      # [B, P]
        loss = F.binary_cross_entropy(inputs, target)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        T_v: float = 0.1,
        T_q: float = 0.1,
        neg_video_iou: float = 0.5,
        pos_video_topk: int = 1,
        intra: bool = True,
        inter: bool = True,
    ):
        super().__init__()
        self.T_v = T_v                          # 0.1
        self.T_q = T_q                          # 0.1
        self.neg_video_iou = neg_video_iou      # 0.5
        self.pos_video_topk = pos_video_topk    # 1
        self.intra = intra                      # TODO:
        self.inter = inter                      # TODO:

    def log_cross_entropy(
        self,
        pos_score: torch.Tensor,                # [...]
        all_score: torch.Tensor,                # [..., Number_of_samples]
        neg_mask: torch.Tensor,                 # [..., Number_of_samples]
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
        video_feats: torch.Tensor,      # [B, C, D, D]
        query_feats: torch.Tensor,      # [B, C]
        sents_feats: torch.Tensor,      # [T, C]
        iou2d: torch.Tensor,            # [B, D, D]
        iou2ds: torch.Tensor,           # [T, D, D]
        num_targets: torch.Tensor,      # [B]    ex. [1, 2, 1, 3, ...]
        scatter_idx: torch.Tensor,      # [T]    ex. [0, 1, 1, 2, 3, 3, 3, ...]
    ):
        """
            B: (B)atch size
            C: (C)hannel
            D: (D)imension of prorposal matrix
            T: number of (T)argets in batch
            P: number of (P)roposals = the number of upper triangular elements
        """
        B, C, D, _ = video_feats.shape      # (B)atch, (C)hannel, (D)imension of prorposal matrix
        P = (D + 1) * D // 2                # number of (P)roposals
        T = iou2ds.shape[0]                 # number of (T)argets in batch
        K = self.pos_video_topk             # positive top(K) proposals
        assert sum(num_targets) == T
        assert sents_feats.shape[0] == T
        assert scatter_idx.shape[0] == T
        device = video_feats.device

        # upper triangular mask
        ones = torch.ones(D, D, device=device).bool()                   # [D, D]
        mask = torch.triu(ones, diagonal=0)                             # [D, D]
        video_feats = video_feats.masked_select(mask).view(B, C, -1)    # [B, C, P]
        video_feats = video_feats.permute(0, 2, 1)                      # [B, P, C]
        iou2d = iou2d.masked_select(mask).view(B, -1)                   # [B, P]
        iou2ds = iou2ds.masked_select(mask).view(T, -1)                 # [T, P]

        # normalize for cosine similarity
        video_feats = F.normalize(video_feats.contiguous(), dim=-1)     # [B, P, C]
        query_feats = F.normalize(query_feats.contiguous(), dim=-1)     # [B, C]
        sents_feats = F.normalize(sents_feats.contiguous(), dim=-1)     # [T, C]

        topk_idxs = iou2ds.topk(K, dim=1)[1]                            # [T, K]
        topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)           # [T, K, C]
        scatter_video_feats = video_feats[scatter_idx]                  # [T, C, P]
        topk_video_feats = scatter_video_feats.gather(
            dim=1, index=topk_idxs)                                     # [T, K, C]

        losses = []

        # === inter video
        inter_video_pos = torch.bmm(
            topk_video_feats,                           # [T, K, C]
            query_feats[scatter_idx].unsqueeze(-1)      # [T, C, 1]
        ).squeeze(-1)                                   # [T, K]
        inter_video_all = torch.matmul(
            topk_video_feats,                           # [T, K, C]
            query_feats.t(),                            # [C, B]
        )                                               # [T, K, B]
        neg_mask = ~torch.eye(B, device=device).bool()  # [B, B]
        neg_mask = neg_mask[scatter_idx]                # [T, B]
        inter_video_neg_mask = neg_mask.unsqueeze(1)    # [T, 1, B]

        loss_inter_video = self.log_cross_entropy(
            inter_video_pos,                            # [T, K]
            inter_video_all,                            # [T, K, B]
            inter_video_neg_mask,                       # [T, 1, B]
            self.T_v,
        )
        losses.append(loss_inter_video)

        # === inter query
        inter_query_pos = inter_video_pos               # [T, K]
        inter_query_all = torch.mm(
            query_feats,                                # [B, C]
            video_feats.view(-1, C).t(),                # [C, B * P]
        ).unsqueeze(1)                                  # [B, 1, B * P]
        inter_query_all = inter_query_all[scatter_idx]  # [T, 1, B * P]
        pos_mask = torch.eye(B, device=device).bool()   # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)               # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)           # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)              # [B, B * P]
        pos_mask[pos_mask.clone()] = \
            iou2d.gt(self.neg_video_iou).view(-1)       # [B, P]
        neg_mask = ~pos_mask.unsqueeze(1)               # [B, 1, B * P]
        inter_query_neg_mask = neg_mask[scatter_idx]    # [T, 1, B * P]

        loss_inter_query = self.log_cross_entropy(
            inter_query_pos,                            # [T, K]
            inter_query_all,                            # [T, 1, B * P]
            inter_query_neg_mask,                       # [T, 1, B * P]
            self.T_q,
        )
        losses.append(loss_inter_query)

        # === intra video
        # E: (E)numerate positive pairs in `topk_video_feats`
        combinations = []
        shift = 0
        for num in num_targets:
            combinations.append(
                torch.cartesian_prod(
                    torch.arange(num * K, device=device),
                    torch.arange(num * K, device=device),
                ) + shift)                                      # [num * K * num * K, 2]
            shift += num * K
        a, b = torch.cat(combinations, dim=0).t()               # [E], [E]
        topk_video_feats = topk_video_feats.reshape(-1, C)      # [T * K, C]
        intra_video_pos = torch.mul(
            topk_video_feats[a],                                # [E, C]
            topk_video_feats[b],                                # [E, C]
        ).sum(dim=1)                                            # [E]
        scatter_topk_idx = \
            scatter_idx.repeat_interleave(K)                    # [T * K]
        intra_video_all = torch.mul(
            topk_video_feats.unsqueeze(1),                      # [T * K, 1, C]
            video_feats[scatter_topk_idx],                      # [T * K, P, C]
        ).sum(dim=2)                                            # [T * K, P]
        intra_video_neg_mask = iou2ds < self.neg_video_iou      # [T, P]
        intra_video_neg_mask = \
            intra_video_neg_mask.repeat_interleave(K, dim=0)    # [T * K, P]

        loss_intra_video = self.log_cross_entropy(
            intra_video_pos,                                    # [E]
            intra_video_all[a],                                 # [E, P]
            intra_video_neg_mask[a],                            # [E, P]
            self.T_v,
        )
        losses.append(loss_intra_video)

        # === intra query
        if (num_targets > 0).any():
            multi_query_mask = num_targets > 1                      # [B]
            multi_sents_mask = multi_query_mask.repeat_interleave(
                repeats=num_targets, dim=0)                         # [T]
            masked_sents_feats = sents_feats[multi_sents_mask]      # [T', C]
            masked_scatter_idx = scatter_idx[multi_sents_mask]      # [T']
            masked_num_targets = num_targets[multi_query_mask]      # [B']
            intra_query_pos = torch.mul(
                masked_sents_feats,                                 # [T', C]
                query_feats[masked_scatter_idx],                    # [T', C]
            ).sum(dim=1)                                            # [T']
            intra_query_all = torch.mm(
                masked_sents_feats,                                 # [T', C]
                query_feats[multi_query_mask].t()                   # [C, B']
            )                                                       # [T', B']
            T_, B_ = intra_query_all.shape
            pos_mask = torch.eye(B_, device=device).bool()          # [B', B']
            pos_mask = pos_mask.repeat_interleave(
                masked_num_targets, dim=0)                          # [T', B']
            intra_query_neg_mask = ~pos_mask                        # [T', B']
            loss_intra_query = self.log_cross_entropy(
                intra_query_pos,                                    # [T']
                intra_query_all,                                    # [T', B']
                intra_query_neg_mask,                               # [T', B']
                self.T_q,
            )
        else:
            loss_intra_query = torch.zeros((), device=device)
        losses.append(loss_intra_query)

        # loss_inter_video, loss_inter_query, loss_intra_video, loss_intra_query
        return losses


if __name__ == '__main__':
    B = 8
    C = 256
    D = 16
    T = 15

    num_targets = torch.Tensor([1, 2, 2, 1, 3, 1, 3, 2]).long()
    scatter_idx = torch.arange(B).repeat_interleave(num_targets)
    video_feats = torch.randn(B, C, D, D)
    query_feats = torch.randn(B, C)
    iou2d = torch.rand(B, D, D)
    iou2ds = torch.rand(T, D, D)
    sents_feats = torch.randn(T, C)

    loss_fn = ContrastiveLoss(pos_video_topk=3)
    (
        loss_inter_video,
        loss_inter_query,
        loss_intra_video,
        loss_intra_query,
    ) = loss_fn(
        video_feats,
        query_feats,
        sents_feats,
        iou2d,
        iou2ds,
        num_targets,
        scatter_idx,
    )

    loss_fn = ScaledBCELoss(min_iou=0.1, max_iou=1)
    scores2d = torch.rand(B, D, D)
    iou2d = torch.rand(B, D, D)
    loss = loss_fn(scores2d, iou2d)
