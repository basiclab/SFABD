import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledIoULoss(nn.Module):
    def __init__(self, min_iou, max_iou):
        super().__init__()
        self.min_iou = min_iou
        self.max_iou = max_iou

    def linear_scale(self, iou: torch.Tensor):
        return iou.sub(self.min_iou).div(self.max_iou - self.min_iou).clamp(0, 1)

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
        mask = torch.ones(D, D, device=device).bool().triu()    # [D, D]

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
        self.intra = intra
        self.inter = inter

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

    def gen_scatter_idx(self, num_targets):
        B, = num_targets.shape
        idx = torch.arange(B, device=num_targets.device)                # [B]
        idx = idx.repeat_interleave(num_targets, dim=0)                 # [T]
        return idx

    def forward(
        self,
        video_feats: torch.Tensor,      # [B, C, D, D]
        query_feats: torch.Tensor,      # [B, C]
        sents_feats: torch.Tensor,      # [T, C]
        iou2d: torch.Tensor,            # [B, D, D]
        iou2ds: torch.Tensor,           # [T, D, D]
        num_targets: torch.Tensor,      # [B]    ex. [1, 2, 1, 3, ...]
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
        device = video_feats.device

        # [B] -> [T]
        scatter_idx = self.gen_scatter_idx(num_targets)                 # [T]

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

        if self.inter:
            # === inter video
            inter_video_pos = torch.mul(
                topk_video_feats,                           # [T, K, C]
                query_feats[scatter_idx].unsqueeze(1)       # [T, 1, C]
            ).sum(dim=-1)                                   # [T, K]

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
        else:
            losses.append(torch.tensor(0, device=device))

        if self.inter:
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
        else:
            losses.append(torch.tensor(0, device=device))

        if self.intra:
            # === intra video
            combinations = []
            shift = 0
            for num in num_targets:
                pairs = torch.ones(
                    num * K, num * K, device=device).nonzero()      # [num * K * num * K, 2]
                combinations.append(pairs + shift)
                shift += num * K
            # E: number of (E)numerated positive pairs
            ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
            topk_video_feats = topk_video_feats.reshape(-1, C)      # [T * K, C]
            intra_video_pos = torch.mul(
                topk_video_feats[ref_idx],                          # [E, C]
                topk_video_feats[pos_idx],                          # [E, C]
            ).sum(dim=1)                                            # [E]

            # [B] -> [T * K]
            scatter_topk_idx = \
                scatter_idx.repeat_interleave(K)                    # [T * K]
            intra_video_all = torch.mul(
                topk_video_feats.unsqueeze(1),                      # [T * K, 1, C]
                video_feats[scatter_topk_idx],                      # [T * K, P, C]
            ).sum(dim=2)                                            # [T * K, P]
            neg_mask = iou2d < self.neg_video_iou                   # [B, P]
            intra_video_neg_mask = neg_mask[scatter_topk_idx]       # [T * K, P]

            loss_intra_video = self.log_cross_entropy(
                intra_video_pos,                                    # [E]
                intra_video_all[ref_idx],                           # [E, P]
                intra_video_neg_mask[ref_idx],                      # [E, P]
                self.T_v,
            )
            losses.append(loss_intra_video)
        else:
            losses.append(torch.tensor(0, device=device))

        if self.intra and (num_targets > 1).any():
            # === intra query
            multi_query_mask = num_targets > 1                      # [B]
            multi_sents_mask = multi_query_mask[scatter_idx]        # [T]
            query_feats = query_feats[multi_query_mask]             # [B', C]
            sents_feats = sents_feats[multi_sents_mask]             # [T', C]
            num_targets = num_targets[multi_query_mask]             # [B']
            scatter_idx = self.gen_scatter_idx(num_targets)         # [B'] -> [T']

            intra_query_pos = torch.mul(
                sents_feats,                                        # [T', C]
                query_feats[scatter_idx],                           # [T', C]
            ).sum(dim=1)                                            # [T']

            intra_query_all = torch.mm(
                query_feats,                                        # [B', C]
                sents_feats.t()                                     # [C, T']
            )                                                       # [B', T']
            B_, T_ = intra_query_all.shape
            neg_mask = ~torch.eye(B_, device=device).bool()         # [B', B']
            intra_query_neg_mask = neg_mask.repeat_interleave(
                num_targets, dim=1)                                 # [B', T']
            loss_intra_query = self.log_cross_entropy(
                intra_query_pos,                                    # [T']
                intra_query_all[scatter_idx],                       # [T', B']
                intra_query_neg_mask[scatter_idx],                  # [T', B']
                self.T_q,
            )
            losses.append(loss_intra_query)
        else:
            losses.append(torch.tensor(0, device=device))

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
    )
    print(
        loss_inter_video,
        loss_inter_query,
        loss_intra_video,
        loss_intra_query,
    )

    loss_fn = ScaledIoULoss(min_iou=0.1, max_iou=1)
    scores2d = torch.rand(B, D, D)
    iou2d = torch.rand(B, D, D)
    loss = loss_fn(scores2d, iou2d)
