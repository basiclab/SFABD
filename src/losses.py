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
        scores2d: torch.Tensor,     # [S, N, N]
        iou2ds: torch.Tensor,       # [S, N, N]
        mask2d: torch.Tensor,       # [N, N]
    ):
        """
            B: (B)atch size
            N: (N)um clips
            S: number of (S)entences
            P: number of (P)roposals = number of 1 in mask2d
        """
        S, _, _ = scores2d.shape
        assert scores2d.shape == iou2ds.shape, f"{scores2d.shape} != {iou2ds.shape}"
        scores1d = scores2d.masked_select(mask2d).view(S, -1)   # [S, P]
        iou1d = iou2ds.masked_select(mask2d).view(S, -1)        # [S, P]
        iou1d = self.linear_scale(iou1d)                        # [S, P]
        loss = F.binary_cross_entropy(scores1d, iou1d)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        T_v: float = 0.1,
        T_q: float = 0.1,
        neg_video_iou: float = 0.5,
        pos_video_topk: int = 1,
        margin: float = 0,
    ):
        super().__init__()
        self.T_v = T_v                          # 0.1
        self.T_q = T_q                          # 0.1
        self.neg_video_iou = neg_video_iou      # 0.5
        self.pos_video_topk = pos_video_topk    # 1
        self.margin = margin

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
        num_targets: torch.Tensor,      # [B]
        iou2ds: torch.Tensor,           # [S, N, N]
        mask2d: torch.Tensor,           # [N, N]
    ):
        """
            B: (B)atch size
            C: (C)hannel
            N: (N)um clips
            S: number of (S)entences
            P: number of (P)roposals = the number 1 in mask2d
        """
        B, C, N, _ = video_feats.shape
        S = iou2ds.shape[0]
        P = mask2d.long().sum()
        K = self.pos_video_topk
        assert sum(num_targets) == S
        assert sents_feats.shape[0] == S
        device = video_feats.device

        # [B] -> [S]
        scatter_idx = torch.arange(B, device=device)                    # [B]
        scatter_idx = scatter_idx.repeat_interleave(num_targets)        # [S]

        video_feats = video_feats.masked_select(mask2d).view(B, C, -1)  # [B, C, P]
        video_feats = video_feats.permute(0, 2, 1)                      # [B, P, C]
        iou2ds = iou2ds.masked_select(mask2d).view(S, -1)               # [S, P]

        # normalize for cosine similarity
        video_feats = F.normalize(video_feats.contiguous(), dim=-1)     # [B, P, C]
        sents_feats = F.normalize(sents_feats.contiguous(), dim=-1)     # [S, C]

        # === inter video
        topk_idxs = iou2ds.topk(K, dim=1)[1]                    # [S, K]
        topk_idxs = topk_idxs.unsqueeze(-1).expand(-1, -1, C)   # [S, K, C]
        scatter_video_feats = video_feats[scatter_idx]          # [S, C, P]
        topk_video_feats = scatter_video_feats.gather(
            dim=1, index=topk_idxs)                             # [S, K, C]

        inter_video_pos = torch.mul(
            topk_video_feats,                                   # [S, K, C]
            sents_feats.unsqueeze(1)                            # [S, 1, C]
        ).sum(dim=-1)                                           # [S, K]

        inter_video_all = torch.matmul(
            topk_video_feats,                                   # [S, K, C]
            sents_feats.t(),                                    # [C, S]
        )                                                       # [S, K, S]
        mask = ~torch.eye(S, device=device).bool()              # [S, S]
        inter_video_neg_mask = mask.unsqueeze(1)                # [S, 1, S]

        loss_inter_video = self.log_cross_entropy(
            inter_video_pos,                                    # [S, K]
            inter_video_all,                                    # [S, K, S]
            inter_video_neg_mask,                               # [S, 1, S]
            self.T_v,
            self.margin,
        )

        # === inter query
        inter_query_pos = inter_video_pos               # [S, K]

        inter_query_all = torch.mm(
            sents_feats,                                # [S, C]
            video_feats.view(-1, C).t(),                # [C, B * P]
        ).unsqueeze(1)                                  # [S, 1, B * P]
        pos_mask = torch.eye(B, device=device).bool()   # [B, B]
        pos_mask = pos_mask.unsqueeze(-1)               # [B, B, 1]
        pos_mask = pos_mask.expand(-1, -1, P)           # [B, B, P]
        pos_mask = pos_mask.reshape(B, -1)              # [B, B * P]
        pos_mask = pos_mask[scatter_idx]                # [S, B * P]
        pos_mask[pos_mask.clone()] = \
            iou2ds.gt(self.neg_video_iou).view(-1)      # [S, P]
        inter_query_neg_mask = ~pos_mask.unsqueeze(1)   # [S, 1, B * P]

        loss_inter_query = self.log_cross_entropy(
            inter_query_pos,                            # [S, K]
            inter_query_all,                            # [S, 1, B * P]
            inter_query_neg_mask,                       # [S, 1, B * P]
            self.T_q,
            self.margin,
        )

        return loss_inter_video, loss_inter_query


if __name__ == '__main__':
    B = 8
    C = 256
    N = 16
    S = 15

    num_targets = torch.Tensor([1, 2, 2, 1, 3, 1, 3, 2]).long()
    video_feats = torch.randn(B, C, N, N)
    iou2ds = torch.rand(S, N, N)
    sents_feats = torch.randn(S, C)

    loss_fn = ContrastiveLoss(pos_video_topk=3)
    (
        loss_inter_video,
        loss_inter_query,
    ) = loss_fn(
        video_feats,
        sents_feats,
        iou2ds,
        num_targets,
    )
    print(
        loss_inter_video,
        loss_inter_query,
    )

    loss_fn = ScaledIoULoss(min_iou=0.1, max_iou=1)
    scores2d = torch.rand(B, N, N)
    iou2d = torch.rand(B, N, N)
    loss = loss_fn(scores2d, iou2d)
