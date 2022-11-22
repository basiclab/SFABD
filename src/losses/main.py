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
        logits2d: torch.Tensor,     # [S, N, N]
        iou2d: torch.Tensor,        # [S, N, N]
        mask2d: torch.Tensor,       # [N, N]
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
        loss = F.binary_cross_entropy_with_logits(logits1d, iou1d)
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
            # === inter video
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
            # === inter query
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
            pos_mask[pos_mask.clone()] = s2v_pos_mask.view(-1)
            inter_query_neg_mask = ~pos_mask.unsqueeze(1)           # [S, 1, B * P]

            loss_inter_query = self.log_cross_entropy(
                inter_query_pos,                                    # [M, K]
                inter_query_all[scatter_m2s],                       # [M, 1, B * P]
                inter_query_neg_mask[scatter_m2s],                  # [M, 1, B * P]
                self.T_q,
                self.margin,
            )
        else:
            loss_inter_query = torch.tensor(0., device=device)

        if self.intra:
            # === intra video
            shift = 0
            combinations = []
            scatter_e2s = []
            for i, num in enumerate(num_targets):
                pairs = torch.ones(
                    num * K, num * K, device=device).nonzero()      # [num * K * num * K, 2]
                combinations.append(pairs + shift)
                scatter_e2s.append(torch.ones(len(pairs), device=device) * i)
                shift += num * K
            # E: number of (E)numerated positive pairs
            ref_idx, pos_idx = torch.cat(combinations, dim=0).t()   # [E], [E]
            scatter_e2s = torch.cat(scatter_e2s, dim=0).long()      # [E]
            assert (ref_idx < M * K).all()
            assert (pos_idx < M * K).all()

            pos_video_feats = topk_video_feats.reshape(M * K, C)    # [M * K, C]
            intra_video_pos = torch.mul(
                pos_video_feats[ref_idx],                           # [E, C]
                pos_video_feats[pos_idx],                           # [E, C]
            ).sum(dim=1)                                            # [E]

            intra_video_all = torch.mul(
                topk_video_feats.unsqueeze(2),                      # [M, K, 1, C]
                video_feats[scatter_m2v].unsqueeze(1),              # [M, 1, P, C]
            ).sum(dim=-1).reshape(M * K, -1)                        # [M * K, P]
            intra_video_neg_mask = iou2d < self.neg_iou             # [S, P]

            loss_intra_video = self.log_cross_entropy(
                intra_video_pos,                                    # [E]
                intra_video_all[ref_idx],                           # [E, P]
                intra_video_neg_mask[scatter_e2s],                  # [E, P]
                self.T_v,
            )
        else:
            loss_intra_video = torch.tensor(0., device=device)

        return loss_inter_video, loss_inter_query, loss_intra_video


if __name__ == '__main__':
    B = 4
    C = 256
    N = 32
    S = 8

    video_feats = torch.randn(B, C, N, N)
    sents_feats = torch.randn(S, C)
    num_sentences = torch.Tensor([1, 2, 3, 2]).long()
    num_targets = torch.Tensor([1, 2, 2, 1, 7, 1, 3, 2]).long()
    M = num_targets.sum().cpu().item()
    iou2d = torch.rand(S, N, N)
    iou2ds = torch.rand(M, N, N)
    mask2d = (torch.rand(N, N) > 0.5).triu().bool()

    loss_fn = ContrastiveLoss(pos_video_topk=3, intra=True)
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

    loss_fn = ScaledIoULoss(min_iou=0.1, max_iou=1)
    scores2d = torch.rand(B, N, N)
    iou2d = torch.rand(B, N, N)
    loss = loss_fn(scores2d, iou2d, mask2d)
