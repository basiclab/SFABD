import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        T_v: float = 0.1,
        T_q: float = 0.1,
        neg_iou: float = 0.5,
        pos_iou: float = 0.9,
        margin: float = 0,
        inter: bool = True,
        intra: bool = False,
        **dummy,
    ):
        super().__init__()
        self.T_v = T_v
        self.T_q = T_q
        self.neg_iou = neg_iou
        self.pos_iou = pos_iou
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
        if len(loss) != 0:
            return loss.mean()
        else:
            return torch.tensor(0.).to(pos_score.device)

    def forward(
        self,
        video_feats: torch.Tensor,      # [B, C, N, N]
        sents_feats: torch.Tensor,      # [S, C]
        num_sentences: torch.Tensor,    # [B]           number of sentences for each video
        iou2d: torch.Tensor,            # [S, N, N]     iou map for each sentence
        mask2d: torch.Tensor,           # [N, N]
        **dummy,
    ):
        """
            B: (B)atch size
            C: (C)hannel
            N: (N)um clips
            S: number of (S)entences in batch
            M: number of (M)oments in batch
            V: number of proposals in a (V)ideo
            P: number of (P)ositive proposals in batch
            E: number of (E)numerated intra-video positive pairs
        """
        device = video_feats.device
        B, C, N, _ = video_feats.shape
        S = num_sentences.sum().cpu().item()
        V = mask2d.long().sum().cpu().item()

        assert iou2d.shape == (S, N, N), f"{iou2d.shape} != {(S, N, N)}"

        # sentence idx -> video idx
        scatter_s2v = torch.arange(B).to(device).long()                 # [B]
        scatter_s2v = scatter_s2v.repeat_interleave(num_sentences)      # [S]

        # flatten video feats
        video_feats = video_feats.masked_select(mask2d).view(B, C, -1)  # [B, C, V]
        video_feats = video_feats.permute(0, 2, 1)                      # [B, V, C]
        # flatten iou2d
        iou2d = iou2d.masked_select(mask2d).view(S, -1)                 # [S, V]
        # normalize for cosine similarity
        video_feats = F.normalize(video_feats.contiguous(), dim=-1)     # [B, V, C]
        sents_feats = F.normalize(sents_feats.contiguous(), dim=-1)     # [S, C]

        # local positive mask
        pos_iou = iou2d.max(dim=1, keepdim=True).values - 1e-7          # [S, 1]
        pos_iou = pos_iou.clamp(max=self.pos_iou)                       # [S, 1]
        # print(pos_iou)
        pos_mask_local = iou2d > pos_iou                                # [S, V]
        # local negative mask
        neg_mask_local = iou2d < self.neg_iou                           # [S, V]

        # local mask of global mask
        global_local_mask = torch.eye(B).to(device).bool()              # [B, B]
        global_local_mask = global_local_mask.unsqueeze(-1)             # [B, B, 1]
        global_local_mask = global_local_mask.expand(-1, -1, V)         # [B, B, V]
        global_local_mask = global_local_mask.reshape(B, -1)            # [B, B * V]
        global_local_mask = global_local_mask[scatter_s2v]              # [S, B * V]

        # global positive mask
        pos_mask_global = global_local_mask.clone()                     # [S, B * V]
        pos_mask_global[global_local_mask] = pos_mask_local.view(-1)    # [S, B * V]
        # global negative mask
        neg_mask_global = ~global_local_mask.clone()                    # [S, B * V]
        neg_mask_global[global_local_mask] = neg_mask_local.view(-1)    # [S, B * V]

        # positive proposals features
        video_setns_feats = video_feats[scatter_s2v]                    # [S, V, C]
        pos_proposals_feats = video_setns_feats[pos_mask_local]         # [P, C]
        num_positives = pos_mask_local.long().sum(dim=1)                # [S]
        # print(num_positives)
        P = num_positives.sum().cpu().item()

        # positive idx -> sentence idx
        scatter_p2s = torch.arange(S).to(device).long()                 # [S]
        scatter_p2s = scatter_p2s.repeat_interleave(num_positives)      # [P]

        if self.inter:
            # === inter video
            inter_video_pos = torch.mul(
                pos_proposals_feats,                                    # [P, C]
                sents_feats[scatter_p2s]                                # [P, C]
            ).sum(dim=-1)                                               # [P]
            inter_video_all = torch.matmul(
                pos_proposals_feats,                                    # [P, C]
                sents_feats.t(),                                        # [C, S]
            )                                                           # [P, S]
            inter_video_neg_mask = ~torch.eye(S).to(device).bool()      # [S, S]
            inter_video_neg_mask = inter_video_neg_mask[scatter_p2s]    # [P, S]
            loss_inter_video = self.log_cross_entropy(
                inter_video_pos,                                        # [P]
                inter_video_all,                                        # [P, S]
                inter_video_neg_mask,                                   # [P, S]
                self.T_v,
                self.margin,
            )
        else:
            loss_inter_video = torch.tensor(0.).to(device)

        if self.inter:
            # === inter query
            inter_query_pos = inter_video_pos                           # [P]
            inter_query_all = torch.mm(
                sents_feats,                                            # [S, C]
                video_feats.view(-1, C).t(),                            # [C, B * V]
            )                                                           # [S, B * V]
            inter_query_neg_mask = neg_mask_global                      # [S, B * V]
            loss_inter_query = self.log_cross_entropy(
                inter_query_pos,                                        # [P]
                inter_query_all[scatter_p2s],                           # [P, B * V]
                inter_query_neg_mask[scatter_p2s],                      # [P, B * V]
                self.T_q,
                self.margin,
            )
        else:
            loss_inter_query = torch.tensor(0.).to(device)

        if self.intra:
            # === intra video
            pairs = []
            num_pairs = []
            shift = 0
            for num_p in num_positives:
                not_I = ~torch.eye(num_p).to(device).bool()             # [num_p, num_p]
                pairs_of_video = not_I.nonzero()
                pairs.append(pairs_of_video + shift)
                num_pairs.append(len(pairs_of_video))
                shift += num_p
            num_pairs = torch.tensor(num_pairs).to(device)
            assert shift == P and len(num_pairs) == S
            scatter_e2s = torch.arange(S).to(device).long()             # [S]
            scatter_e2s = scatter_e2s.repeat_interleave(num_pairs)      # [E]
            ref_idx, pos_idx = torch.cat(pairs, dim=0).t()              # [E], [E]
            assert (ref_idx < P).all() and (pos_idx < P).all()

            intra_video_pos = torch.mul(
                pos_proposals_feats[ref_idx],                           # [E, C]
                pos_proposals_feats[pos_idx],                           # [E, C]
            ).sum(dim=-1)                                               # [E]
            intra_video_all = torch.matmul(
                pos_proposals_feats,                                    # [P, C]
                video_feats.view(-1, C).t(),                            # [C, B * V]
            )                                                           # [P, B * V]
            intra_video_neg_mask = neg_mask_global                      # [S, B * V]

            loss_intra_video = self.log_cross_entropy(
                intra_video_pos,                                        # [E]
                intra_video_all[ref_idx],                               # [E, B * V]
                intra_video_neg_mask[scatter_e2s],                      # [E, B * V]
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
    iou2d = torch.rand(S, N, N) / 10
    mask2d = (torch.rand(N, N) > 0.5).triu().bool()

    loss_fn = ContrastiveLoss(
        T_v=0.1,
        T_q=0.1,
        neg_iou=0.5,
        pos_iou=0.975,
        margin=0.3,
        inter=True,
        intra=True,
    )
    loss_inter_video, loss_inter_query, loss_intra_video = loss_fn(
        video_feats.cuda(),
        sents_feats.cuda(),
        num_sentences.cuda(),
        iou2d.cuda(),
        mask2d.cuda(),
    )
    print(loss_inter_video, loss_inter_query, loss_intra_video)
