import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledIoULoss(nn.Module):
    def __init__(self, min_iou, max_iou, weight=1.0, *args, **kwargs,):
        super().__init__()
        self.min_iou = min_iou
        self.max_iou = max_iou
        self.weight = weight

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
        device = logits2d.device
        if self.weight == 0:
            zero = torch.tensor(0., device=device)
            return zero, {'loss/iou': zero}

        S, _, _ = logits2d.shape
        assert logits2d.shape == iou2d.shape, f"{logits2d.shape} != {iou2d.shape}"
        logits1d = logits2d.masked_select(mask2d).view(S, -1)   # [S, P]
        iou1d = iou2d.masked_select(mask2d).view(S, -1)         # [S, P]
        iou1d = self.linear_scale(iou1d)                        # [S, P]
        loss = F.binary_cross_entropy_with_logits(logits1d, iou1d)
        return (
            loss * self.weight,
            {
                'loss/iou': loss,
            }
        )


class ScaledIoULossDNS(nn.Module):
    def __init__(self, min_iou, max_iou, weight=1.0, *args, **kwargs,):
        super().__init__()
        self.min_iou = min_iou
        self.max_iou = max_iou
        self.weight = weight

    def linear_scale(self, iou: torch.Tensor):
        return iou.sub(self.min_iou).div(self.max_iou - self.min_iou).clamp(0, 1)

    def forward(
        self,
        logits2d: torch.Tensor,                 # [S, N, N]
        iou2d: torch.Tensor,                    # [S, N, N]
        mask2d: torch.Tensor,                   # [N, N]
        false_neg_mask: torch.Tensor = None,    # [S, P]
    ):
        """
            B: (B)atch size
            N: (N)um clips
            S: number of (S)entences
            P: number of (P)roposals = number of 1 in mask2d
        """
        device = logits2d.device
        if self.weight == 0:
            zero = torch.tensor(0., device=device)
            return zero, {'loss/iou': zero}

        S, _, _ = logits2d.shape
        assert logits2d.shape == iou2d.shape, f"{logits2d.shape} != {iou2d.shape}"
        logits1d = logits2d.masked_select(mask2d).view(S, -1)   # [S, P]
        iou1d = iou2d.masked_select(mask2d).view(S, -1)         # [S, P]
        iou1d = self.linear_scale(iou1d)                        # [S, P]

        if false_neg_mask is not None:
            loss = F.binary_cross_entropy_with_logits(logits1d, iou1d, reduction='none')
            # ignore false neg
            loss[false_neg_mask] = 0                                # [S, P]
            loss = loss.sum() / (~false_neg_mask).sum()
        else:
            loss = F.binary_cross_entropy_with_logits(logits1d, iou1d)

        return (
            loss * self.weight,
            {
                'loss/iou': loss,
            }
        )


class ScaledIoUFocalLoss(nn.Module):
    def __init__(
        self,
        min_iou: float,
        max_iou: float,
        alpha: float,
        gamma: float,
        weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.min_iou = min_iou
        self.max_iou = max_iou
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

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
        device = logits2d.device
        if self.weight == 0:
            zero = torch.tensor(0., device=device)
            return zero, {'loss/iou': zero}

        S, _, _ = logits2d.shape
        assert logits2d.shape == iou2d.shape, f"{logits2d.shape} != {iou2d.shape}"
        logits1d = logits2d.masked_select(mask2d).view(S, -1)   # [S, P]
        scores1d = torch.sigmoid(logits1d)                      # [S, P]
        iou1d = iou2d.masked_select(mask2d).view(S, -1)         # [S, P]
        iou1d = self.linear_scale(iou1d)                        # [S, P]

        # Focal Loss
        # -(y * \alpha * (1-p)^\gamma * \log(p) + (1 - y) * (1 - \alpha) * p^\gamma * \log(1-p))
        y = iou1d
        p = scores1d
        loss = \
            - y * self.alpha * (1 - p).pow(self.gamma) * p.log() \
            - (1 - y) * (1 - self.alpha) * p.pow(self.gamma) * (1 - p).log()    # [S, P]
        loss = loss.mean()

        return (
            loss * self.weight,
            {
                'loss/iou': loss,
            }
        )


if __name__ == '__main__':
    N = 64
    S = 8

    scores2d = torch.rand(S, N, N)
    iou2d = torch.rand(S, N, N)
    mask2d = (torch.rand(N, N) > 0.5).triu().bool()

    loss_fn = ScaledIoUFocalLoss(
        min_iou=0.5,
        max_iou=1.0,
        alpha=0.25,
        gamma=2,
    )
    loss = loss_fn(scores2d, iou2d, mask2d)
    print(f"Focal loss: {loss.item()}")

    # test iou loss
    loss_fn = ScaledIoULoss(
        min_iou=0.5,
        max_iou=1.0,
    )
    loss = loss_fn(scores2d, iou2d, mask2d)
    print(f"BCE loss  : {loss.item()}")
