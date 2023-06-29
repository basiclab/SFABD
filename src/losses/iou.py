import torch
import torch.nn as nn
import torch.nn.functional as F


# BCE loss
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


# BCE loss with false negative masking
class ScaledIoULossAFND(nn.Module):
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
