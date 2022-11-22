import os

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import CollateBase


class QVHighlights(CollateBase):
    def __init__(
        self,
        ann_file,           # path to annotation file (.json)
        feat_dirs=[
            './data/QVHighlights/features/clip_features/',
            './data/QVHighlights/features/slowfast_features/',
        ],                  # path to feature directories
    ):
        super().__init__(ann_file)
        self.feat_dirs = feat_dirs

    def get_feat_dim(self):
        return 2816

    # override
    def get_feat(self, anno):
        feats = []
        min_len = 1000000
        for dir in self.feat_dirs:
            x = np.load(os.path.join(dir, anno['vid'] + '.npz'))['features']
            x = torch.from_numpy(x).float()
            x = F.normalize(x, dim=1)
            feats.append(x)
            min_len = min(min_len, x.shape[0])
        feats = [feat[:min_len] for feat in feats]
        feats = torch.cat(feats, dim=1)
        return feats


class QVHighlightsTrain(QVHighlights):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/train.json")


class QVHighlightsVal(QVHighlights):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/val.json")


# class QVHighlightsTest(QVHighlights):
#     def __init__():
#         super().__init__(ann_file="./data/QVHighlights/test.json")
