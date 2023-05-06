import os

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import CollateBase


class QVHighlights(CollateBase):
    def __init__(
        self,
        do_augmentation,
        mixup_alpha,
        aug_expand_rate,
        downsampling_method,
        aug_prob,
        downsampling_prob,
        ann_file,               # path to annotation file (.json)
        feat_dirs,              # path to feature directories
    ):
        super().__init__(
            ann_file,
            do_augmentation,
            mixup_alpha,
            aug_expand_rate,
            downsampling_method,
            aug_prob,
            downsampling_prob,
        )
        self.feat_dirs = feat_dirs

    def get_feat_dim(self):
        return 2816     # slowfast: 2304 + clip: 512

    # override
    def get_feat(self, anno):
        feats = []
        min_len = 1000000
        for dir in self.feat_dirs:
            path = os.path.join(dir, anno['vid'] + '.npz')
            x = np.load(path)['features']
            x = torch.from_numpy(x).float()
            x = F.normalize(x, dim=1)
            feats.append(x)
            min_len = min(min_len, x.shape[0])
        feats = [feat[:min_len] for feat in feats]
        feats = torch.cat(feats, dim=1)
        return feats


class QVHighlightsTrain(QVHighlights):
    def __init__(
        self,
        do_augmentation=False,
        mixup_alpha=0.9,
        aug_expand_rate=1.0,
        downsampling_method='odd',
        aug_prob=0.5,
        downsampling_prob=0.5,
    ):
        super().__init__(
            do_augmentation,
            mixup_alpha,
            aug_expand_rate,
            downsampling_method,
            aug_prob,
            downsampling_prob,
            ann_file="./data/QVHighlights/train.json",
            feat_dirs=[
                './data/QVHighlights/features/clip_features/',
                './data/QVHighlights/features/slowfast_features/',
            ]
        )


class QVHighlightsVal(QVHighlights):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            aug_expand_rate=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/QVHighlights/val.json",
            feat_dirs=[
                './data/QVHighlights/features/clip_features/',
                './data/QVHighlights/features/slowfast_features/',
            ]
        )


class QVHighlightsTest(QVHighlights):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            aug_expand_rate=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/QVHighlights/test.json",
            feat_dirs=[
                './data/QVHighlights/features/clip_features/',
                './data/QVHighlights/features/slowfast_features/',
            ]
        )
