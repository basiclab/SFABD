import os

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import CollateBase


class QVHighlights(CollateBase):
    def __init__(
        self,
        ann_file,               # path to annotation file (.json)
        feat_dirs,              # path to feature directories
        fallback_feat_dir=[
            './data/QVHighlights/features/clip_features/',
            './data/QVHighlights/features/slowfast_features/',
        ],                      # path to fallback feature directory
    ):
        super().__init__(ann_file)
        self.feat_dirs = feat_dirs
        self.fallback_feat_dir = fallback_feat_dir

        assert len(self.feat_dirs) == len(self.fallback_feat_dir)

    def get_feat_dim(self):
        return 2816  ## slowfast: 2304 + clip: 512

    # override
    def get_feat(self, anno):
        feats = []
        min_len = 1000000
        for dir, fallback_dir in zip(self.feat_dirs, self.fallback_feat_dir):
            path = os.path.join(dir, anno['vid'] + '.npz')
            if os.path.exists(path):
                x = np.load(path)['features']
            else:
                path = os.path.join(fallback_dir, anno['vid'] + '.npz')
                assert os.path.exists(path), f"fallback path {path} doest not exist"
                x = np.load(path)['features']
            x = torch.from_numpy(x).float()
            x = F.normalize(x, dim=1)
            feats.append(x)
            min_len = min(min_len, x.shape[0])
        feats = [feat[:min_len] for feat in feats]
        feats = torch.cat(feats, dim=1)
        return feats


class QVHighlights2s(QVHighlights):
    def __init__(self, ann_file):
        super().__init__(
            ann_file,
            feat_dirs=[
                './data/QVHighlights/features/clip_features/',
                './data/QVHighlights/features/slowfast_features/',
            ]
        )


class QVHighlightsTrain2s(QVHighlights2s):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/train.json")


class QVHighlightsVal2s(QVHighlights2s):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/val.json")


class QVHighlightsTest2s(QVHighlights2s):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/test.json")


class QVHighlights1s(QVHighlights):
    def __init__(self, ann_file):
        super().__init__(
            ann_file,
            feat_dirs=[
                './data/QVHighlights/features_1s/clip_features/',
                './data/QVHighlights/features_1s/slowfast_features/',
            ]
        )


class QVHighlightsTrain1s(QVHighlights1s):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/train.json")


class QVHighlightsVal1s(QVHighlights1s):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/val.json")


class QVHighlightsTest1s(QVHighlights1s):
    def __init__(self):
        super().__init__(ann_file="./data/QVHighlights/test.json")
