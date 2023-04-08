import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import CollateBase


class Charades(CollateBase):
    def __init__(
        self,
        ann_file,           # path to annotation file (.json)
        feat_file,          # path to feature file
    ):
        super().__init__(ann_file)
        self.feat_file = feat_file

    def get_feat_dim(self):
        return 4096

    # override
    def get_feat(self, anno):
        with h5py.File(self.feat_file, 'r') as f:
            feats = f[anno['vid']][:]
            feats = torch.from_numpy(feats).float()
            feats = F.normalize(feats, dim=-1)
        return feats


class CharadesSTATrain(Charades):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/train.json",
            feat_file="./data/CharadesSTA/vgg_rgb_features.hdf5")


class CharadesSTATest(Charades):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/test.json",
            feat_file="./data/CharadesSTA/vgg_rgb_features.hdf5")


class CharadesI3D(CollateBase):
    def __init__(
        self,
        ann_file,           # path to annotation file (.json)
        feat_dir,          # path to feature file
    ):
        super().__init__(ann_file)
        self.feat_dir = feat_dir

    def get_feat_dim(self):
        return 1024

    # override
    def get_feat(self, anno):
        path = os.path.join(self.feat_dir, f"{anno['vid']}.npy")
        feats = np.load(path, 'r')                  # [seq_len, 1, 1, 1024]
        feats = torch.from_numpy(feats.copy()).float()
        feats = feats.squeeze(1).squeeze(1)         # [seq_len, 1024]
        feats = F.normalize(feats, dim=-1)

        return feats


class CharadesSTAI3DTrain(CharadesI3D):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/charades_train.json",
            feat_dir="./data/CharadesSTA/I3D/features/")


class CharadesSTAI3DTest(CharadesI3D):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/charades_test.json",
            feat_dir="./data/CharadesSTA/I3D/features/")


class CharadesSTAI3DMultiTest(CharadesI3D):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/charades_multi_test.json",
            feat_dir="./data/CharadesSTA/I3D/features/")
