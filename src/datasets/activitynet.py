import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.base import CollateBase
from src import dist


class ActivityNetC3D(CollateBase):
    def __init__(
        self,
        ann_file,
        feat_file='./data/ActivityNet/C3D/activitynet_v1-3_c3d.hdf5'
    ):
        super().__init__(ann_file)
        self.feat_file = feat_file

    def get_feat_dim(self):
        return 500

    # override
    def get_feat(self, anno):
        with h5py.File(self.feat_file, 'r') as f:
            feats = f[anno['vid']]['c3d_features'][:]
            feats = torch.from_numpy(feats).float()
            feats = F.normalize(feats, dim=-1)
        return feats


class ActivityNetC3DTrain(ActivityNetC3D):
    def __init__(self):
        super().__init__(ann_file='./data/ActivityNet/train.json')


class ActivityNetC3DVal(ActivityNetC3D):
    def __init__(self):
        super().__init__(ann_file='./data/ActivityNet/val.json')


class ActivityNetC3DTest(ActivityNetC3D):
    def __init__(self):
        super().__init__(ann_file='./data/ActivityNet/test.json')


class ActivityNetC3DMultiTest(ActivityNetC3D):
    def __init__(self):
        super().__init__(ann_file='./data/ActivityNet/multi_test.json')


class ActivityNetI3D(CollateBase):
    def __init__(
        self,
        ann_file,
        feat_dir="./data/ActivityNet/I3D/"
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


class ActivityNetI3DTrain(ActivityNetI3D):
    def __init__(self):
        super().__init__(
            ann_file='./data/ActivityNet/train.json')


class ActivityNetI3DVal(ActivityNetI3D):
    def __init__(self):
        super().__init__(
            ann_file='./data/ActivityNet/val.json')


class ActivityNetI3DTest(ActivityNetI3D):
    def __init__(self):
        super().__init__(
            ann_file='./data/ActivityNet/test.json')


class ActivityNetI3DMultiTest(ActivityNetI3D):
    def __init__(self):
        super().__init__(
            ann_file='./data/ActivityNet/multi_test.json')
