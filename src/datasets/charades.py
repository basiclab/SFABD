import h5py
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


class CharadesMultitargetTrain(Charades):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/train_multitarget.json",
            feat_file="./data/CharadesSTA/Charades_C3D.hdf5")


class CharadesMultitargetTest(Charades):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/test_multitarget.json",
            feat_file="./data/CharadesSTA/Charades_C3D.hdf5")
