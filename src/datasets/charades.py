import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import CollateBase


# VGG feature
class CharadesVGG(CollateBase):
    def __init__(
        self,
        do_augmentation,
        mixup_alpha,
        downsampling_method,
        aug_prob,
        downsampling_prob,
        ann_file,           # path to annotation file (.json)
        feat_file,          # path to feature file
    ):
        super().__init__(
            ann_file,
            do_augmentation,
            mixup_alpha,
            downsampling_method,
            aug_prob,
            downsampling_prob,
        )
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


class CharadesSTAVGGTrain(CharadesVGG):
    def __init__(
        self,
        do_augmentation=False,
        mixup_alpha=0.9,
        downsampling_method='odd',
        aug_prob=0.5,
        downsampling_prob=0.5,
    ):
        super().__init__(
            do_augmentation,
            mixup_alpha,
            downsampling_method,
            aug_prob,
            downsampling_prob,
            ann_file="./data/CharadesSTA/train.json",
            feat_file="./data/CharadesSTA/VGG/vgg_rgb_features.hdf5",
        )


class CharadesSTAVGGTest(CharadesVGG):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/CharadesSTA/test.json",
            feat_file="./data/CharadesSTA/VGG/vgg_rgb_features.hdf5",
        )


class CharadesSTAVGGMultiTest(CharadesVGG):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/CharadesSTA/multi_test.json",
            feat_file="./data/CharadesSTA/VGG/vgg_rgb_features.hdf5",
        )


# C3D feature
class CharadesC3D(CollateBase):
    def __init__(
        self,
        do_augmentation,
        mixup_alpha,
        downsampling_method,
        aug_prob,
        downsampling_prob,
        ann_file,           # path to annotation file (.json)
        feat_dir,          # path to feature file
    ):
        super().__init__(
            ann_file,
            do_augmentation,
            mixup_alpha,
            downsampling_method,
            aug_prob,
            downsampling_prob,
        )
        self.feat_dir = feat_dir

    def get_feat_dim(self):
        return 4096

    # override
    # hdf5 version
    def get_feat(self, anno):
        with h5py.File(self.feat_dir, 'r') as f:
            feats = f[anno['vid']][:]
            feats = torch.from_numpy(feats).float()
            feats = F.normalize(feats, dim=-1)

        return feats


class CharadesSTAC3DTrain(CharadesC3D):
    def __init__(
        self,
        do_augmentation=False,
        mixup_alpha=0.9,
        downsampling_method='odd',
        aug_prob=0.5,
        downsampling_prob=0.5,
    ):
        super().__init__(
            do_augmentation,
            mixup_alpha,
            downsampling_method,
            aug_prob,
            downsampling_prob,
            ann_file="./data/CharadesSTA/train.json",
            feat_dir="./data/CharadesSTA/C3D/Charades_C3D.hdf5",
        )


class CharadesSTAC3DTest(CharadesC3D):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/CharadesSTA/test.json",
            feat_dir="./data/CharadesSTA/C3D/Charades_C3D.hdf5",
        )


class CharadesSTAC3DMultiTest(CharadesC3D):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/CharadesSTA/multi_test.json",
            feat_dir="./data/CharadesSTA/C3D/Charades_C3D.hdf5",   # .hdf5 file
        )


# I3D feature
class CharadesI3D(CollateBase):
    def __init__(
        self,
        do_augmentation,
        mixup_alpha,
        downsampling_method,
        aug_prob,
        downsampling_prob,
        ann_file,           # path to annotation file (.json)
        feat_dir,           # path to feature file
    ):
        super().__init__(
            ann_file,
            do_augmentation,
            mixup_alpha,
            downsampling_method,
            aug_prob,
            downsampling_prob,
        )
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
    def __init__(
        self,
        do_augmentation=False,
        mixup_alpha=0.9,
        downsampling_method='odd',
        aug_prob=0.5,
        downsampling_prob=0.5,
    ):
        super().__init__(
            do_augmentation,
            mixup_alpha,
            downsampling_method,
            aug_prob,
            downsampling_prob,
            ann_file="./data/CharadesSTA/train.json",
            feat_dir="./data/CharadesSTA/I3D/features/",
        )


class CharadesSTAI3DTest(CharadesI3D):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/CharadesSTA/test.json",
            feat_dir="./data/CharadesSTA/I3D/features/",
        )


class CharadesSTAI3DMultiTest(CharadesI3D):
    def __init__(self):
        super().__init__(
            do_augmentation=False,
            mixup_alpha=0.0,
            downsampling_method='None',
            aug_prob=0.0,
            downsampling_prob=0.0,
            ann_file="./data/CharadesSTA/multi_test.json",
            feat_dir="./data/CharadesSTA/I3D/features/",
        )
