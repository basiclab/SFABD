import h5py
import torch

from src.datasets.base import CollateBase


class Charades(CollateBase):
    def __init__(
        self,
        ann_file,           # path to annotation file (.json)
        feat_file,          # path to feature file
        num_init_clips,     # number of initial clips. e.g., 32 for Charades
    ):
        super().__init__(ann_file)
        self.feat_file = feat_file
        self.num_init_clips = num_init_clips

    # override
    def get_feat(self, anno):
        with h5py.File(self.feat_file, 'r') as f:
            feats = f[anno['vid']][:]
            feats = torch.from_numpy(feats).float()
        return feats
