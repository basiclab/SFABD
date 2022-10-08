import h5py
import torch

from src.datasets.base import CollateBase, aggregate_feats


class ActivityNet(CollateBase):
    def __init__(
        self,
        # annotation related
        ann_file,           # path to statically generated multi-target annotation file
        num_clips,          # number of final clips, e.g., 32 for Charades
        # feature related
        feat_file,          # path to feature file
        num_init_clips,     # number of initial clips. e.g., 64 for Charades
        **dummy,
    ):
        super().__init__(ann_file, num_clips)
        self.feat_file = feat_file
        self.num_init_clips = num_init_clips

    # override
    def get_feat(self, anno):
        with h5py.File(self.feat_file, 'r') as f:
            feats = f[anno['vid']]['c3d_features'][:]
            feats = torch.from_numpy(feats).float()

        return aggregate_feats(feats, self.num_init_clips, op_type='avg')
