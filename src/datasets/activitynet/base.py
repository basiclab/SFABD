from math import ceil, floor

import h5py
import torch

from src.datasets.charades.base import Charades, aggregate_feats


class ActivityNet(Charades):
    # override
    def get_feat(self, anno):
        vids = anno['vids']
        durations = anno['durations']
        timestamps = anno['timestamps']
        assert len(vids) == len(timestamps)

        with h5py.File(self.feat_file, 'r') as f:
            feats = []
            for vid, duration, (st, ed) in zip(vids, durations, timestamps):
                feat = f[vid]['c3d_features']
                st_idx = floor(st / duration * len(feat))
                ed_idx = ceil(ed / duration * len(feat))
                feat = feat[st_idx: ed_idx]
                feats.append(torch.from_numpy(feat).float())
            feats = torch.cat(feats, dim=0)

        return aggregate_feats(feats, self.num_init_clips, op_type='avg')
