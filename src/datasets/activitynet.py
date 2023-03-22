import json

import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.base import CollateBase
from src import dist


class ActivityNet(CollateBase):
    def __init__(
        self,
        ann_file,
        feat_file='./data/ActivityNet/activitynet_v1-3_c3d.hdf5'
    ):
        super().__init__(ann_file)
        self.feat_file = feat_file

    def parse_anno(self, ann_file):
        with open(ann_file, 'r') as f:
            raw_annos = json.load(f)

        annos = []
        pbar = tqdm(
            raw_annos.items(),
            ncols=0,
            leave=False,
            desc=self.__class__.__name__,
            disable=not dist.is_main()
        )
        for vid, video_data in pbar:
            duration = torch.tensor(video_data['duration'])
            sentences = []
            tgt_moments = []
            num_targets = []
            qids = []
            for timestamp, sentence in zip(
                    video_data['timestamps'], video_data['sentences']):
                timestamp = torch.Tensor(timestamp)
                timestamp = torch.clamp(timestamp / duration, 0, 1)
                if timestamp[0] <= timestamp[1]:
                    sentences.append(sentence)
                    tgt_moments.append(timestamp.view(1, 2))
                    num_targets.append(torch.tensor(1))
                    qids.append(0)

            if len(sentences) != 0:
                annos.append({
                    'vid': vid,
                    'sentences': sentences,
                    'num_sentences': torch.tensor(len(sentences)),
                    'num_targets': torch.stack(num_targets, dim=0),
                    'tgt_moments': torch.cat(tgt_moments, dim=0),
                    'duration': duration,
                    'qids': torch.tensor(qids),
                })
        pbar.close()

        return annos

    def get_feat_dim(self):
        return 500

    # override
    def get_feat(self, anno):
        with h5py.File(self.feat_file, 'r') as f:
            feat = f[anno['vid']]['c3d_features'][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
        return feat


class ActivityNetTrain(ActivityNet):
    def __init__(self):
        super().__init__(ann_file='./data/ActivityNet/train.json')


class ActivityNetVal(ActivityNet):
    def __init__(self):
        super().__init__(ann_file='./data/ActivityNet/val.json')


class ActivityNetTest(ActivityNet):
    def __init__(self):
        super().__init__(ann_file='./data/ActivityNet/test.json')
