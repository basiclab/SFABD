from math import ceil, floor
import json

import h5py
import torch
from tqdm import tqdm

from src.datasets.base import CollateBase, aggregate_feats
from src.utils import moment_to_iou2d, moments_to_iou2d


class CharadesBase(CollateBase):
    def __init__(
        self,
        feat_file,          # path to feature file
        num_init_clips,     # number of small basic clips features. 32 for Charades (now 64)
    ):
        super().__init__()
        self.feat_file = feat_file
        self.num_init_clips = num_init_clips

    # override
    def get_feat(self, anno):
        vids = anno['vids']
        durations = anno['durations']
        timestamps = anno['timestamps']
        assert len(vids) == len(timestamps)

        with h5py.File(self.feat_file, 'r') as f:
            feats = []
            for vid, duration, (st, ed) in zip(vids, durations, timestamps):
                feat = f[vid]
                st_idx = floor(st / duration * len(feat))
                ed_idx = ceil(ed / duration * len(feat))
                feat = feat[st_idx: ed_idx]
                feats.append(torch.from_numpy(feat).float())
            feats = torch.cat(feats, dim=0)

        return aggregate_feats(feats, self.num_init_clips, op_type='avg')


class Charades(CharadesBase):
    """Statically sampled multi-target Charades dataset.
    Accepted format:
    [
        {
            "video": [
                "VZY0C",
            ],
            "timestamps":[
                [0.6, 7.6],
                [18.0, 24.8]
            ],
            "sentences": [
                "a person was running to go get the groceries.",
                "a person runs into the room."
            ],
            "duration": 39.5,
            "query": "a person is running"
        },
        ...
    ]
    """
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
        super().__init__(feat_file, num_init_clips)
        self.annos = self.parse_anno(ann_file, num_clips)

    # override
    def __len__(self):
        return len(self.annos)

    # override
    def get_anno(self, idx):
        return self.annos[idx]

    def parse_anno(self, ann_file, num_clips):
        with open(ann_file, 'r') as f:
            raw_annos = json.load(f)

        annos = []
        desc = self.__class__.__name__
        with tqdm(raw_annos, ncols=0, leave=False, desc=desc) as pbar:
            for anno in pbar:
                timestamps = anno['timestamps']
                sentences = anno['sentences']
                duration = anno['duration']     # video length
                moments = []                    # start and end time of each moment
                iou2ds = []                     # iou2d for each mement
                sents = []                      # sentence for each moment
                for (start, end), sent in zip(timestamps, sentences):
                    moment = torch.Tensor([max(start, 0), min(end, duration)])
                    if moment[0] < moment[1]:
                        iou2d = moment_to_iou2d(moment, num_clips, duration)
                        iou2ds.append(iou2d)
                        moments.append(moment)
                        sents.append(sent)
                if len(moments) == 0:
                    continue
                moments = torch.stack(moments, dim=0)
                iou2ds = torch.stack(iou2ds)
                iou2d = moments_to_iou2d(moments, num_clips, duration)
                num_targets = torch.tensor(len(moments))

                assert len(moments) != 0
                assert len(anno['video']) == 1
                annos.append({
                    'vids': anno['video'],
                    'timestamps': [[0, duration]],
                    'query': anno['query'],                 # query string
                    'sents': sents,                         # original sentences
                    'iou2d': iou2d,                         # 2d iou map
                    'iou2ds': iou2ds,                       # list of 2d iou map
                    'num_targets': num_targets,             # number of target moments
                    'moments': moments,                     # clips moments in seconds
                    'durations': torch.tensor([duration]),  # video length in seconds
                    'duration': torch.tensor(duration),     # video length in seconds
                })
        return annos
