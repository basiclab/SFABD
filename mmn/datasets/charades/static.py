import json

import torch
from tqdm import tqdm

from mmn.datasets.base import CollateBase
from mmn.utils import moment_to_iou2d, moments_to_iou2d, multi_vgg_feats


class CharadesBase(CollateBase):
    def __init__(
        self,
        vgg_feat_file,      # path to feature file
        num_init_clips,     # number of small basic clips features. 32 for Charades (now 64)
    ):
        super().__init__()
        self.vgg_feat_file = vgg_feat_file
        self.num_init_clips = num_init_clips

    def get_anno(self, idx):
        """Get annotation for a given index.
        Returns:
            anno: {
                'vid': List[str],
                'seq_index': List[List[int]],
                'query': str,
                'sents': List[str],
                'iou2d': torch.Tensor,
                'iou2ds': List[torch.Tensor],
                'num_targets': torch.Tensor,
                'moments': List[torch.Tensor],
                'duration': torch.Tensor,
            }
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        anno = self.get_anno(idx)
        video_feats = multi_vgg_feats(
            self.vgg_feat_file,
            anno['vid'],
            anno['seq_index'],
            self.num_init_clips)         # [NUM_INIT_CLIPS, 4096]

        return (
            video_feats,                 # pooled frame features
            anno['query'],               # query string
            anno['sents'],               # original sentences
            anno['iou2d'],               # combined iou2d
            anno['iou2ds'],              # iou2d for each moment
            anno['num_targets'],         # num_targets
            anno['moments'],             # moments in seconds
            anno['duration'],            # duration
            anno['vid'],                 # video ids
            idx,                         # index
        )

    def __len__(self):
        return len(self.annos)


class StaticMultiTargetCharades(CharadesBase):
    """Statically sampled multi-target Charades dataset.
    Accepted format:
    [
        {
            "video": [
                "VZY0C",
                "QDZ38"
            ],
            "timestamps":[
                [0.6, 7.6],
                [18.0, 24.8]
            ],
            "seq_index": [
                [8, 115],
                [13, 143]
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
        vgg_feat_file,      # path to feature file
        num_init_clips,     # number of initial clips. e.g., 64 for Charades
        **dummy,
    ):
        super().__init__(vgg_feat_file, num_init_clips)
        self.annos = self.parse_anno(ann_file, num_clips)

    def parse_anno(self, ann_file, num_clips):
        with open(ann_file, 'r') as f:
            raw_annos = json.load(f)

        annos = []
        for anno in tqdm(raw_annos, ncols=0, leave=False):
            duration = anno['duration']     # video length
            moments = []                    # start and end time of each moment
            iou2ds = []                     # iou2d for each mement
            sents = []                      # sentence for each moment
            for (start, end), sentence in zip(anno['timestamps'], anno['sentences']):
                moment = torch.Tensor([max(start, 0), min(end, duration)])
                if moment[0] < moment[1]:
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    iou2ds.append(iou2d)
                    moments.append(moment)
                    sents.append(sentence)
            moments = torch.stack(moments, dim=0)
            iou2ds = torch.stack(iou2ds)
            iou2d = moments_to_iou2d(moments, num_clips, duration)
            num_targets = torch.tensor(len(moments))

            assert len(moments) != 0
            annos.append({
                'vid': anno['video'],
                'seq_index': anno['seq_index'],
                'query': anno['query'],                 # query string
                'sents': sents,                         # original sentences
                'iou2d': iou2d,                         # 2d iou map
                'iou2ds': iou2ds,                       # list of 2d iou map
                'num_targets': num_targets,             # number of target moments
                'moments': moments,                     # clips moments in seconds
                'duration': torch.tensor(duration),     # video length in seconds
            })
        return annos

    def get_anno(self, idx):
        return self.annos[idx]
