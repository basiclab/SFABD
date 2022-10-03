import json
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer

from mmn.utils import moment_to_iou2d, moments_to_iou2d, multi_vgg_feats


class MultiTargetCharadesDataset(torch.utils.data.Dataset):
    """
    Accepted format:
    [
        "14128": {
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
            "query": "a person a person is running"
        },
        ...
    ]
    """
    def __init__(
        self,
        ann_file,
        vgg_feat_file,      # path to feature file
        c3d_feat_folder,
        num_init_clips,     # number of small basic clips features. 32 for Charades (now 64)
        num_clips,
        feat_type,          # `vgg` or `c3d` case insensitive
    ):
        self.vgg_feat_file = vgg_feat_file
        self.c3d_feat_folder = c3d_feat_folder
        self.num_init_clips = num_init_clips
        self.feat_type = feat_type
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")

        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        for _, anno in tqdm(annos.items(), ncols=0, leave=False):
            duration = anno['duration']            # video length
            moments = []                           # start and end time of each annotation in a video
            iou2ds = []
            sents = []                             # all query from multiple samples
            for (start, end), sentence in zip(anno['timestamps'], anno['sentences']):
                moment = [max(start, 0), min(end, duration)]
                if moment[0] < moment[1]:
                    iou2d = moment_to_iou2d(
                        torch.Tensor(moment), num_clips, duration)
                    iou2ds.append(iou2d)
                    moments.append(moment)
                    sents.append(sentence)
            iou2ds = torch.stack(iou2ds)
            iou2d = moments_to_iou2d(
                torch.tensor(moments), num_clips, duration)
            num_targets = torch.tensor(len(moments))

            self.annos.append({
                'vid': anno['video'],
                'seq_index': anno['seq_index'],
                'query': anno['query'],             # query string
                'sents': sents,                     # original sentences
                'iou2d': iou2d,                     # 2d iou map
                'iou2ds': iou2ds,                   # list of 2d iou map
                'num_targets': num_targets,         # number of target moments
                'moments': moments,                 # clips moments in seconds
                'duration': duration,               # video length in seconds
            })

    def __getitem__(self, idx):
        if self.feat_type.lower() == "vgg":
            video_feats = multi_vgg_feats(
                self.vgg_feat_file,
                self.annos[idx]['vid'],
                self.annos[idx]['seq_index'],
                self.num_init_clips)                # [NUM_INIT_CLIPS, 4096]
        else:
            raise ValueError("Invalid feature type: {}".format(self.feat_type))

        return (
            video_feats,                            # pooled frame features
            self.annos[idx]['query'],               # query string
            self.annos[idx]['sents'],               # original sentences
            self.annos[idx]['iou2d'],               # combined iou2d
            self.annos[idx]['iou2ds'],              # iou2d for each moment
            self.annos[idx]['num_targets'],         # num_targets
            self.annos[idx]['moments'],             # moments in seconds
            self.annos[idx]['duration'],            # duration
            self.annos[idx]['vid'],                 # video ids
            idx,                                    # index
        )

    def collate_fn(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, List]]:
        (
            video_feats,
            query_list,
            sents_list,
            iou2d_list,
            iou2ds_list,
            num_targets_list,
            moments_list,
            duration_list,
            vid_list,
            idx_list,
        ) = list(zip(*batch))       # list of batch to batch of list

        num_targets = torch.stack(num_targets_list, dim=0)
        B = len(num_targets_list)
        scatter_idx = torch.arange(B).repeat_interleave(num_targets, dim=0)

        sents_all = sum(sents_list, [])
        assert len(sents_all) == len(scatter_idx)

        query = self.tokenizer(
            query_list, padding=True, return_tensors="pt", return_length=True)
        sents = self.tokenizer(
            sents_all, padding=True, return_tensors="pt", return_length=True)

        return {    # first dictionary must contain only tensors
            'video_feats': torch.stack(video_feats, dim=0),
            'query_tokens': query['input_ids'],
            'query_length': query['length'],
            'iou2d': torch.stack(iou2d_list, dim=0),
            'iou2ds': torch.cat(iou2ds_list, dim=0),
            'sents_tokens': sents['input_ids'],
            'sents_length': sents['length'],
            'num_targets': num_targets,
            'scatter_idx': scatter_idx,
        }, {        # second dictionary must not contain tensors
            'query': query_list,
            'sents': sents_list,
            'moments': moments_list,
            'duration': duration_list,
            'vid': vid_list,
            'idx': idx_list,
        }

    def __len__(self):
        return len(self.annos)


class CharadesDataset(MultiTargetCharadesDataset):
    """
    Accpeted format:
    [
        "3MSZA": {
            "duration": 30.96,
            "timestamps": [
                [24.3, 30.4],
                [24.3, 30.4],
            ],
            "sentences": [
                "person turn a light on.",
                "person flipped the light switch near the door.",
            ]
        },
        ...
    ]
    """
    def __init__(
        self,
        ann_file,
        vgg_feat_file,      # path to feature file
        c3d_feat_folder,
        num_init_clips,     # number of small basic clips features. 32 for Charades (now 64)
        num_clips,
        feat_type,          # `vgg` or `c3d` case insensitive
    ):
        self.vgg_feat_file = vgg_feat_file
        self.c3d_feat_folder = c3d_feat_folder
        self.num_init_clips = num_init_clips
        self.feat_type = feat_type
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")

        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        for vid, anno in tqdm(annos.items(), ncols=0, leave=False):
            duration = anno['duration']            # video length
            for sent, moment in zip(anno['sentences'], anno['timestamps']):
                iou2d = moment_to_iou2d(
                    torch.Tensor(moment), num_clips, duration)
                self.annos.append({
                    'vid': [vid],
                    'seq_index': [[0, None]],           # include all frames
                    'query': sent,                      # query string
                    'sents': [sent],                    # original sentences
                    'iou2d': iou2d,                     # 2d iou map
                    'iou2ds': iou2d.unsqueeze(0),       # list of 2d iou map
                    'num_targets': torch.tensor(1),     # number of target moments
                    'moments': [moment],                # clips moments in seconds
                    'duration': duration,               # video length in seconds
                })


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    def test(dataset):
        for i in torch.randint(0, len(dataset), (2, )):
            for data in dataset[i]:
                if hasattr(data, 'shape'):
                    print(f"type: {type(data)}, shape: {data.shape}")
                else:
                    print(f"type: {type(data)}")
            print('-' * 80)

        batch_size = 16
        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
        for batch, info in loader:
            name_length = max(
                [len(k) for k in batch.keys()] + [len(k) for k in info.keys()])
            print("batch:")
            for k, v in batch.items():
                print(f"{k:<{name_length}s}: {v.shape}")
            print('-' * 80)
            print("info:")
            for k, v in info.items():
                print(f"{k:<{name_length}s}: {len(v)}")
                assert len(v) == batch_size
            break

    dataset = MultiTargetCharadesDataset(
        ann_file='datadir/Charades_STA/combined_charades_train_remove_repeat_action_videos.json',
        vgg_feat_file="./datadir/Charades_STA/vgg_rgb_features.hdf5",
        c3d_feat_folder=None,
        num_init_clips=64,
        num_clips=32,
        feat_type='vgg',
    )
    test(dataset)

    dataset = CharadesDataset(
        ann_file='datadir/Charades_STA/charades_train.json',
        vgg_feat_file="./datadir/Charades_STA/vgg_rgb_features.hdf5",
        c3d_feat_folder=None,
        num_init_clips=64,
        num_clips=32,
        feat_type='vgg',
    )
    test(dataset)
