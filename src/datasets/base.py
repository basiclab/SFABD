from math import ceil, floor
from typing import List, Dict, Tuple

import h5py
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer


class CollateBase(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")

    def get_anno(self, idx):
        """Get annotation for a given index.
        Returns:
            anno: {
                'vid': List[str],
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

    def get_feat(self, anno):
        raise NotImplementedError

    def __getitem__(self, idx):
        anno = self.get_anno(idx)
        video_feats = self.get_feat(anno)           # [NUM_INIT_CLIPS, 4096]

        return {
            'idx': idx,                             # index
            'video_feats': video_feats,             # pooled frame features
            **anno,
        }

    def collate_fn(
        self,
        batch
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List]]:
        batch = {
            key: [x[key] for x in batch] for key in batch[0].keys()
        }

        num_targets = torch.stack(batch['num_targets'], dim=0)

        query = self.tokenizer(
            batch['query'],
            padding=True,
            return_tensors="pt",
            return_length=True)
        sents = self.tokenizer(
            sum(batch['sents'], []),    # List of List of str -> List of str
            padding=True,
            return_tensors="pt",
            return_length=True)

        return (
            # must contain only tensors
            {
                'video_feats': torch.stack(batch['video_feats'], dim=0),
                'query_tokens': query['input_ids'],
                'query_length': query['length'],
                'iou2d': torch.stack(batch['iou2d'], dim=0),
                'iou2ds': torch.cat(batch['iou2ds'], dim=0),
                'sents_tokens': sents['input_ids'],
                'sents_length': sents['length'],
                'num_targets': num_targets,
            },
            # for evaluation
            {
                'query': batch['query'],                            # List[str]
                'sents': batch['sents'],                            # List[List[str]]
                'moments': batch['moments'],                        # List[torch.Tensor]
                'duration': torch.stack(batch['duration'], dim=0),  # torch.Tensor
                'vid': batch['vids'],                               # List[str]
                'idx': torch.tensor(batch['idx']),                  # torch.Tensor
            }
        )


def aggregate_feats(
    feats: torch.Tensor,                        # [NUM_SRC_CLIPS, C]
    num_tgt_clips: int,                         # number of target clip
    op_type: str = 'avg',                       # 'avg' or 'max'
) -> torch.Tensor:
    """Produce the feature of per video into fixed shape by averaging.

    Returns:
        avgfeats: [C, num_tgt_clips]
    """
    assert op_type in ['avg', 'max']

    num_src_clips, _ = feats.shape
    idxs = torch.arange(0, num_tgt_clips + 1) / num_tgt_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips - 1)
    feats = F.normalize(feats, dim=1)
    feats_bucket = []
    for i in range(num_tgt_clips):
        s, e = idxs[i], idxs[i + 1]
        # To prevent an empty selection, check the indices are valid.
        if s < e:
            if op_type == 'avg':
                feats_bucket.append(feats[s:e].mean(dim=0))
            if op_type == 'max':
                feats_bucket.append(feats[s:e].max(dim=0)[0])
        else:
            feats_bucket.append(feats[s])
    return torch.stack(feats_bucket, dim=1)                     # channel first


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from src.datasets.charades.base import Charades
    from src.datasets.charades.dynamic import DynamicCharades
    from src.datasets.charades.static import StaticCharades
    from src.datasets.activitynet.base import ActivityNet

    def test(dataset: CollateBase):
        print(f"dataset length: {len(dataset)}")
        print()

        # data_dict = dataset[torch.randint(len(dataset), ())]
        data_dict = dataset[-1]
        name_length = max(len(k) for k in data_dict.keys())
        print('Single data:')
        for key, value in data_dict.items():
            if hasattr(value, 'shape'):
                print(f"name: {key:<{name_length}s}, type: {type(value)}, shape: {value.shape}")
            else:
                print(f"name: {key:<{name_length}s}, type: {type(value)}")
        print()

        batch_size = 16
        batch, info = next(iter(DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )))
        name_length = max(
            [len(k) for k in batch.keys()] + [len(k) for k in info.keys()])
        print("batch:")
        for k, v in batch.items():
            print(f"{k:<{name_length}s}: {v.shape}")
        print()
        print("info:")
        for k, v in info.items():
            print(f"{k:<{name_length}s}: {len(v)}")
        print('-' * 80)

    print("CharadesSTA train")
    dataset = Charades(
        ann_file='data/CharadesSTA/original/train.json',
        num_clips=32,
        feat_file="./data/CharadesSTA/vgg_rgb_features.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("CharadesSTA test")
    dataset = Charades(
        ann_file='data/CharadesSTA/original/test.json',
        num_clips=32,
        feat_file="./data/CharadesSTA/vgg_rgb_features.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("Dynamic Multi-target train")
    dataset = DynamicCharades(
        ann_file='data/CharadesSTA/new/query_template_group_train.json',
        num_clips=32,
        feat_file="./data/CharadesSTA/vgg_rgb_features_all.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("Real Multi-target test")
    dataset = Charades(
        ann_file='data/CharadesSTA/new/00_percent/test.json',
        num_clips=32,
        feat_file="./data/CharadesSTA/vgg_rgb_features_all.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("Real Single-target train")
    dataset = Charades(
        ann_file='data/CharadesSTA/new/00_percent/train.json',
        num_clips=32,
        feat_file="./data/CharadesSTA/vgg_rgb_features_all.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("Static Multi-target train")
    dataset = StaticCharades(
        ann_file='data/CharadesSTA/new/query_template_group_train.json',
        num_clips=32,
        feat_file="./data/CharadesSTA/vgg_rgb_features_all.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("ActivityNet train")
    dataset = ActivityNet(
        ann_file='./data/ActivityNet/train.json',
        num_clips=32,
        feat_file="./data/ActivityNet/sub_activitynet_v1-3_c3d.hdf5",
        num_init_clips=64,
    )
    test(dataset)
