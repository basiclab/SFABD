import json
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer

import src.dist as dist
from src.utils import moment_to_iou2d


class CollateBase(torch.utils.data.Dataset):
    def __init__(
        self,
        # annotation related
        ann_file,           # path to statically generated multi-target annotation file
        num_clips,          # number of final clips, e.g., 32 for Charades
        **dummy,
    ):
        self.num_clips = num_clips
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.annos = self.parse_anno(ann_file)

    def get_feat(self, anno):
        raise NotImplementedError

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
        for vid, anno in pbar:
            sentences = anno['sentences']
            duration = torch.tensor(anno['duration'])
            timestamps = torch.Tensor(anno['timestamps'])
            timestamps = (timestamps / duration).clamp(min=0, max=1)
            moments = []                        # start and end time of each moment
            sents = []                          # sentence for each moment
            for moment, sent in zip(timestamps, sentences):
                if moment[0] < moment[1]:
                    moments.append(moment)
                    sents.append(sent)

            if len(moments) != 0:
                moments = torch.stack(moments, dim=0)
                num_targets = torch.tensor(len(moments))
                annos.append({
                    'vid': vid,
                    'sents': sents,
                    'num_targets': num_targets,
                    'moments': moments,
                    'duration': duration,
                })
        pbar.close()

        return annos

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        video_feats = self.get_feat(anno)                   # [NUM_INIT_CLIPS, 4096]
        idx = torch.ones(anno['num_targets'].item()) * idx  # [NUM_TARGETS]
        return {
            'idxs': idx,
            'video_feats': video_feats,
            **anno,
        }

    def collate_fn(
        self,
        batch
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List]]:
        batch = {
            key: [x[key] for x in batch] for key in batch[0].keys()
        }

        sents = self.tokenizer(
            sum(batch['sents'], []),    # List of List of str -> List of str
            padding=True,
            return_tensors="pt")

        moments = torch.cat(batch['moments'], dim=0)
        iou2ds = moment_to_iou2d(moments, self.num_clips)

        return {
            'video_feats': torch.stack(batch['video_feats'], dim=0),
            'sents_tokens': sents['input_ids'],
            'sents_masks': sents['attention_mask'],
            'iou2ds': iou2ds,
            'num_targets': torch.stack(batch['num_targets'], dim=0),
            'moments': moments,
            'idxs': torch.cat(batch['idxs'], dim=0),
        }


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

    from src.datasets.charades import Charades
    from src.datasets.activitynet import ActivityNet
    from src.datasets.tacos import TACoS

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
        batch = next(iter(DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )))
        name_length = max(len(k) for k in batch.keys())
        print("batch:")
        for k, v in batch.items():
            print(f"{k:<{name_length}s}: {v.shape}")
        print('-' * 80)

    print("Charades-STA train")
    dataset = Charades(
        ann_file='data/CharadesSTA/train.json',
        num_clips=16,
        feat_file="./data/CharadesSTA/vgg_rgb_features.hdf5",
        num_init_clips=32,
    )
    test(dataset)

    print("Charades-STA test")
    dataset = Charades(
        ann_file='data/CharadesSTA/test.json',
        num_clips=16,
        feat_file="./data/CharadesSTA/vgg_rgb_features.hdf5",
        num_init_clips=32,
    )
    test(dataset)

    print("ActivityNet Caption train")
    dataset = ActivityNet(
        ann_file='data/ActivityNet/train.json',
        num_clips=64,
        feat_file="./data/ActivityNet/sub_activitynet_v1-3_c3d.hdf5",
        num_init_clips=256,
    )
    test(dataset)

    print("ActivityNet Caption validation")
    dataset = ActivityNet(
        ann_file='data/ActivityNet/val.json',
        num_clips=64,
        feat_file="./data/ActivityNet/sub_activitynet_v1-3_c3d.hdf5",
        num_init_clips=256,
    )
    test(dataset)

    print("ActivityNet Caption test")
    dataset = ActivityNet(
        ann_file='data/ActivityNet/test.json',
        num_clips=64,
        feat_file="./data/ActivityNet/sub_activitynet_v1-3_c3d.hdf5",
        num_init_clips=256,
    )
    test(dataset)

    print("TACoS train")
    dataset = TACoS(
        ann_file='data/TACoS/train.json',
        num_clips=128,
        feat_file="./data/TACoS/tall_c3d_features.hdf5",
        num_init_clips=256,
    )
    test(dataset)

    print("TACoS validation")
    dataset = TACoS(
        ann_file='data/TACoS/val.json',
        num_clips=128,
        feat_file="./data/TACoS/tall_c3d_features.hdf5",
        num_init_clips=256,
    )
    test(dataset)

    print("TACoS test")
    dataset = TACoS(
        ann_file='data/TACoS/test.json',
        num_clips=128,
        feat_file="./data/TACoS/tall_c3d_features.hdf5",
        num_init_clips=256,
    )
    test(dataset)
