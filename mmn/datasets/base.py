from typing import List, Dict, Tuple

import torch
from transformers import DistilBertTokenizer


class CollateBase(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")

    def collate_fn(
        self,
        batch
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List]]:
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
        sents_all = sum(sents_list, [])

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
        }, {        # for evaluation
            'query': query_list,                            # List[str]
            'sents': sents_list,                            # List[List[str]]
            'moments': moments_list,                        # List[torch.Tensor]
            'duration': torch.stack(duration_list, dim=0),  # torch.Tensor
            'vid': vid_list,                                # List[str]
            'idx': torch.tensor(idx_list),                  # torch.Tensor
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from mmn.datasets.charades.static import StaticMultiTargetCharades
    from mmn.datasets.charades.dynamic import DynamicMultiTargetCharades

    def test(dataset: CollateBase):
        print(f"dataset length: {len(dataset)}")
        for i in torch.randint(0, len(dataset), (2, )):
            print(f"random index: {i}")
            for data in dataset[i]:
                if hasattr(data, 'shape'):
                    print(f"type: {type(data)}, shape: {data.shape}")
                else:
                    print(f"type: {type(data)}")
            print()

        batch_size = 16
        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
        batch, info = next(iter(loader))
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
    dataset = StaticMultiTargetCharades(
        ann_file='data/Charades_STA/train.json',
        num_clips=32,
        vgg_feat_file="./data/Charades_STA/vgg_rgb_features.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("CharadesSTA test")
    dataset = StaticMultiTargetCharades(
        ann_file='data/Charades_STA/test.json',
        num_clips=32,
        vgg_feat_file="./data/Charades_STA/vgg_rgb_features.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("Dynamic Multi-target train")
    dataset = DynamicMultiTargetCharades(
        ann_file='data/Charades_STA/v2/00_percent/train.json',
        template_file='data/Charades_STA/v2/query_template_group_train.json',
        num_clips=32,
        vgg_feat_file="./data/Charades_STA/vgg_rgb_features_all.hdf5",
        num_init_clips=64,
    )
    test(dataset)

    print("Real Multi-target test")
    dataset = StaticMultiTargetCharades(
        ann_file='data/Charades_STA/v2/00_percent/test.json',
        num_clips=32,
        vgg_feat_file="./data/Charades_STA/vgg_rgb_features_all.hdf5",
        num_init_clips=64,
    )
    test(dataset)
