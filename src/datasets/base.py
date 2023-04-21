import json
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer

from src import dist


class CollateBase(torch.utils.data.Dataset):
    def __init__(self, ann_file):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.annos = self.parse_anno(ann_file)

    def get_feat(self, anno):
        """Get video features for a single video"""
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
        for vid, video_data in pbar:
            duration = torch.tensor(video_data['duration'])
            sentences = []
            tgt_moments = []
            num_targets = []
            qids = []
            for anno in video_data['annotations']:
                timestamps = []
                for timestamp in anno['timestamps']:
                    timestamp = torch.Tensor(timestamp)
                    timestamp = torch.clamp(timestamp / duration, 0, 1)

                    if timestamp[0] < timestamp[1]:
                        timestamps.append(timestamp)
                    else:
                        assert False, f"Invalid timestamp: {timestamp}"

                if len(timestamps) != 0:
                    sentences.append(anno['query'])
                    tgt_moments.append(torch.stack(timestamps, dim=0))
                    num_targets.append(torch.tensor(len(timestamps)))

                    if 'qid' in anno:
                        qids.append(anno['qid'])
                    else:
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

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        video_feats = self.get_feat(anno)

        # Do video feature-level augmentation here

        return {
            'idx': torch.ones(anno['num_sentences'], dtype=torch.long) * idx,
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

        sentences = self.tokenizer(
            sum(batch['sentences'], []),    # List of List of str -> List of str
            padding=True,
            return_tensors="pt")

        video_lens = torch.tensor([x.shape[0] for x in batch['video_feats']])
        pad_len = video_lens.max()
        for i, video_feats in enumerate(batch['video_feats']):
            batch['video_feats'][i] = F.pad(
                video_feats, [0, 0, 0, pad_len - len(video_feats)])
        video_masks = torch.arange(pad_len)[None, :] < video_lens[:, None]

        return {
            'video_feats': torch.stack(batch['video_feats'], dim=0),
            'video_masks': video_masks,
            'sents_tokens': sentences['input_ids'],
            'sents_masks': sentences['attention_mask'],
            'num_sentences': torch.stack(batch['num_sentences'], dim=0),
            'num_targets': torch.cat(batch['num_targets'], dim=0),
            'tgt_moments': torch.cat(batch['tgt_moments'], dim=0),
        }, {
            'qids': batch['qids'],
            'sentences': batch['sentences'],
            'vid': batch['vid'],
            'idx': torch.cat(batch['idx']),
            'duration': batch['duration'],
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from src.datasets.charades import (
        CharadesMultitargetTrain, CharadesMultitargetTest,
        CharadesSTATrain, CharadesSTATest)
    from src.datasets.qvhighlights import QVHighlightsTrain, QVHighlightsVal
    from src.datasets.activitynet import (
        ActivityNetTrain, ActivityNetVal, ActivityNetTest)

    def test(dataset: CollateBase):
        def show_dict(data):
            name_length = max(len(k) for k in data.keys())
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k:<{name_length}s}: type: {type(v)}, shape: {v.shape}")
                elif isinstance(v, list):
                    print(f"{k:<{name_length}s}: type: {type(v)}, len: {len(v)}")
                else:
                    print(f"{k:<{name_length}s}: type: {type(v)}")
            print()

        print(f"dataset length: {len(dataset)}")
        print()

        print('Single data:')
        data = dataset[torch.randint(len(dataset), ())]
        show_dict(data)

        batch_size = 16
        batch, info = next(iter(DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )))
        print("batch:")
        show_dict(batch)

        print("info:")
        show_dict(info)
        print('-' * 80)

    print("Charades-STA train")
    dataset = CharadesSTATrain()
    test(dataset)

    print("Charades-STA test")
    dataset = CharadesSTATest()
    test(dataset)

    print("Charades-STA Multitarget train")
    dataset = CharadesMultitargetTrain()
    test(dataset)

    print("Charades-STA Multitarget test")
    dataset = CharadesMultitargetTest()
    test(dataset)

    print("QVHighlight train")
    dataset = QVHighlightsTrain()
    test(dataset)

    print("QVHighlight val")
    dataset = QVHighlightsVal()
    test(dataset)

    print("ActivityNet train")
    dataset = ActivityNetTrain()
    test(dataset)

    print("ActivityNet val")
    dataset = ActivityNetVal()
    test(dataset)

    print("ActivityNet test")
    dataset = ActivityNetTest()
    test(dataset)
