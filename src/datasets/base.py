import json
import random
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

    # def augmentation(self, anno, video_feats):
    #     """Do multi positive augmentation"""
    #     raise NotImplementedError

    # def augmentation(self, anno, video_feats):
    #     """Do multi positive augmentation"""
    #     duration = anno['duration']
    #     # target moments of whole query-moment pairs in same video
    #     moments = anno['tgt_moments']
    #     '''
    #     anno: {
    #         'vid': 'FLDHS',
    #         'sentences': ['person are putting stuff in a box.', 'person puts it in a box.'],
    #         'num_sentences': tensor(2),
    #         'num_targets': tensor([1, 1]),
    #         'tgt_moments': tensor([[0.1850, 0.4385],
    #                                [0.1850, 0.4385]]),
    #         'duration': tensor(29.1900),
    #         'qids': tensor([0, 0])
    #     }
    #     '''
    #     start_time = 0
    #     empty_clips = []
    #     # # sort moments by the starting time
    #     # sorted_moments = sorted(moments, key=lambda x: x[0])
    #     # # moments_sorted_idx = sorted(range(len(moments)))
    #     # sorted_moments_idx = sorted(range(len(moments)), key=lambda k: moments[k])
    #     # # find all empty clips in video
    #     # for moment in sorted_moments:
    #     #     if (moment[0] - start_time) > 0:
    #     #         empty_clips.append([start_time, moment[0]])
    #     #     start_time = moment[1]
    #     # if (duration - start_time) > 0:
    #     #     empty_clips.append([start_time, duration])
    #     # empty_clips_len = [clip[1] - clip[0] for clip in empty_clips]

    #     # # random choose one moment to do augmentation
    #     # moment_idx = random.choice(range(len(sorted_moments)))
    #     # moment = sorted_moments[moment_idx]
    #     # moment_len = moment[1] - moment[0]
    #     # # enlarge moment len to include some background on boundary
    #     # moment_len = moment_len * 1.2
    #     # possible_start_index = []
    #     # for empty_clip, empty_clip_len in zip(empty_clips, empty_clips_len):
    #     #     if moment_len < empty_clip_len:
    #     #         for start_time in np.arange(empty_clip[0], empty_clip[1] - moment_len, 0.5):
    #     #             possible_start_index.append(start_time)

    #     # # sample from the all possible start index to do augmentation
    #     # aug_start = random.choice(possible_start_index)
    #     # aug_end = aug_start + moment_len

    #     # # update anno
    #     # original_moment_idx = sorted_moments_idx[moment_idx]
    #     # anno['tgt_moments'] = torch.cat(anno['tgt_moments'], [aug_start, aug_end])
    #     # anno['num_targets'] += 1

    #     # # do mixup
    #     # alpha = 0.9
    #     # seq_len = video_feats.shape[0]
    #     # target_seq_start_idx = int(seq_len * moment[0])
    #     # target_seq_end_idx = int(seq_len * moment[1])
    #     # target_feat = video_feats[target_seq_start_idx:target_seq_end_idx]

    #     # aug_seq_start_idx = int(seq_len * aug_start)
    #     # aug_seq_end_idx = int(seq_len * aug_end)
    #     # assert (target_seq_end_idx - target_seq_start_idx) == \
    #     #     (aug_seq_end_idx - aug_seq_start_idx)
    #     # video_feats[aug_seq_start_idx: aug_seq_end_idx] = \
    #     #     target_feat * alpha + \
    #     #     video_feats[aug_seq_start_idx: aug_seq_end_idx] * (1 - alpha)

    #     return anno, video_feats

    def __len__(self):
        return len(self.annos)

    # def __getitem__(self, idx):
    #     anno = self.annos[idx]
    #     video_feats = self.get_feat(anno)   # [seq_len, dim]

    #     return {
    #         'idx': torch.ones(anno['num_sentences'], dtype=torch.long) * idx,
    #         'video_feats': video_feats,
    #         **anno,
    #     }

    def __getitem__(self, idx):
        '''
        anno: {
            'vid': 'FLDHS',
            'sentences': ['person are putting stuff in a box.', 'person puts it in a box.'],
            'num_sentences': tensor(2),
            'num_targets': tensor([1, 1]),
            'tgt_moments': tensor([[0.1850, 0.4385],
                                    [0.1850, 0.4385]]),
            'duration': tensor(29.1900),
            'qids': tensor([0, 0])
        }
        '''
        anno = self.annos[idx]
        video_feats = self.get_feat(anno)   # [seq_len, dim]
        # duplicate video feats for each query, anno['num_sentences'] times
        video_feats = video_feats.unsqueeze(0).repeat(anno['num_sentences'], 1, 1)

        # 50% do video feature-level augmentation
        # if random.random() > 0.5:
        #     anno, video_feats = self.augmentation(anno, video_feats)
        # anno, video_feats = self.augmentation(anno, video_feats)

        return {
            'idx': torch.ones(anno['num_sentences'], dtype=torch.long) * idx,
            'video_feats': video_feats,
            **anno,
        }

    # def collate_fn(
    #     self,
    #     batch
    # ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List]]:
    #     batch = {
    #         key: [x[key] for x in batch] for key in batch[0].keys()
    #     }

    #     sentences = self.tokenizer(
    #         sum(batch['sentences'], []),    # List of List of str -> List of str
    #         padding=True,
    #         return_tensors="pt")

    #     video_lens = torch.tensor([x.shape[0] for x in batch['video_feats']])
    #     pad_len = video_lens.max()
    #     for i, video_feats in enumerate(batch['video_feats']):
    #         # video_feats: [num_sent, seq_len, feat_dim]
    #         batch['video_feats'][i] = F.pad(
    #             video_feats, [0, 0, 0, pad_len - len(video_feats)])
    #     video_masks = torch.arange(pad_len)[None, :] < video_lens[:, None]

    #     # return batch, info
    #     return {
    #         'video_feats': torch.stack(batch['video_feats'], dim=0),      # [num_sents, max_seq_len, feat_dim]
    #         'video_masks': video_masks,                                   # [num_sents, max_seq_len]
    #         'sents_tokens': sentences['input_ids'],                       # [num_sents, max_sent_len]
    #         'sents_masks': sentences['attention_mask'],                   # [num_sents, max_sent_len]
    #         'num_sentences': torch.stack(batch['num_sentences'], dim=0),  # [bs] sum = num_sents
    #         'num_targets': torch.cat(batch['num_targets'], dim=0),        # [num_targets]
    #         'tgt_moments': torch.cat(batch['tgt_moments'], dim=0),        # [num_targets, 2]
    #     }, {
    #         'qids': batch['qids'],
    #         'sentences': batch['sentences'],
    #         'vid': batch['vid'],
    #         'idx': torch.cat(batch['idx']),
    #         'duration': batch['duration'],
    #     }

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

        video_lens = []
        for video_feat_per_sample in batch['video_feats']:
            for video_feat_per_query in video_feat_per_sample:
                video_lens.append(torch.tensor([video_feat_per_query.shape[0]]))
        video_lens = torch.tensor(video_lens)           # [num_sents]
        pad_len = video_lens.max()
        for i, video_feats in enumerate(batch['video_feats']):
            # video_feats: [num_sent, seq_len, feat_dim]
            batch['video_feats'][i] = F.pad(
                video_feats,
                [
                    0, 0,                               # for dim = -1
                    0, pad_len - video_feats.shape[1]   # for dim = -2
                ]
            )
        # for i, video_feat in enumerate(batch['video_feats']):
        #     print(f"{i}, :{video_feat.shape}")
        video_masks = torch.arange(pad_len)[None, :] < video_lens[:, None]

        # return batch, info
        return {
            'video_feats': torch.cat(batch['video_feats'], dim=0),        # [num_sents, max_seq_len, feat_dim]
            'video_masks': video_masks,                                   # [num_sents, max_seq_len]
            'sents_tokens': sentences['input_ids'],                       # [num_sents, max_sent_len]
            'sents_masks': sentences['attention_mask'],                   # [num_sents, max_sent_len]
            'num_sentences': torch.stack(batch['num_sentences'], dim=0),  # [bs] sum = num_sents
            'num_targets': torch.cat(batch['num_targets'], dim=0),        # [num_targets]
            'tgt_moments': torch.cat(batch['tgt_moments'], dim=0),        # [num_targets, 2]
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
