import json
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer

from src import dist


class CollateBase(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        do_augmentation=False,
        mixup_alpha=0.9,
        downsampling_method='odd',
        aug_prob=0.75,
        downsampling_prob=0.5,
    ):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.annos = self.parse_anno(ann_file)
        self.do_augmentation = do_augmentation
        self.mixup_alpha = mixup_alpha
        self.downsampling_method = downsampling_method
        self.aug_prob = aug_prob
        self.downsampling_prob = downsampling_prob

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
            # Sample at most k samples to prevent large VRAM usage caused by augmentation
            k = 7
            if len(video_data['annotations']) > k:
                sampled_annos = random.sample(video_data['annotations'], k)
            else:
                sampled_annos = video_data['annotations']

            for anno in sampled_annos:
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

    def downsample(
        self,
        video_feats: torch.tensor,          # [seq_len, feat_dim]
        method: str,
    ) -> torch.tensor:
        if method == 'odd':
            video_feats = video_feats[::2]                          # [ceil(seq_len / 2), feat_dim]

        elif method == 'avg_pooling':
            video_feats = F.avg_pool1d(video_feats.t(), 2, 2, ceil_mode=True).t()   # [ceil(seq_len / 2), feat_dim]

        elif method == 'None':
            pass

        else:
            raise ValueError(f"Unknown downsampling method {method}")

        return video_feats

    def augmentation(self, anno, video_feats):
        """Do multi positive augmentation"""
        '''
        anno: {
            'vid': 'FLDHS',
            'sentences': ['person are putting stuff in a box.', 'person puts it in a box.'],
            'num_sentences': tensor(2),
            'num_targets': tensor([1, 1]),
            'tgt_moments': tensor([[0.1850, 0.4385],
                                   [0.1850, 0.4385]]),
            'duration': tensor(29.1900),
            'qids': tensor([0, 0]) # Placeholder for QVHighlights test server evaluation
        }
        video_feats: [num_sent, seq_len, feat_dim]
        '''
        shift_t = 0
        new_num_targets = []
        new_tgt_moments = []
        for idx, num_target in enumerate(anno['num_targets']):
            if random.random() > self.aug_prob:
                # Don't do augmentation
                new_num_targets.append(num_target)
                new_tgt_moments.append(anno['tgt_moments'][shift_t:shift_t + num_target])
                shift_t += num_target
                continue

            # Find all empty clips in video for each query
            tgt_moments = anno['tgt_moments'][shift_t:shift_t + num_target]     # [num_target, 2]
            # sort moments by the starting time
            sorted_moments = sorted(tgt_moments, key=lambda x: x[0])
            empty_clips = []
            start_time = torch.tensor(0.0)
            for moment in sorted_moments:
                if (moment[0] - start_time) > 0:
                    empty_clips.append([start_time, moment[0]])
                start_time = moment[1]
            # the last empty clip at the end
            if (torch.tensor(1.0) - start_time) > 0.0001:
                empty_clips.append([start_time, torch.tensor(1.0)])
            empty_clips_len = torch.tensor(([clip[1] - clip[0] for clip in empty_clips]))

            if len(empty_clips_len) == 0:
                # no empty clips to do augmentation
                new_num_targets.append(num_target)
                new_tgt_moments.append(anno['tgt_moments'][shift_t:shift_t + num_target])
                shift_t += num_target
                continue

            if random.random() < self.downsampling_prob:
                # Do downsampling to half the moment length
                mask = [(empty_clips_len > (tgt_moment[1] - tgt_moment[0]) / 2).any()
                        for tgt_moment in tgt_moments]
                mask = torch.tensor(mask).float()

                # check if any target can do augmentation
                if (mask > 0).any():
                    # Do augmentation
                    # Sample one target to do augmentation
                    do_aug_target_idx = torch.multinomial(mask, 1, replacement=False)
                    do_aug_target_moment = tgt_moments[do_aug_target_idx].squeeze()
                    do_aug_target_len = do_aug_target_moment[1] - do_aug_target_moment[0]
                    # Final aug target len after downsampling
                    final_aug_target_len = do_aug_target_len / 2

                    # Find all available augmentation timestamp
                    possible_start_time = []
                    for empty_clip, empty_clip_len in zip(empty_clips, empty_clips_len):
                        if final_aug_target_len < empty_clip_len:
                            for start_time in np.arange(
                                empty_clip[0],
                                empty_clip[1] - final_aug_target_len - 0.02,
                                0.02
                            ):
                                possible_start_time.append(torch.tensor(start_time, dtype=torch.float32))

                    # Check for empty possible_start_time
                    if len(possible_start_time) == 0:
                        new_num_targets.append(num_target)
                        new_tgt_moments.append(anno['tgt_moments'][shift_t:shift_t + num_target])
                        shift_t += num_target
                        continue

                    # Sample from the all possible start index to do augmentation
                    aug_start = random.choice(possible_start_time)
                    aug_end = aug_start + final_aug_target_len
                    seq_len = video_feats[idx].shape[-2]

                    # Target moment
                    target_seq_start_idx = int(seq_len * do_aug_target_moment[0])
                    target_seq_start_idx = max(0, min(target_seq_start_idx, seq_len - 2))
                    target_seq_end_idx = int(seq_len * do_aug_target_moment[1])
                    target_seq_end_idx = max(1, min(target_seq_end_idx, seq_len - 1))
                    target_seq_len = target_seq_end_idx - target_seq_start_idx

                    # Augmented moment: len = target_len / 2
                    aug_seq_start_idx = int(seq_len * aug_start)
                    aug_seq_start_idx = max(0, min(aug_seq_start_idx, seq_len - 2))
                    aug_seq_end_idx = int(seq_len * aug_end)
                    aug_seq_end_idx = max(1, min(aug_seq_end_idx, seq_len - 1))
                    aug_seq_len = aug_seq_end_idx - aug_seq_start_idx

                    # self.downsample will return ceil(target_seq_len / 2)
                    # Make sure the aug_seq_len = ceil(target_seq_len / 2)
                    if aug_seq_len != math.ceil(target_seq_len / 2):
                        aug_seq_end_idx = aug_seq_start_idx + math.ceil(target_seq_len / 2)
                        aug_seq_len = math.ceil(target_seq_len / 2)

                        # aug_seq_end_idx could possiblily == seq_len?
                        if aug_seq_end_idx == seq_len:
                            aug_seq_end_idx = aug_seq_end_idx - 1
                            aug_seq_start_idx = aug_seq_start_idx - 1

                    # Feature-level mixup
                    mixup_feat = self.downsample(
                        self.mixup_alpha * video_feats[idx][target_seq_start_idx:target_seq_end_idx],
                        self.downsampling_method,
                    ) + (1 - self.mixup_alpha) * video_feats[idx][aug_seq_start_idx:aug_seq_end_idx]

                    # Normalize for cosine similarity
                    video_feats[idx][aug_seq_start_idx:aug_seq_end_idx] = F.normalize(mixup_feat.contiguous(), dim=-1)

                    # Add the augmented moment as pseudo-label
                    new_num_targets.append(num_target + 1)
                    new_tgt_moments.append(
                        torch.cat([tgt_moments, torch.tensor([[aug_start, aug_end]])], dim=0)
                    )
                    shift_t += num_target

                else:
                    # No proper empty clip for augmentation
                    new_num_targets.append(num_target)
                    new_tgt_moments.append(anno['tgt_moments'][shift_t:shift_t + num_target])
                    shift_t += num_target

            else:
                # Don't do downsampling
                mask = [(empty_clips_len > (tgt_moment[1] - tgt_moment[0])).any()
                        for tgt_moment in tgt_moments]
                mask = torch.tensor(mask).float()     # [0, 1, 0, ...]

                # check if any target can do augmentation
                if (mask > 0).any():
                    # sample one target to do augmentation
                    do_aug_target_idx = torch.multinomial(mask, 1, replacement=False)
                    do_aug_target_moment = tgt_moments[do_aug_target_idx].squeeze()
                    do_aug_target_len = do_aug_target_moment[1] - do_aug_target_moment[0]
                    final_aug_target_len = do_aug_target_len
                    # find all available augmentation timestamp
                    possible_start_time = []
                    for empty_clip, empty_clip_len in zip(empty_clips, empty_clips_len):
                        if final_aug_target_len < empty_clip_len:
                            for start_time in np.arange(
                                empty_clip[0],
                                empty_clip[1] - final_aug_target_len,
                                0.02
                            ):
                                possible_start_time.append(torch.tensor(start_time, dtype=torch.float32))

                    # check for empty possible_start_time
                    if len(possible_start_time) == 0:
                        new_num_targets.append(num_target)
                        new_tgt_moments.append(anno['tgt_moments'][shift_t:shift_t + num_target])
                        shift_t += num_target
                        continue

                    # sample from the all possible start index to do augmentation
                    aug_start = random.choice(possible_start_time)
                    aug_end = aug_start + final_aug_target_len
                    seq_len = video_feats[idx].shape[-2]

                    # Target moment
                    target_seq_start_idx = int(seq_len * do_aug_target_moment[0])
                    target_seq_start_idx = max(0, min(target_seq_start_idx, seq_len - 2))
                    target_seq_end_idx = int(seq_len * do_aug_target_moment[1])
                    target_seq_end_idx = max(1, min(target_seq_end_idx, seq_len - 1))
                    target_seq_len = target_seq_end_idx - target_seq_start_idx

                    # Augmented moment
                    aug_seq_start_idx = int(seq_len * aug_start)
                    aug_seq_start_idx = max(0, min(aug_seq_start_idx, seq_len - 2))
                    aug_seq_end_idx = aug_seq_start_idx + target_seq_len
                    aug_seq_end_idx = max(1, min(aug_seq_end_idx, seq_len - 1))
                    aug_seq_len = aug_seq_end_idx - aug_seq_start_idx

                    final_seq_len = min(target_seq_len, aug_seq_len)
                    # Make sure the length is the same
                    if target_seq_len > final_seq_len:
                        target_seq_end_idx = target_seq_start_idx + final_seq_len
                    elif aug_seq_len > final_seq_len:
                        aug_seq_end_idx = aug_seq_start_idx + final_seq_len
                    assert (aug_seq_end_idx - aug_seq_start_idx) == (target_seq_end_idx - target_seq_start_idx)

                    # Feature-level mixup
                    mixup_feat = self.mixup_alpha * video_feats[idx][target_seq_start_idx:target_seq_end_idx] \
                        + (1 - self.mixup_alpha) * video_feats[idx][aug_seq_start_idx:aug_seq_end_idx]

                    # Normalize for cosine similarity
                    video_feats[idx][aug_seq_start_idx:aug_seq_end_idx] = F.normalize(mixup_feat.contiguous(), dim=-1)

                    # Add the augmented moment as pseudo-label
                    new_num_targets.append(num_target + 1)
                    new_tgt_moments.append(
                        torch.cat([tgt_moments, torch.tensor([[aug_start, aug_end]])], dim=0)
                    )
                    shift_t += num_target

                else:
                    # No proper empty clip for augmentation
                    new_num_targets.append(num_target)
                    new_tgt_moments.append(anno['tgt_moments'][shift_t:shift_t + num_target])
                    shift_t += num_target

        # Update annotation
        anno['num_targets'] = torch.tensor(new_num_targets)
        anno['tgt_moments'] = torch.cat(new_tgt_moments, dim=0)

        return anno, video_feats

    def __len__(self):
        return len(self.annos)

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
        # Duplicate video feats for each query, anno['num_sentences'] times for augmentation
        video_feats = video_feats.unsqueeze(0).repeat(
            anno['num_sentences'], 1, 1
        )                                   # [num_sent, seq_len, feat_dim]

        if self.do_augmentation:
            anno, video_feats = self.augmentation(anno, video_feats)

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
        video_masks = torch.arange(pad_len)[None, :] < video_lens[:, None]

        if 'aug_frames_st_ed' in batch:
            augmented_data = {
                'aug_frames': torch.cat(batch['aug_frames'], dim=0),
                'aug_frames_st_ed': torch.cat(batch['aug_frames_st_ed'], dim=0),
                'aug_num': torch.cat(batch['aug_num'], dim=0),
            }
        else:
            augmented_data = None

        # Return batch, info
        return {
            'video_feats': torch.cat(batch['video_feats'], dim=0),        # [num_sents, max_seq_len, feat_dim]
            'video_masks': video_masks,                                   # [num_sents, max_seq_len]
            'video_lens': video_lens,                                     # [num_sents]
            'sents_tokens': sentences['input_ids'],                       # [num_sents, max_sent_len]
            'sents_masks': sentences['attention_mask'],                   # [num_sents, max_sent_len]
            'num_sentences': torch.stack(batch['num_sentences'], dim=0),  # [bs] sum = num_sents
            'num_targets': torch.cat(batch['num_targets'], dim=0),        # [num_targets]
            'tgt_moments': torch.cat(batch['tgt_moments'], dim=0),        # [num_targets, 2]
        }, {
            'qids': batch['qids'],              # Placeholder for QVHighlights test server evaluation
            'sentences': batch['sentences'],
            'vid': batch['vid'],
            'idx': torch.cat(batch['idx']),     # Sample idx for each query [S]
            'duration': batch['duration'],

            # augmentation
            'augmented_data': augmented_data,
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
