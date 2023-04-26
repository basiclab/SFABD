import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import CollateBase


class CharadesVGG(CollateBase):
    def __init__(
        self,
        ann_file,           # path to annotation file (.json)
        feat_file,          # path to feature file
    ):
        super().__init__(ann_file)
        self.feat_file = feat_file

    def get_feat_dim(self):
        return 4096

    # override
    def get_feat(self, anno):
        with h5py.File(self.feat_file, 'r') as f:
            feats = f[anno['vid']][:]
            feats = torch.from_numpy(feats).float()
            feats = F.normalize(feats, dim=-1)
        return feats


class CharadesSTAVGGTrain(CharadesVGG):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/train.json",
            feat_file="./data/CharadesSTA/VGG/vgg_rgb_features.hdf5")


class CharadesSTAVGGTest(CharadesVGG):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/test.json",
            feat_file="./data/CharadesSTA/VGG/vgg_rgb_features.hdf5")


class CharadesSTAVGGMultiTest(CharadesVGG):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/multi_test.json",
            feat_file="./data/CharadesSTA/VGG/vgg_rgb_features.hdf5")


class CharadesI3D(CollateBase):
    def __init__(
        self,
        ann_file,           # path to annotation file (.json)
        feat_dir,          # path to feature file
    ):
        super().__init__(ann_file)
        self.feat_dir = feat_dir

    # def augmentation(self, anno, video_feats):
    #     """Do multi positive augmentation"""
    #     duration = anno['duration']
    #     # target moments of whole query-moment pairs in same video
    #     moments = anno['tgt_moments']
    #     print(f"num_targets:{anno['num_targets']}")
    #     print(f"num_targets:{len(anno['num_targets'])}")
    #     print(f"num sentences: {anno['num_sentences']}")
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

    def get_feat_dim(self):
        return 1024

    # override
    def get_feat(self, anno):
        path = os.path.join(self.feat_dir, f"{anno['vid']}.npy")
        feats = np.load(path, 'r')                  # [seq_len, 1, 1, 1024]
        feats = torch.from_numpy(feats.copy()).float()
        feats = feats.squeeze(1).squeeze(1)         # [seq_len, 1024]
        feats = F.normalize(feats, dim=-1)

        return feats


class CharadesSTAI3DTrain(CharadesI3D):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/train.json",
            feat_dir="./data/CharadesSTA/I3D/features/")


class CharadesSTAI3DTest(CharadesI3D):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/test.json",
            feat_dir="./data/CharadesSTA/I3D/features/")


class CharadesSTAI3DMultiTest(CharadesI3D):
    def __init__(self):
        super().__init__(
            ann_file="./data/CharadesSTA/multi_test.json",
            feat_dir="./data/CharadesSTA/I3D/features/")
