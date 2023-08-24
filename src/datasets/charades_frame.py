import glob
import os
import random
from typing import Tuple, List

import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import VGG16_Weights
from tqdm import tqdm

from src.datasets.base import CollateBase


# VGG feature
class CharadesVGG(CollateBase):
    def __init__(
        self,
        do_augmentation,
        aug_prob,
        downsampling_prob,
        ann_file,           # path to annotation file (.json)
        feat_file,          # path to feature file
        frame_root,         # path to frame directory
        sample_rate=4,      # 24 / 4 = 6 fps
    ):
        super().__init__(
            ann_file,
            do_augmentation,
            0.0,
            'odd',
            aug_prob,
            downsampling_prob,
        )
        self.feat_file = feat_file
        self.frame_root = frame_root
        self.sample_rate = sample_rate

        # files cache
        self.frame_transform = VGG16_Weights.DEFAULT.transforms()
        self.frame_files = dict()

    def get_frame_files(self, vid: str) -> List[str]:
        if vid not in self.frame_files:
            frame_dir = os.path.join(self.frame_root, vid)
            frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
            frame_files = frame_files[::self.sample_rate]
            self.frame_files[vid] = frame_files
        return self.frame_files[vid]

    def read_frames(
        self,
        vid: str,
        st: int,
        ed: int,
        downsample: bool = False
    ) -> torch.Tensor:
        frame_files = self.get_frame_files(vid)

        assert st < ed
        assert ed <= len(frame_files)

        frames = []
        for frame_file in frame_files[st: ed: 2 if downsample else 1]:
            frame_temp = Image.open(frame_file)
            frame = self.frame_transform(frame_temp.copy())
            frames.append(frame)
            frame_temp.close()
        return torch.stack(frames, dim=0)

    def augment(
        self,
        vid: str,
        num_frames: int,
        tgt_moments: torch.Tensor,
        num_sample: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tgt_moments = tgt_moments.tolist()
        aug_frames_st_ed = [torch.zeros(0, 2)]
        aug_frames = [torch.zeros(0, 3, 224, 224).float()]
        tgt_frames = [torch.zeros(0, 3, 224, 224).float()]

        if self.do_augmentation and random.random() > self.aug_prob:
            for tgt_moment in random.sample(tgt_moments, num_sample):
                tgt_st = round((num_frames - 1) * tgt_moment[0])
                tgt_ed = round((num_frames - 1) * tgt_moment[1]) + 1
                assert tgt_st < tgt_ed
                tgt_len = tgt_ed - tgt_st

                downsample = random.random() < self.downsampling_prob
                if downsample:
                    tgt_len = (tgt_len // 2) + (tgt_len % 2)

                # Find all possible start frames in source video
                st_candidates = []
                prev = 0
                for moment in sorted(tgt_moments, key=lambda moment: moment[0]):
                    st = round((num_frames - 1) * moment[0])
                    ed = round((num_frames - 1) * moment[1]) + 1
                    assert st < ed
                    if st - prev >= tgt_len:
                        st_candidates.extend(range(prev, st - tgt_len + 1))
                    prev = ed
                if num_frames - prev >= tgt_len:
                    st_candidates.extend(
                        list(range(prev, num_frames - tgt_len + 1)))

                if len(st_candidates) > 0:
                    # If at least one candidate exists, randomly choose one start
                    # index in source video
                    aug_st = random.choice(st_candidates)
                    aug_ed = aug_st + tgt_len
                    aug_frames_st_ed.append(torch.tensor([[aug_st, aug_ed]]))
                    aug_frames.append(self.read_frames(vid, aug_st, aug_ed))
                    tgt_frames.append(self.read_frames(vid, tgt_st, tgt_ed, downsample))
                    aug_moment = [
                        aug_st / (num_frames - 1),
                        (aug_ed - 1) / (num_frames - 1)
                    ]
                    tgt_moments.append(aug_moment)

        aug_frames_st_ed = torch.cat(aug_frames_st_ed, dim=0).long()
        aug_frames = torch.cat(aug_frames, dim=0)
        tgt_frames = torch.cat(tgt_frames, dim=0)
        tgt_moments = torch.tensor(tgt_moments).float()

        return aug_frames_st_ed, aug_frames, tgt_frames, tgt_moments

    # override
    def get_feat_dim(self):
        return 4096

    # override
    def get_feat(self, vid):
        with h5py.File(self.feat_file, 'r') as f:
            feats = f[vid][:]
            feats = torch.from_numpy(feats).float()
            feats = F.normalize(feats, dim=-1)
        return feats

    # override
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
        vid = anno['vid']
        video_feats = self.get_feat(vid)
        num_frames = len(self.get_frame_files(vid))
        assert video_feats.shape[0] == num_frames

        # augmentation
        aug_frames_st_ed_list = []
        aug_frames_list = []
        tgt_frames_list = []
        aug_num = []
        new_tgt_moments_list = []
        new_num_targets_list = []
        for tgt_moments in anno['tgt_moments'].split(anno['num_targets'].tolist()):
            aug_frames_st_ed, aug_frames, tgt_frames, tgt_moments = \
                self.augment(vid, num_frames, tgt_moments)
            aug_frames_st_ed_list.append(aug_frames_st_ed)
            aug_frames_list.append(aug_frames)
            tgt_frames_list.append(tgt_frames)
            aug_num.append(len(aug_frames_st_ed))

            new_tgt_moments_list.append(tgt_moments)
            new_num_targets_list.append(len(tgt_moments))

        aug_frames_st_ed = torch.cat(aug_frames_st_ed_list, dim=0)
        aug_frames = torch.cat(aug_frames_list, dim=0)
        tgt_frames = torch.cat(tgt_frames_list, dim=0)
        aug_num = torch.tensor(aug_num)

        tgt_moments = torch.cat(new_tgt_moments_list, dim=0)
        num_targets = torch.tensor(new_num_targets_list)
        video_feats = video_feats.unsqueeze(0).repeat(
            anno['num_sentences'], 1, 1
        )

        return {
            'idx': torch.ones(anno['num_sentences'], dtype=torch.long) * idx,
            'video_feats': video_feats,
            'aug_frames': aug_frames,
            'tgt_frames': tgt_frames,
            'aug_frames_st_ed': aug_frames_st_ed,   # [sum(aug_num), 2]
            'aug_num': aug_num,                     # [num_sentences]
            **anno,
            'tgt_moments': tgt_moments,
            'num_targets': num_targets,
        }


class CharadesSTAVGGTrain(CharadesVGG):
    def __init__(
        self,
        do_augmentation=True,
        aug_prob=0.25,
        downsampling_prob=0.5,
        **kwargs
    ):
        super().__init__(
            do_augmentation,
            aug_prob,
            downsampling_prob,
            ann_file="./data/CharadesSTA/train.json",
            feat_file="./data/CharadesSTA/VGG/pt_vgg_rgb_features.hdf5",
            frame_root="./data/CharadesSTA/Charades_v1_rgb",
        )


if __name__ == '__main__':
    import torch.multiprocessing
    from torch.utils.data import DataLoader
    from src.training import update_vgg_features

    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    dataset = CharadesSTAVGGTrain()
    for k, v in dataset[1].items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, list):
            print(k, len(v))
        else:
            print(k, v)
    print('---')

    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=dataset.collate_fn,
        num_workers=1,
    )
    batch, info = next(iter(loader))
    augmented_data = info['augmented_data']
    for key, value in augmented_data.items():
        print(key, value.shape)

    print(batch['tgt_moments'].shape, batch['num_targets'].sum())
    print(augmented_data['aug_frames_st_ed'].shape, augmented_data['aug_num'].sum())
    print('---')

    for batch, batch_info in tqdm(loader, ncols=0, desc='simulate training'):
        batch['video_feats'] = update_vgg_features(
            batch['video_feats'],
            batch_info['augmented_data'],
            mixup_alpha=0.9)
        assert batch['tgt_moments'].shape[0] == batch['num_targets'].sum()

        augmented_data = info['augmented_data']
        assert augmented_data['aug_frames_st_ed'].shape[0] == augmented_data['aug_num'].sum()
