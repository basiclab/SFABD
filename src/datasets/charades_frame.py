import glob
import os
import random
from typing import Tuple, Dict

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

from src.datasets.base import CollateBase
from src import dist


# VGG feature
class CharadesVGG(CollateBase):
    def __init__(
        self,
        ann_file,           # path to annotation file (.json)
        feat_file,          # path to feature file
        frame_root,         # path to frame directory
        transform,          # augmentation
        sample_rate=4,      # 24 / 4 = 6 fps
    ):
        super().__init__(ann_file, do_augmentation=False)
        self.feat_file = feat_file
        self.frame_root = frame_root
        self.transform = transform
        self.sample_rate = sample_rate
        self.img_transform = VGG16_Weights.DEFAULT.transforms()

        # VGG16
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT, progress=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        self.vgg = self.vgg.to(dist.get_device())
        self.vgg.eval()

    def get_frames(self, anno: Dict) -> torch.Tensor:
        frame_dir = os.path.join(self.frame_root, anno['vid'])
        frame_files = glob.glob(os.path.join(frame_dir, '*.jpg'))
        frame_files = sorted(frame_files)
        frame_files = frame_files[::self.sample_rate]
        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file)
            frame = self.img_transform(frame)
            frames.append(frame)
        return torch.stack(frames, dim=0)

    # override
    def get_feat_dim(self):
        return 4096

    # override
    def get_feat(self, anno):
        with h5py.File(self.feat_file, 'r') as f:
            feats = f[anno['vid']][:]
            feats = torch.from_numpy(feats).float()
            feats = F.normalize(feats, dim=-1)
        return feats

    # override
    def __len__(self):
        return len(self.annos)

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
        frames = self.get_frames(anno)                  # [seq_len, 3, 224, 224]
        video_feats = self.get_feat(anno)               # [seq_len, 4096]
        assert video_feats.shape[0] == frames.shape[0]

        # augmentation
        new_video_feats_list = []
        new_tgt_moments_list = []
        new_num_targets_list = []
        for tgt_moments in anno['tgt_moments'].split(anno['num_targets'].tolist()):
            new_video_feats = video_feats.clone()
            if self.transform:
                # apply augmentation
                new_frames, new_tgt_moments = self.transform(frames.clone(), tgt_moments)
                # extract vgg features for augmented frames only
                for moments in new_tgt_moments[len(tgt_moments):]:
                    st = round((frames.shape[0] - 1) * moments[0].item())
                    ed = round((frames.shape[0] - 1) * moments[1].item()) + 1
                    assert st < ed
                    feats_list = []
                    for batch in new_frames[st:ed].split(32):
                        batch = batch.to(dist.get_device())
                        with torch.no_grad():
                            feats_list.append(self.vgg(batch).cpu())
                    feats = torch.cat(feats_list, dim=0)    # [ed - st, 4096]
                    new_video_feats[st:ed] = feats

                new_video_feats_list.append(new_video_feats)
                new_tgt_moments_list.append(new_tgt_moments)
                new_num_targets_list.append(len(new_tgt_moments))
            else:
                new_video_feats_list.append(new_video_feats)
                new_tgt_moments_list.append(tgt_moments)
                new_num_targets_list.append(len(tgt_moments))

        video_feats = torch.stack(new_video_feats_list, dim=0)
        tgt_moments = torch.cat(new_tgt_moments_list, dim=0)
        num_targets = torch.tensor(new_num_targets_list)

        return {
            'idx': torch.ones(anno['num_sentences'], dtype=torch.long) * idx,
            'video_feats': video_feats,
            **anno,
            'tgt_moments': tgt_moments,
            'num_targets': num_targets,
        }


class CharadesSTAVGGTrain(CharadesVGG):
    def __init__(self, *args, **kwargs):
        super().__init__(
            ann_file="./data/CharadesSTA/train.json",
            feat_file="./data/CharadesSTA/VGG/pt_vgg_rgb_features.hdf5",
            frame_root="./data/CharadesSTA/Charades_v1_rgb",
            transform=Mixup(),
        )


class AugmentationBase:
    def __init__(
        self,
        min_num: float = 0.0,
        max_num: float = 1.0,
    ):
        self.min_num = min_num
        self.max_num = max_num

    def fuse(
        self, back_frames: torch.Tensor, front_frames: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self, frames: torch.Tensor, tgt_moments: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_moments = tgt_moments.tolist()
        num_frames = frames.shape[0]
        num_targets = len(tgt_moments)
        num_mix = random.random() * (self.max_num - self.min_num) + self.min_num
        # print('num_mix * num_targets', num_mix * num_targets)
        num_mix = round(num_mix * num_targets)
        # print(num_mix)
        # num_mix = 1

        front_moments = random.sample(tgt_moments, num_mix)
        for front_moment in front_moments:
            front_st = round((num_frames - 1) * front_moment[0])
            front_ed = round((num_frames - 1) * front_moment[1]) + 1
            assert front_st < front_ed
            front_len = front_ed - front_st

            # Find all possible start frames in background
            back_st_candidates = []
            prev = 0
            for tgt_moment in sorted(tgt_moments, key=lambda m: m[0]):
                tgt_st = round((num_frames - 1) * tgt_moment[0])
                tgt_ed = round((num_frames - 1) * tgt_moment[1]) + 1
                assert tgt_st < tgt_ed
                if tgt_st - prev > front_len:
                    back_st_candidates.extend(
                        list(range(prev, tgt_st - front_len)))
                prev = tgt_ed
            if num_frames - prev > front_len:
                back_st_candidates.extend(
                    list(range(prev, num_frames - front_len)))
            # print(back_st_candidates)

            if len(back_st_candidates) > 0:
                # If at least one candidate exists, randomly choose one start
                # frame in background
                back_st = random.choice(back_st_candidates)
                back_ed = back_st + front_len
                frames[back_st: back_ed] = self.fuse(
                    frames[back_st: back_ed], frames[front_st: front_ed])
                tgt_moments.append((
                    back_st / (num_frames - 1), (back_ed - 1) / (num_frames - 1)
                ))

        return frames, torch.tensor(tgt_moments)


class Mixup(AugmentationBase):
    def __init__(
        self,
        alpha: float = 0.9,
        min_num: float = 0.0,
        max_num: float = 1.0,
    ):
        super().__init__(min_num, max_num)
        self.alpha = alpha

    def fuse(
        self, back_frames: torch.Tensor, front_frames: torch.Tensor
    ) -> torch.Tensor:
        return (1 - self.alpha) * back_frames + self.alpha * front_frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = CharadesSTAVGGTrain()
    data = dataset[1]
    print(data['video_feats'].shape)
    print(data['tgt_moments'].shape)
    print(data['num_targets'])

    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    batch, info = next(iter(loader))
    for key, value in batch.items():
        print(key, value.shape)
