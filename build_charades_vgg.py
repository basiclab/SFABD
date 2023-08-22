import os
import glob

import h5py
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights


class CharadesFrames(Dataset):
    def __init__(self, root_dir, transform, skip_ids=set(), sample_rate=4):
        self.transform = transform

        frame_dirs = sorted(glob.glob(os.path.join(root_dir, '*/')))
        self.id_str2idx = {}
        self.id_idx2str = []
        for frame_dir in frame_dirs:
            id_str = os.path.basename(os.path.dirname(frame_dir))
            self.id_idx2str.append(id_str)
            self.id_str2idx[id_str] = len(self.id_idx2str) - 1
        self.num_videos = len(self.id_idx2str)

        self.frames = []
        for frame_dir in frame_dirs:
            id_str = os.path.basename(os.path.dirname(frame_dir))
            id_idx = self.id_str2idx[id_str]
            if id_str not in skip_ids:
                frame_files = glob.glob(os.path.join(frame_dir, '*.jpg'))
                frame_files = sorted(frame_files)
                # print(id_str, len(frame_files))
                frame_files = frame_files[::sample_rate]
                # print(id_str, len(frame_files))
                for frame_file in frame_files:
                    self.frames.append((frame_file, id_idx))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame_file, idx = self.frames[index]
        frame = Image.open(frame_file)
        if self.transform:
            frame = self.transform(frame)

        return frame, idx


def main():
    device = torch.device('cuda:0')

    # find existing ids
    if os.path.exists("pt_vgg_rgb_features.hdf5"):
        with h5py.File("pt_vgg_rgb_features.hdf5", "r") as f:
            skip_ids = set(f.keys())
    else:
        skip_ids = set()

    dataset = CharadesFrames(
        root_dir='./data/CharadesSTA/Charades_v1_rgb',
        transform=VGG16_Weights.DEFAULT.transforms(),
        skip_ids=skip_ids,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False)

    vgg = vgg16(weights=VGG16_Weights.DEFAULT, progress=True)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    vgg = vgg.to(device)
    vgg.eval()

    with h5py.File("pt_vgg_rgb_features.hdf5", "a") as f:
        with tqdm(
            ncols=0,
            initial=len(skip_ids),
            total=dataset.num_videos,
        ) as pbar:
            buffer_feature = torch.empty(0, 4096, device=device).float()
            buffer_idx = torch.empty(0).long()
            for frames, idxs in loader:
                frames = frames.to(device)
                with torch.no_grad():
                    features = vgg(frames)
                buffer_feature = torch.cat([buffer_feature, features], dim=0)
                buffer_idx = torch.cat([buffer_idx, idxs], dim=0)

                used = 0
                for i in range(1, len(buffer_idx)):
                    if buffer_idx[i] != buffer_idx[i - 1]:
                        id_str = dataset.id_idx2str[buffer_idx[i - 1].item()]
                        feature = buffer_feature[used:i].cpu().numpy()
                        f.create_dataset(id_str, data=feature)
                        pbar.write(f'[{id_str}] {feature.shape}')
                        pbar.update(1)
                        used = i
                buffer_feature = buffer_feature[used:]
                buffer_idx = buffer_idx[used:]

            if len(buffer_idx) > 0:
                assert len(set(buffer_idx.tolist())) == 1
                id_str = dataset.id_idx2str[buffer_idx[-1].item()]
                feature = buffer_feature.cpu().numpy()
                f.create_dataset(id_str, data=feature)
                pbar.write(f'[{id_str}] {feature.shape}')
                pbar.update(1)


if __name__ == '__main__':
    main()
