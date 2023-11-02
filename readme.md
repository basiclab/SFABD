# Multi-target Video Moment Retrieval

## Datasets
Download dataset from [OneDrive](https://nycu1-my.sharepoint.com/:f:/g/personal/vin30731_ee10_m365_nycu_edu_tw/Eh77TzJT5MJHm-Wkhmg-A8EBszux3d6v39y4hu1EsjuNAA?e=UpV9mx)

Folder structure of `./data`.
```
./data
├── CharadesSTA
│   ├── VGG
│   │   └── vgg_rgb_features.hdf5
│   ├── C3D
│   │   └── Charades_C3D.hdf5
│   ├── I3D
│   │   └── features
│   │       └── video_id.npy
|   |       └── video_id.npy
|   |       └── ...
│   ├── train.json
│   ├── test.json
│   └── multi_test.json
│
├── ActivityNet
│   ├── C3D
│   │   └── activitynet_v1-3_c3d.hdf5
│   ├── I3D
│   │   └── video_id.npy
|   |   └── video_id.npy
|   |   └── ...
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── multi_test.json
│
└── QVHighlights
    ├── features
    │   ├── clip_features
    │   ├── clip_text_features
    │   └── slowfast_features
    ├── train.json
    ├── val.json
    └── test.json
```

## Pyhton Environments
- Install python packages.
    ```bash
    pip install -r requirements.txt
    ```
- (Optional) Boost mAP calculation.
    ```bash
    python setup.py install
    ```

## Training
- Single GPU training.
    ```
    python main.py --config path/to/config.json --logdir path/to/log/dir
    ```

- Multi-GPU training.
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config path/to/config.json --logdir path/to/log/dir
    ```

## Download Model Weight
[OneDrive](https://nycu1-my.sharepoint.com/:f:/g/personal/vin30731_ee10_m365_nycu_edu_tw/EpZ0TOQDHdVBkE-PSbmt_IIBF9hDj3nXDvxBDSHg4jdOPw?e=KczAkv)

## Testing
- Testing best.pth in the logdir
    ```
    python main.py --test_only --config ./config/charades-I3D.json --logdir path/to/log/dir
    ```

## Tensorboard
```
tensorboard --logdir path/to/log
```
