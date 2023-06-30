# Multi-target Video Moment Retrieval

## Datasets
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
│   ├── train.json
│   ├── test.json
│   └── multi_test.json
│
├── ActivityNet
│   ├── C3D
│   │   └── activitynet_v1-3_c3d.hdf5
│   ├── I3D
│   │   └── video_id.npy
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

## Testing
- Testing best.pth in the logdir.
    ```
    python main.py --test_only --config path/to/config.json --logdir path/to/log/dir
    ```

## Tensorboard
```
tensorboard --logdir path/to/log
```
