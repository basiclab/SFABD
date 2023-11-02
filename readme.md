# Multi-target Video Moment Retrieval

## Datasets
- Download dataset from [OneDrive](https://nycu1-my.sharepoint.com/:f:/g/personal/vin30731_ee10_m365_nycu_edu_tw/Eh77TzJT5MJHm-Wkhmg-A8EBszux3d6v39y4hu1EsjuNAA?e=UpV9mx)
- Make a folder to contain all data and 
    ```
    mkdir data
    ```
  
- Move the downloaded data to ./data folder
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
- Make a folder to contain all logs
    ```
    mkdir logs
    ```
- Move the downloaded logs to ./logs
  ```
  ./logs
  ├── activity-C3D-log
  ├── activity-I3D-log
  ├── charades-C3D-log
  ├── charades-I3D-log
  ├── charades-VGG-log
  └── qv-log
  ```

## Testing
- Testing best.pth in the logdir
    ```
    python main.py --test_only --config ./logs/activity-C3D-log/config.json --logdir ./logs/activity-C3D-log
    ```

## Tensorboard
- The tensorboard records of the trained models are not kept, need to train from stratch 
    ```
    tensorboard --logdir path/to/log
    ```
