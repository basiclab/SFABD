# Multi-target Video Moment Retrieval

## Datasets
Folder structure of `./data`.
```
./data
├── CharadesSTA
│   ├── test.json
│   ├── test_multitarget.json
│   ├── train.json
│   ├── train_multitarget.json
│   └── vgg_rgb_features.hdf5
└── QVHighlights
    ├── features
    │   ├── clip_features
    │   ├── clip_text_features
    │   └── slowfast_features
    ├── test.json
    ├── train.json
    └── val.json
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

## Submission
- Make sumbission zip file.
    ```
    python main.py --test_only --config path/to/config.json --logdir path/to/log/dir
    ```

## Tensorboard
```
tensorboard --logdir path/to/log
```
