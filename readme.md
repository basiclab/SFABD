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
    ├── QVHighlights_c3d.hdf5
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
    python train.py --config path/to/config.json
    ```

- Multi-GPU training.
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config path/to/config.json
    ```
