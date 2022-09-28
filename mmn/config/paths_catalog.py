"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "datasets/"

    DATASETS = {
        "tacos_train":{
            "video_dir": "TACoS/videos",
            "ann_file": "TACoS/train.json",
            "feat_file": "TACoS/tall_c3d_features.hdf5",
        },
        "tacos_val":{
            "video_dir": "TACoS/videos",
            "ann_file": "TACoS/val.json",
            "feat_file": "TACoS/tall_c3d_features.hdf5",
        },
        "tacos_test":{
            "video_dir": "TACoS/videos",
            "ann_file": "TACoS/test.json",
            "feat_file": "TACoS/tall_c3d_features.hdf5",
        },
        "activitynet_train":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/train.json",
            "feat_file": "ActivityNet/sub_activitynet_v1-3_c3d.hdf5",
        },
        "activitynet_val":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/val.json",
            "feat_file": "ActivityNet/sub_activitynet_v1-3_c3d.hdf5",
        },
        "activitynet_test":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/test.json",
            "feat_file": "ActivityNet/sub_activitynet_v1-3_c3d.hdf5",
        },
        "charades_train": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/charades_train.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },

        "charades_combined_train": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/combined_charades_train.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },

        "charades_combined_train_remove_repeat_action_video": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/combined_charades_train_remove_repeat_action_videos.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },


        "charades_separate_train_1gt": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/separate_charades_1gt_train.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },

        "charades_separate_train_multi_gt": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/separate_charades_multi_gt_train.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },

        ## augmented
        "charades_separate_augmented": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/charades_train_aug_query_templates.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },

        "charades_combined_test_authentic_multi_target": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/charades_multi_target.json",
            "feat_file": "Charades_STA/Charades_vgg_rgb.h5df",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },        

        "charades_test": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/charades_test.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },
        "charades_combined_test": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/combined_charades_test.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },

        "charades_separate_test_1gt": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/separate_charades_1gt_test.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },

        "charades_separate_test_multi_gt": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/separate_charades_multi_gt_test.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
            "c3d_feat_folder": "/media/vin30731/Backup/Charades-STA/Charades-STA_C3D/C3D_features"
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            #root=os.path.join(data_dir, attrs["video_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
            c3d_feat_folder=attrs["c3d_feat_folder"],
        )

        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "charades_combined" in name:
            return dict(
                factory = "CharadesCombinedDataset",
                args = args
            )
        elif "charades_separate" in name:
            return dict(
                factory = "CharadesCombinedDataset",
                args = args
            )


        elif "charades" in name:
            return dict(
                factory = "CharadesDataset",
                args = args
            )


        raise RuntimeError("Dataset not available: {}".format(name))

