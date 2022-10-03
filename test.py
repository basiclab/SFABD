import argparse
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from transformers import DistilBertTokenizer

from mmn.config import cfg
from mmn.data.utils import avgfeats, bert_embedding
from mmn.inference_loop import inference
from mmn.inference_loop import inference_specific_video, inference_combined_dataset
from mmn.utils.checkpoint import MmnCheckpointer


def main():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    output_dir = cfg.OUTPUT_DIR
    # used to organized checkpoint
    checkpointer = MmnCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)       # load checkpoint

    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False)[0]

    _ = inference(
        cfg, model, data_loaders_val,
        dataset_name=dataset_names,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
    )


def main_specific_video(video_name):
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    output_dir = cfg.OUTPUT_DIR
    # used to organized checkpoint
    checkpointer = MmnCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None) # load checkpoint


    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False)[0]

    print(f"video_name:{video_name}")
    # inference only one video
    result = inference_specific_video(
        cfg, model, data_loaders_val,
        video_name=video_name,
        dataset_name=dataset_names,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
    )


def main_new_video():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    output_dir = cfg.OUTPUT_DIR
    # used to organized checkpoint
    checkpointer = MmnCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None) # load checkpoint

    # Need these data
    video_file = "v_otWTm1_aAqI.mp4"
    video_name = os.path.basename(video_file)[0:-4]
    video_data_dict = {}
    video_data_dict['video_name'] = video_name
    # feat(256, 500), query(list of word_idx), word_len, sentence (number of query)
    # load video c3d feature
    feat = np.load(f'{video_name}.npy')
    feat = F.normalize(torch.from_numpy(feat), dim=1)
    feat = avgfeats(feat, cfg.INPUT.NUM_INIT_CLIPS) # average the c3d feature -> [256, 500]
    video_data_dict['feat'] = feat

    # load query
    # query = "A boy jumps onto a balance beam."
    # query = "He does a gymnastics routine on the balance beam."
    query = "He dismounts and lands on the blue mat."
    # need word_dict
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # print(f"tokenizer:{tokenizer}")
    query, word_len = bert_embedding(query, tokenizer)
    # print(f"query:{query}, word_len:{word_len}")
    video_data_dict['query'] = query
    video_data_dict['word_len'] = word_len
    video_data_dict['num_sentence'] = 1         # 1 query
    # need to save video duration info
    video_data_dict['duration'] = 40.59
    inference_new_video(
        cfg, video_data_dict, model,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
    )


def main_combined_dataset():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    output_dir = cfg.OUTPUT_DIR
    # used to organized checkpoint
    checkpointer = MmnCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)       # load checkpoint

    dataset_names = cfg.DATASETS.TEST
    # dataset_names = cfg.DATASETS.TRAIN # Modified
    print(f"dataset_names:{dataset_names}")

    data_loaders_val = make_data_loader(cfg, is_train=False)[0]

    # return avg result as a dict
    _ = inference_combined_dataset(
        cfg, model, data_loaders_val,
        dataset_name=dataset_names,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
    )


if __name__ == "__main__":
    start = time.time()
    # original main function
    # main()

    # evaluate specific video in the dataset
    # main_specific_video(video_name='v_1B3XsffrM4M')

    # evaluate new video outside the dataset
    # main_new_video()

    # evaluate the combined dataset
    main_combined_dataset()

    end = time.time()

    print(f"Run time:{end - start}")
