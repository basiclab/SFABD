import argparse

import torch.multiprocessing

from mmn.training import training_loop
from mmn.misc import read_config


def main():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = read_config(args.config)

    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/103#issuecomment-785815218
    torch.multiprocessing.set_sharing_strategy('file_system')

    # TODO: DDP
    training_loop(config)


if __name__ == "__main__":
    main()
