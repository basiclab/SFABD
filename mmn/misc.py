import random
import json
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path: str):
    def to_attrdict(d: Dict):
        attrdict = AttrDict()
        for k, v in d.items():
            if isinstance(v, dict):
                attrdict[k] = to_attrdict(v)
            else:
                attrdict[k] = v
        return attrdict
    config = json.load(open(path, 'r'))
    return to_attrdict(config)


if __name__ == '__main__':
    config = read_config(
        './configs/pool_charades_32x32_k5l8_combined.json')
    print(config.model.conv1d.in_channel)
