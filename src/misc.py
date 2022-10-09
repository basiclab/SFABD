import importlib
import json
import random
from typing import Dict

import click
import numpy as np
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def CommandAwareConfig(config_param_name):
    class CustomCommandClass(click.Command):
        def invoke(self, ctx):
            config_file = ctx.params[config_param_name]
            if config_file is None:
                return super(CustomCommandClass, self).invoke(ctx)
            with open(config_file) as f:
                configs = json.load(f)
            for param in ctx.params.keys():
                if param != config_param_name and param in configs:
                    if ctx.get_parameter_source(param) == click.core.ParameterSource.DEFAULT:
                        ctx.params[param] = configs[param]
            return super(CustomCommandClass, self).invoke(ctx)
    return CustomCommandClass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_class(module, *args, **kwargs):
    module, class_name = module.rsplit('.', maxsplit=1)
    return getattr(importlib.import_module(module), class_name)(*args, **kwargs)


def print_table(
    epoch: int = 0,
    rows_dict: Dict[str, Dict[str, float]] = {},
    columns=[
        "R@1,IoU=0.5",
        "R@1,IoU=0.7",
        "R@5,IoU=0.5",
        "R@5,IoU=0.7",
    ]
) -> None:
    rows = [[f"Epoch {epoch:2d}"] + columns]
    widths = [len(col) for col in rows[0]]
    for row_name, row_dict in rows_dict.items():
        rows.append([row_name])
        for i, col in enumerate(columns):
            if col in row_dict:
                rows[-1].append(f"{row_dict[col] * 100:.3f}")
            else:
                rows[-1].append("Not Found")
            widths[i] = max(widths[i], len(rows[-1][-1]))
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    print(sep)
    for row in rows:
        print('|', end='')
        for width, cell in zip(widths, row):
            print(cell.center(width + 2), end='|')
        print("\n" + sep)
    print()


if __name__ == '__main__':
    import random
    print_table(
        epoch=3,
        rows_dict={
            "train": {
                "R@1,IoU=0.7": random.random(),
                "R@5,IoU=0.7": random.random(),
            },
            "test": {
                "R@1,IoU=0.7": random.random(),
                "R@5,IoU=0.7": random.random(),
            },
        },
        columns=[
            "R@1,IoU=0.5",
            "R@1,IoU=0.7",
            "R@5,IoU=0.5",
            "R@5,IoU=0.7",
        ]
    )
