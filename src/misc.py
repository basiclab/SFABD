import importlib
import json
import random
import warnings
from typing import Dict, Union, List

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
            unused = [k for k in configs.keys() if k not in ctx.params.keys()]
            if len(unused) > 0:
                warnings.warn(f'{",".join(unused)} in the config file is/are not used')
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
    rows: Dict[str, Dict[str, float]] = {},
    keys: Union[None, List[str]] = None,
) -> None:
    column_names = set()
    for row in rows.values():
        if keys:
            for key in row.keys():
                if any(key.startswith(k) for k in keys):
                    column_names.add(key)
        else:
            column_names.update(row.keys())
    column_names = sorted(column_names)

    rows_str = [[f"Epoch {epoch:2d}", *column_names]]
    rows_width = [len(head) for head in rows_str[0]]
    for row_name, row in rows.items():
        row_str = [row_name]
        for column in column_names:
            if column in row:
                row_str.append(f"{row[column] * 100:.3f}")
            else:
                row_str.append("Not Found")
        for i in range(len(row_str)):
            rows_width[i] = max(rows_width[i], len(row_str[i]))
        rows_str.append(row_str)

    sep = '+' + '+'.join('-' * (w) for w in rows_width) + '+'
    print(sep)
    for row_str in rows_str:
        print('|', end='')
        for width, cell in zip(rows_width, row_str):
            print(cell.center(width), end='|')
        print("\n" + sep)


if __name__ == '__main__':
    import random
    print_table(
        epoch=3,
        rows={
            "train": {
                "R@1,IoU=0.5": random.random(),
                "R@1,IoU=0.7": random.random(),
                "R@5,IoU=0.5": random.random(),
                "R@5,IoU=0.7": random.random(),
            },
            "test": {
                "R@1,IoU=0.5": random.random(),
                "R@1,IoU=0.7": random.random(),
                "R@5,IoU=0.5": random.random(),
                "R@5,IoU=0.7": random.random(),
            },
        },
    )
