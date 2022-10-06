import importlib
import json
import random
from typing import Dict

import click
import numpy as np
import torch


def CommandAwareConfig(config_param_name):
    class CustomCommandClass(click.Command):
        def invoke(self, ctx):
            config_file = ctx.params[config_param_name]
            if config_file is None:
                return super(CustomCommandClass, self).invoke(ctx)
            with open(config_file) as f:
                configs = json.load(f)
            for param in ctx.params.keys():
                if ctx.get_parameter_source(param) != click.core.ParameterSource.DEFAULT:
                    continue
                if param in configs:
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
    recalls_dict: Dict[str, Dict[str, Dict[str, float]]] = {},
    table_config={
        "1-target": [
            "R@1,IoU=0.7",
            "R@5,IoU=0.7",
        ],
        "multi-target": [
            "R@5,IoU=0.7",
            "R@10,IoU=0.7",
        ],
    }
):
    for table_name, columns in table_config.items():
        rows = [
            [f"Epoch {epoch:2d}"] + columns,
        ]
        for name, recall in recalls_dict.items():
            rows.append([name] + [
                f"{recall[table_name][col] * 100:.2f}"
                for col in columns
            ])
        widths = [
            max(len(row[j]) for row in rows) for j in range(len(rows[0]))]
        total = sum(widths) + 3 * len(widths) - 1
        sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
        print('+' + '-' * total + '+')
        print('|' + table_name.center(total) + '|')
        print(sep)
        for row in rows:
            print('|', end='')
            for width, cell in zip(widths, row):
                print(cell.center(width + 2), end='|')
            print()
            print(sep)
        print()


if __name__ == '__main__':
    import random
    print_table(
        epoch=3,
        recalls_dict={
            "train": {
                "multi-target": {
                    "R@5,IoU=0.7": random.random(),
                    "R@10,IoU=0.7": random.random(),
                },
                "1-target": {
                    "R@1,IoU=0.7": random.random(),
                    "R@5,IoU=0.7": random.random(),
                },
            },
            "test": {
                "multi-target": {
                    "R@5,IoU=0.7": random.random(),
                    "R@10,IoU=0.7": random.random(),
                },
                "1-target": {
                    "R@1,IoU=0.7": random.random(),
                    "R@5,IoU=0.7": random.random(),
                },
            },
        }
    )
