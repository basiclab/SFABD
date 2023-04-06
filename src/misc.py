import importlib
import json
import random
import warnings
from collections import defaultdict
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


def build_table(cells, row_names, col_names) -> List[str]:
    col_widths = {
        col_name: len(col_name)
        for col_name in col_names
    }
    for row_name in row_names:
        for col_name in col_names:
            col_widths[col_name] = max(
                col_widths[col_name],
                len("%5.2f" % (cells[row_name][col_name] * 100)))
    col_widths['col'] = max(len(row_name) for row_name in row_names)
    total_width = sum(col_widths.values()) + 3 * len(col_widths) + 1

    # build table
    table = ["-" * total_width]

    # header
    header = ['|', ' ' * col_widths['col'], '|']
    for col_name in col_names:
        header.append(col_name.center(col_widths[col_name]))
        header.append('|')
    table.append(" ".join(header))

    # separator
    table.append("-" * total_width)

    # body
    for row_name in row_names:
        row = ['|', f"{row_name:<{col_widths['col']}}", '|']
        for col_name in col_names:
            mAP_str = f"{cells[row_name][col_name] * 100:5.2f}"
            row.append(mAP_str.center(col_widths[col_name]))
            row.append('|')
        table.append(" ".join(row))

    # footer
    table.append("-" * total_width)
    return table


def build_mAP(
    mAPs: Dict[str, float],
    row_names: List[str] = ['all', 'sh', 'md', 'lg', 'sgl', 'mul'],
) -> None:
    col_names = set()
    cells = defaultdict(dict)
    for key, value in mAPs.items():
        row_name, col_name = key.split('/')
        col_names.add(col_name)
        cells[row_name][col_name] = value
        assert row_name in row_names
    col_names = sorted(list(col_names))

    return build_table(cells, row_names, col_names)


def build_recall(recalls: Dict[str, float]) -> None:
    row_names = set()
    col_names = set()
    cells = defaultdict(dict)
    for key, value in recalls.items():
        row_name, col_name = key.split(',')
        row_names.add(row_name)
        col_names.add(col_name)
        cells[row_name][col_name] = value
    row_names = sorted(list(row_names))
    col_names = sorted(list(col_names))

    return build_table(cells, row_names, col_names)


def print_metrics(mAPs: Dict[str, float], recalls: Dict[str, float]) -> None:
    table1 = build_mAP(mAPs)
    table2 = build_recall(recalls)
    rows = max(len(table1), len(table2))

    while len(table1) < rows:
        table1.append(' ' * len(table1[0]))

    while len(table2) < rows:
        table2.append(' ' * len(table2[0]))

    for row1, row2 in zip(table1, table2):
        print(f"{row1}   {row2}")


def print_multi_recall(multi_recalls: Dict[str, float]) -> None:
    table = build_recall(multi_recalls)
    for row in table:
        print(f"{row}")


if __name__ == '__main__':
    import random
    print_metrics({
        "all/mAP@0.50": 0.24011583626270294,
        "all/mAP@0.75": 0.13699504733085632,
        "all/mAP@avg": 0.14617718756198883,
        "sgl/mAP@0.50": 0.30263885855674744,
        "sgl/mAP@0.75": 0.17896008491516113,
        "sgl/mAP@avg": 0.19009943306446075,
        "mul/mAP@0.50": 0.11978838592767715,
        "mul/mAP@0.75": 0.05623213201761246,
        "mul/mAP@avg": 0.061647556722164154,
        "sh/mAP@0.50": 0.113711416721344,
        "sh/mAP@0.75": 0.061229877173900604,
        "sh/mAP@avg": 0.06487669795751572,
        "md/mAP@0.50": 0.23680077493190765,
        "md/mAP@0.75": 0.11836647987365723,
        "md/mAP@avg": 0.1287669688463211,
        "lg/mAP@0.50": 0.9484522938728333,
        "lg/mAP@0.75": 0.6347619295120239,
        "lg/mAP@avg": 0.6652143001556396
    }, {
        "R@1,IoU=0.5": 0.06974927751308287,
        "R@1,IoU=0.7": 0.045926735921268454,
        "R@5,IoU=0.5": 0.15980629539951574,
        "R@5,IoU=0.7": 0.10317894243536671
    })
