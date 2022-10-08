import importlib
import json
import pickle
import random
from typing import Dict

import click
import numpy as np
import torch
import torch.distributed as dist


def CommandAwareConfig(config_param_name):
    class CustomCommandClass(click.Command):
        def invoke(self, ctx):
            config_file = ctx.params[config_param_name]
            if config_file is None:
                return super(CustomCommandClass, self).invoke(ctx)
            with open(config_file) as f:
                configs = json.load(f)
            for param in ctx.params.keys():
                if param in configs:
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
    recalls_dict: Dict[str, Dict[str, float]] = {},
    columns=[
        "R@1,IoU=0.5",
        "R@1,IoU=0.7",
        "R@5,IoU=0.5",
        "R@5,IoU=0.7",
    ]
):
    rows = [
        [f"Epoch {epoch:2d}"] + columns,
    ]
    for name, recall in recalls_dict.items():
        rows.append([name] + [f"{recall[col] * 100:.2f}" for col in columns])
    widths = [
        max(len(row[j]) for row in rows) for j in range(len(rows[0]))]
    total = sum(widths) + 3 * len(widths) - 1
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    print(sep)
    for row in rows:
        print('|', end='')
        for width, cell in zip(widths, row):
            print(cell.center(width + 2), end='|')
        print()
        print(sep)
    print()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return


if __name__ == '__main__':
    import random
    print_table(
        epoch=3,
        recalls_dict={
            "train": {
                "R@1,IoU=0.7": random.random(),
                "R@5,IoU=0.7": random.random(),
            },
            "test": {
                "R@1,IoU=0.7": random.random(),
                "R@5,IoU=0.7": random.random(),
            },
        }
    )
