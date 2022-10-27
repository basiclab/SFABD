from typing import List

import torch
import torch.distributed as dist


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


def barrier():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if dist.get_world_size() == 1:
        return
    dist.barrier()


def is_main():
    return get_rank() == 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Run gather on arbitrary length tensors
    Args:
        tensor: tensor of shape (N, ...)
    Returns:
        tensor_list: list of tensor gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    device = get_device()

    # obtain Tensor size of each rank
    N, *shape = tensor.shape
    N = torch.LongTensor([N]).to(device)
    Ns = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(Ns, N)
    Ns = torch.cat(Ns, dim=0)
    max_N = Ns.max().cpu().item()

    # create same sized tensors on all ranks
    tensor_list = []
    for _ in range(world_size):
        tensor_list.append(tensor.new_empty(max_N, *shape))
    # padding if necessary
    pad = [0] * (len(shape) * 2) + [0, max_N - N]
    tensor = torch.nn.functional.pad(tensor, pad)
    dist.all_gather(tensor_list, tensor)

    tensor_list = [tensor[: N] for N, tensor in zip(Ns, tensor_list)]

    return tensor_list


def gather_dict(data_dict, to_cpu=True):
    data = {}
    for key, value in data_dict.items():
        data[key] = torch.cat(gather(value), dim=0)
        if to_cpu:
            data[key] = data[key].cpu()
    return data
