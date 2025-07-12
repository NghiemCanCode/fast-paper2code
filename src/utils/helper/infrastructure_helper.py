import torch
from typing import Optional


def prepare_device(n_gpu_use) -> Optional[list]:
    """
    Setup GPU device if available. Get gpu device indices which are used for DataParallel
    :param n_gpu_use:
    :return: An id or list of GPU's ids to use.
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu

    list_ids = list(range(n_gpu_use))
    return list_ids