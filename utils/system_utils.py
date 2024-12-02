import torch


def get_memory_usage(device='cuda:0'):
    free, total = torch.cuda.mem_get_info(device)
    mem_used_GB = (total - free) / 1024 ** 3
    
    return mem_used_GB