import os
import torch
import random


def sample_save_for_vis(prefix: str, tensor: torch.Tensor, sample_prob: float = 0.125) -> None:
    """
    utility function used to save intermediate tensors for visualizations
    """

    # user warning
    if "_" in prefix: print("Warning: when using save_vis_image, do not use underscores in the prefix!")

    # rather than saving on a count, we randomly sample
    if random.random() > sample_prob: return

    # ensure vis directory exists
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")

    # find existing names and iterate name count by 1
    max_file_count = 0
    for file_name in os.listdir("visualizations"):
        index = file_name.find("_")
        if file_name[:index] != prefix: continue
        index += 1

        try:
            max_file_count = max(int(file_name[index:index+4]), max_file_count)
        except:
            print("There was an error visualization saving!")

    # save tensor
    torch.save(tensor, f"visualizations/{prefix}_{max_file_count+1:04d}.pt")