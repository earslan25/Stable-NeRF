
# TODO: run a inference

# input image
# reference pose
# target pose

# run through sd encoder
# generate with trained nerf
# finish stable diffusion calls

# return final image












import torch
import itertools
from tqdm import tqdm
import gc
import os
import matplotlib.pyplot as plt

from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from stable_diffusion.network import SDNetwork
from nerf.network import NeRFNetwork

from datasets.dataset import StableNeRFDataset, collate_fn
from utils.graphics_utils import *
from utils.loss_utils import *


# TODO:
    # 1. (DONE) figure out the memory issues
    # 2. (DONE) clean up the code

    # 3. double check all the logic here 
    # 4. start adding visualizations


# accelerator 
output_dir = Path("output")
logging_dir = Path("output", "logs")
accelerator = Accelerator(
    mixed_precision="fp16", # decreased numeric accuracy for memory performance
    project_config=ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
)
device = accelerator.device

print("accelerator device: ", device)

# instantiate stable diffusion and nerf
sd = SDNetwork('stabilityai/stable-diffusion-xl-base-1.0', 'openai/clip-vit-large-patch14', cat_cam=True).to(device)
nerf = NeRFNetwork(channel_dim=sd.channel_dim).to(device)
nerf.train()

# torch script for optimization, not quite working
# sd = torch.jit.script(sd) # failed optimization
# nerf = torch.jit.script(nerf)

print("completed model instantiation")

# variables
bg_color = torch.ones(sd.channel_dim, device=device)
max_steps = 1 # originally 1024

encoder_input_dim = 512
encoder_output_dim = 64

# TODO: 
    # it may be that we can't?
    # or at least without some more work, it's hard to tell where it's going wrong

# dataset
batch_size = 2
dataset = StableNeRFDataset('nerf', shape=encoder_input_dim, encoded_shape=encoder_output_dim, cache_cuda=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print("completed dataset loading")

# training variables
epochs = 1
lr = 1e-4
weight_decay = 1e-4
params_to_opt = itertools.chain(sd.image_proj_model.parameters(),  sd.adapter_modules.parameters(), nerf.parameters())
optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)

sd, nerf, optimizer, dataloader = accelerator.prepare(sd, nerf, optimizer, dataloader)
nerf.mark_untrained_grid(dataset.reference_poses, dataset.intrinsic)





sd.to("cpu")

noise_pred = torch.load(os.getcwd() + f"/cache/noise_pred.pt").to("cpu")

decoded_noise_pred = sd.decode_latents(noise_pred)

print(decoded_noise_pred.shape)

plt.figure()
plt.imshow(decoded_noise_pred[0].detach().numpy())
plt.title("Decoded noise pred")
plt.savefig(os.getcwd() + f"/cache/decoded_noise_pred.png")