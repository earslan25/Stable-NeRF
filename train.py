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

# TODO: be able to mess with these dimensions... 

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

print("completed training preparation")

# training
print("starting training")
for epoch in tqdm(range(epochs)):

    nerf.update_extra_state()
    for i, batch in enumerate(dataloader):
        
        
        # start accumulation
        with accelerator.accumulate(sd, nerf):
            target_image = batch['target_image']
            reference_image = batch['reference_image']

            # image encoding 
            with torch.no_grad():
                target_image = sd.encode_images(target_image)  # latent space
                reference_image = sd.encode_images(reference_image)  # latent space


            # -- nerf with ip adapter --

            # target (novel view for conditioning and ip adaptor)
                # NOTE: what does this do at a high level in our pipeline...

            target_pose = batch['target_pose'].to(device).view(batch_size, -1)
            target_rays_o, target_rays_d, target_rays_inds = batch['target_rays_o'].to(device), batch['target_rays_d'].to(device), batch['target_rays_inds'].to(device)
            target_image_gt = torch.gather(target_image.view(batch_size, -1, sd.channel_dim), 1, torch.stack(sd.channel_dim * [target_rays_inds], -1))
            pred_target_latent = nerf.render(target_rays_o, target_rays_d, bg_color=bg_color, max_steps=max_steps)['image']
            nerf_loss_0 = l1_loss(pred_target_latent, target_image_gt)

            # reshpae target view latent
            pred_target_latent = pred_target_latent.view(batch_size, sd.channel_dim, encoder_output_dim, encoder_output_dim)

            # reference (reference view for reconstruction loss for nerf)
                # NOTE: same question as above
            reference_pose = batch['reference_pose'].to(device).view(batch_size, -1)
            reference_rays_o, reference_rays_d, reference_rays_inds = batch['reference_rays_o'].to(device), batch['reference_rays_d'].to(device), batch['reference_rays_inds'].to(device)
            reference_image_gt = torch.gather(reference_image.view(batch_size, -1, sd.channel_dim), 1, torch.stack(sd.channel_dim * [reference_rays_inds], -1))
            pred_reference_latent = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=max_steps)['image']
            nerf_loss_1 = l1_loss(pred_reference_latent, reference_image_gt)

            # combine losses 
            nerf_loss = nerf_loss_0 + nerf_loss_1

            # TODO: visualizations here?

            print("target latent shape: ", pred_target_latent.shape) # torch.Size([2, 4, 64, 64])
            plt.imshow(pred_target_latent[0][0].detach().cpu().numpy(), cmap='gray')
            plt.title("Single Channel Image")
            plt.axis('off')

            plt.savefig(os.getcwd() + "cache/pred_target_latent_channel_0.png")

            



            # clean unneeded variables to free memory
            del target_rays_o, target_rays_d, target_rays_inds, target_image_gt
            del reference_rays_o, reference_rays_d, reference_rays_inds, reference_image_gt
            torch.cuda.empty_cache()
            gc.collect()

            # NOTE: garbage collection calls might be causing memory problems?

            # -- stable diffusion --

            # sd inputs
            target_latent_cat_cam = pred_target_latent.view(batch_size, -1)
            target_latent_cat_cam = torch.cat([target_latent_cat_cam, target_pose], dim=-1).view(batch_size, 1, -1)
            reference_latent_cat_cam = reference_image.view(batch_size, -1)
            reference_latent_cat_cam = torch.cat([reference_latent_cat_cam, reference_pose], dim=-1).view(batch_size, 1, -1)

            image_embeds = torch.cat([target_latent_cat_cam, reference_latent_cat_cam], dim=1)

            # unet inputs
            noise = torch.randn_like(reference_image)
            timesteps = torch.randint(0, sd.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

            # forward diffusion
                # NOTE: why is this needed? overall pipeline...
            noisy_latents = sd.noise_scheduler.add_noise(reference_image, noise, timesteps)
            add_time_ids = [
                torch.tensor([[encoder_input_dim, encoder_input_dim]]).to(device),
                torch.tensor([[0, 0]]).to(device),
                torch.tensor([[encoder_input_dim, encoder_input_dim]]).to(device),
            ]
            add_time_ids = torch.cat(add_time_ids, dim=1).to(device).repeat(batch_size,1)
            added_cond_kwargs = {"text_embeds":sd.pooled_empty_text_embeds.repeat(batch_size,1).to(device), "time_ids":add_time_ids}
            noise_pred = sd(noisy_latents, timesteps, added_cond_kwargs, image_embeds)

            sd_loss = mse_loss(noise_pred.float(), noise.float())

            # clean unneed variables to free memory
            del target_pose, reference_pose, target_latent_cat_cam, reference_latent_cat_cam
            del image_embeds, timesteps, noisy_latents, add_time_ids, added_cond_kwargs
            torch.cuda.empty_cache()
            
            # backprop losses
            accelerator.backward(sd_loss + nerf_loss)
            optimizer.step()
            optimizer.zero_grad()

            # clean unneeded variables to free memory
            del noise, noise_pred
            del sd_loss, nerf_loss
            torch.cuda.empty_cache()
            gc.collect()


            # -- save model checkpoints for caching and inference -- 

            # TODO: ... 

        # for now, stops memory issues
        break
