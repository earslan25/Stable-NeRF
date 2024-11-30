import itertools
import torch
from nerf.network import NeRFNetwork
from stable_diffusion.network import SDNetwork
from datasets.dataset import StableNeRFDataset, collate_fn
from utils.graphics_utils import *
from utils.loss_utils import *


def training():
    device = torch.device("cuda")
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    channel_dim = 4  # SD config supports 3 latent channels!! but we also have 4 channel support nerf
    sd = SDNetwork(pretrained_models_path, image_encoder_path, channel_dim=3).train().to(device)
    nerf = NeRFNetwork(channel_dim=channel_dim).train().to(device)
    bg_color = torch.ones(channel_dim, device=device)

    encoder_input_dim = sd.vae.config.block_out_channels[-1]
    encoder_output_dim = sd.vae.config.block_out_channels[0]

    dataset_name = 'nerf'
    batch_size = 4
    dataset = StableNeRFDataset(dataset_name, shape=encoder_input_dim, encoded_shape=encoder_output_dim)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    epochs = 10
    lr = 1e-4
    weight_decay = 1e-4
    params_to_opt = itertools.chain(sd.image_proj_model.parameters(),  sd.adapter_modules.parameters(), nerf.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        for batch in dataloader:
            target_image = batch['target_image'].to(device)
            reference_image = batch['reference_image'].to(device)

            with torch.no_grad():
                target_image = sd.encode_images(target_image)  # latent space
                reference_image = sd.encode_images(reference_image)  # latent space

            target_pose = batch['target_pose'].to(device)
            reference_pose = batch['reference_pose'].to(device)

            target_rays_o = batch['target_rays_o'].to(device)
            target_rays_d = batch['target_rays_d'].to(device)
            target_rays_inds = batch['target_rays_inds'].to(device)

            reference_rays_o = batch['reference_rays_o'].to(device)
            reference_rays_d = batch['reference_rays_d'].to(device)
            reference_rays_inds = batch['reference_rays_inds'].to(device)

            target_image_gt = torch.gather(
                target_image.view(batch_size, -1, channel_dim), 1, torch.stack(channel_dim * [target_rays_inds], -1)
            )
            reference_image_gt = torch.gather(
                reference_image.view(batch_size, -1, channel_dim), 1, torch.stack(channel_dim * [reference_rays_inds], -1)
            )

            # novel for conditioning and ip adaptor
            pred_target_latent = nerf.render(target_rays_o, target_rays_d, bg_color=bg_color)['image']

            # reference for reconstruction loss for nerf
            pred_reference_latent = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color)['image']

            # nerf loss on target and reference for reconstruction
            # ...

            # concat cam params to latents after flattening, maybe not needed since nerf already has cam params
            target_latent_cat_cam = torch.cat(
                [pred_target_latent.view(batch_size, -1), target_pose.view(batch_size, -1)], dim=-1
            )
            # not using nerf output for reference latent
            reference_latent_cat_cam = torch.cat(
                [reference_image.view(batch_size, -1), reference_pose.view(batch_size, -1)], dim=-1
            )
            # reference and target latents put together, or run through clip
            # in any case, either the inputs or ip adaptor network input should be the adjusted
            image_embeds = None  

            # input to unet
            noise = torch.randn_like(reference_image)

            timesteps = torch.randint(0, sd.noise_scheduler.num_train_timesteps, (batch_size,), device=device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = sd.noise_scheduler.add_noise(reference_image, noise, timesteps)

            unet_added_cond_kwargs = {"text_embeds": sd.pooled_empty_text_embeds}  # not sure what this is
            noise_pred = sd(noisy_latents, timesteps, sd.empty_text_embeds, unet_added_cond_kwargs, image_embeds)

            # sd loss on denoising latent reference + noise
            # ...


if __name__ == "__main__":
    training()