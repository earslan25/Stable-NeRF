import itertools
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from nerf.network import NeRFNetwork
from stable_diffusion.network import SDNetwork
from datasets.dataset import StableNeRFDataset, collate_fn
from utils.graphics_utils import *
from utils.loss_utils import *
from utils.system_utils import get_memory_usage
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration


def training(data_args, model_args, opt_args):
    output_dir = Path("output")
    logging_dir = Path("output", "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        # mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    device = accelerator.device
    print(device)
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    channel_dim = 3  # SD config supports 3 latent channels!! but we also have 4 channel support nerf
    cat_cam = True # can be 0 to not concat cam to latent
    print(f"initiating sd network with channel_dim={channel_dim} and cat_cam={cat_cam}")
    sd = SDNetwork(pretrained_models_path, image_encoder_path, channel_dim=channel_dim, cat_cam=cat_cam)
    print("sd network initiated and moved to device")
    print(f"initiating nerf network with channel_dim={channel_dim}")
    nerf = NeRFNetwork(channel_dim=channel_dim)
    nerf.train()
    print("nerf network initiated and moved to device")
    bg_color = torch.ones(channel_dim, device=device)
    max_steps = 4  # reduced from 1024 for memory reasons during testing

    # hardcoded, could not find where to get these from
    encoder_input_dim = 512  
    encoder_output_dim = 64  
    clip_text_output_dim = 768 # 224

    dataset_name = 'nerf'
    batch_size = 2
    cache_cuda = True
    print(f"initiating dataset {dataset_name} with batch_size={batch_size}, encoder_input_dim={encoder_input_dim}, encoder_output_dim={encoder_output_dim}")
    dataset = StableNeRFDataset(dataset_name, shape=encoder_input_dim, encoded_shape=encoder_output_dim, cache_cuda=cache_cuda)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print("dataset and dataloader initiated")

    epochs = 1
    lr = 1e-4
    weight_decay = 1e-4
    params_to_opt = itertools.chain(sd.image_proj_model.parameters(),  sd.adapter_modules.parameters(), nerf.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)

    sd, nerf, optimizer, dataloader = accelerator.prepare(sd, nerf, optimizer, dataloader)
    print("accelerator prepared")

    nerf.mark_untrained_grid(dataset.reference_poses, dataset.intrinsic)

    # note: could be optimized by batching ref/target images together
    for epoch in tqdm(range(epochs)):
        nerf.update_extra_state()
        for i, batch in enumerate(dataloader):
            with accelerator.accumulate(sd, nerf):
                print(f"batch {i} of epoch {epoch}")
                target_image = batch['target_image']
                reference_image = batch['reference_image']

                with torch.no_grad():
                    target_image = sd.encode_images(target_image)  # latent space
                    reference_image = sd.encode_images(reference_image)  # latent space

                # allocate if needed
                if cat_cam:
                    target_pose = batch['target_pose'].view(batch_size, -1)
                    reference_pose = batch['reference_pose'].view(batch_size, -1)

                target_rays_o = batch['target_rays_o']
                target_rays_d = batch['target_rays_d']
                target_rays_inds = batch['target_rays_inds']

                reference_rays_o = batch['reference_rays_o']
                reference_rays_d = batch['reference_rays_d']
                reference_rays_inds = batch['reference_rays_inds']

                target_image_gt = torch.gather(
                    target_image.view(batch_size, -1, channel_dim), 1, torch.stack(channel_dim * [target_rays_inds], -1)
                )
                reference_image_gt = torch.gather(
                    reference_image.view(batch_size, -1, channel_dim), 1, torch.stack(channel_dim * [reference_rays_inds], -1)
                )

                # novel for conditioning and ip adaptor
                pred_target_latent = nerf.render(target_rays_o, target_rays_d, bg_color=bg_color, max_steps=max_steps)['image']

                # reference for reconstruction loss for nerf
                pred_reference_latent = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=max_steps)['image']

                # nerf loss on target and reference for reconstruction
                nerf_loss = l1_loss(pred_target_latent, target_image_gt) + l1_loss(pred_reference_latent, reference_image_gt)
                pred_target_latent = pred_target_latent.view(batch_size, channel_dim, encoder_output_dim, encoder_output_dim)
                if channel_dim == 3:
                    nerf_ssim = ssim(pred_target_latent, 
                        target_image_gt.view(batch_size, channel_dim, encoder_output_dim, encoder_output_dim)) + \
                        ssim(pred_reference_latent.view(batch_size, channel_dim, encoder_output_dim, encoder_output_dim), 
                        reference_image_gt.view(batch_size, channel_dim, encoder_output_dim, encoder_output_dim))
                    nerf_loss = nerf_loss * 0.85 + nerf_ssim * 0.15

                # concat cam params to latents after flattening
                # reference and target latents put together, or run through clip
                # not using nerf output for reference latent
                if channel_dim == 4:
                    target_latent_cat_cam = pred_target_latent.view(batch_size, -1)
                    reference_latent_cat_cam = reference_image.view(batch_size, -1)
                    if cat_cam:
                        target_latent_cat_cam = torch.cat([target_latent_cat_cam, target_pose], dim=-1)
                        reference_latent_cat_cam = torch.cat([reference_latent_cat_cam, reference_pose], dim=-1)
                    image_embeds = torch.cat([target_latent_cat_cam, reference_latent_cat_cam], dim=0)
                else:
                    # resizing (upscaled from 64x64 to clip's 224x224) is inside method, but could be moved somewhere else
                    # OR concat images like they are one image
                    # 2x64x64 -> 1x64x128 -> 1x224x224
                    with torch.no_grad():
                        cond_image = sd.clip_encode_images(torch.cat([reference_image, target_image], dim=2))
                        # reference_embed = sd.clip_encode_images(reference_image)  
                        # target_embed = sd.clip_encode_images(pred_target_latent)
                    if cat_cam:
                        cond_image = torch.cat([cond_image, reference_pose, target_pose], dim=-1)  # 768 -> [...768 ... 16 ... 16]
                        # reference_embed = torch.cat([reference_embed, reference_pose], dim=-1)
                        # target_embed = torch.cat([target_embed, target_pose], dim=-1)
                    image_embeds = cond_image
                    # image_embeds = torch.cat([target_embed, reference_embed], dim=0)

                # input to unet
                noise = torch.randn_like(reference_image)

                timesteps = torch.randint(0, sd.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = sd.noise_scheduler.add_noise(reference_image, noise, timesteps)

                # unet_added_cond_kwargs = {"text_embeds": sd.pooled_empty_text_embeds}  # not sure what this is
                dummy_text_embeds = torch.zeros(batch_size, sd.num_tokens, clip_text_output_dim, device=device)
                noise_pred = sd(noisy_latents, timesteps, dummy_text_embeds, image_embeds)

                # sd loss on denoising latent reference + noise
                sd_loss = mse_loss(noise_pred.float(), noise.float())
                
                total_loss = sd_loss + nerf_loss
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()


if __name__ == "__main__":
    # TODO
    # resolve hacks (fit latents to clip/image projector dims) -> chia
    # validate loaded weights when channel_dim=3 -> chia
    # argparser -> daniel
    # distributed training setup -> emre
    # do training -> chia
    # prep dataset for objaverse -> emre
    # visualize/plot results -> daniel

    arg_1, arg_2, arg_3 = None, None, None
    training(arg_1, arg_2, arg_3)