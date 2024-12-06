import itertools
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nerf.network import NeRFNetwork
from stable_diffusion.network import SDNetwork
from datasets.dataset import StableNeRFDataset, collate_fn
from utils.graphics_utils import *
from utils.loss_utils import *
from utils.system_utils import get_memory_usage

# TODO
# argparser -> daniel
# distributed training setup -> emre
# prep dataset for objaverse -> emre
# visualize/plot results -> daniel


def training(data_args, model_args, opt_args):
    device = torch.device("cuda")
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    # channel_dim = 3  # SD config supports 3 latent channels!! but we also have 4 channel support nerf
    cat_cam = True # can be 0 to not concat cam to latent
    # print(f"initiating sd network with channel_dim={channel_dim} and cat_cam={cat_cam}")
    print(f"initiating sd network with cat_cam={cat_cam}")
    sd = SDNetwork(pretrained_models_path, image_encoder_path, cat_cam=cat_cam).to(device)
    print("sd network initiated and moved to device")
    print(f"initiating nerf network with channel_dim={sd.channel_dim}")
    nerf = NeRFNetwork(channel_dim=sd.channel_dim).to(device)
    nerf.train()
    print("nerf network initiated and moved to device")




    bg_color = torch.ones(sd.channel_dim, device=device)
    max_steps = 4  # reduced from 1024 for memory reasons during testing

    # hardcoded, could not find where to get these from
    # [CH] default latent dimension encoded by sd's vae is: 4x(H/8)x(W/8)
    encoder_input_dim = 512  
    encoder_output_dim = 64  
    clip_text_output_dim = 768

    dataset_name = 'nerf'
    batch_size = 10
    print(f"initiating dataset {dataset_name} with batch_size={batch_size}, encoder_input_dim={encoder_input_dim}, encoder_output_dim={encoder_output_dim}")
    dataset = StableNeRFDataset(dataset_name, shape=encoder_input_dim, encoded_shape=encoder_output_dim)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print("dataset and dataloader initiated")

    nerf.mark_untrained_grid(dataset.reference_poses, dataset.intrinsic)

    epochs = 1
    lr = 1e-4
    weight_decay = 1e-4
    params_to_opt = itertools.chain(sd.image_proj_model.parameters(),  sd.adapter_modules.parameters(), nerf.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)

    # note: could be optimized by batching ref/target images together
    for epoch in tqdm(range(epochs)):
        nerf.update_extra_state()
        for i, batch in enumerate(dataloader):
            print(f"batch {i} of epoch {epoch}")
            target_image = batch['target_image'].to(device)
            reference_image = batch['reference_image'].to(device)

            with torch.no_grad():
                target_image = sd.encode_images(target_image)  # latent space
                reference_image = sd.encode_images(reference_image)  # latent space

            # allocate if needed
            if cat_cam:
                target_pose = batch['target_pose'].to(device).view(batch_size, -1)
                reference_pose = batch['reference_pose'].to(device).view(batch_size, -1)

            target_rays_o = batch['target_rays_o'].to(device)
            target_rays_d = batch['target_rays_d'].to(device)
            target_rays_inds = batch['target_rays_inds'].to(device)

            reference_rays_o = batch['reference_rays_o'].to(device)
            reference_rays_d = batch['reference_rays_d'].to(device)
            reference_rays_inds = batch['reference_rays_inds'].to(device)

            target_image_gt = torch.gather(
                target_image.view(batch_size, -1, sd.channel_dim), 1, torch.stack(sd.channel_dim * [target_rays_inds], -1)
            )
            reference_image_gt = torch.gather(
                reference_image.view(batch_size, -1, sd.channel_dim), 1, torch.stack(sd.channel_dim * [reference_rays_inds], -1)
            )

            # novel for conditioning and ip adaptor
            pred_target_latent = nerf.render(target_rays_o, target_rays_d, bg_color=bg_color, max_steps=max_steps)['image']

            # reference for reconstruction loss for nerf
            pred_reference_latent = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=max_steps)['image']

            # nerf loss on target and reference for reconstruction
            nerf_loss = l1_loss(pred_target_latent, target_image_gt) + l1_loss(pred_reference_latent, reference_image_gt)
            pred_target_latent = pred_target_latent.view(batch_size, sd.channel_dim, encoder_output_dim, encoder_output_dim)
            #if channel_dim == 3:
            #    nerf_ssim = ssim(pred_target_latent, 
            #        target_image_gt.view(batch_size, channel_dim, encoder_output_dim, encoder_output_dim)) + \
            #        ssim(pred_reference_latent.view(batch_size, channel_dim, encoder_output_dim, encoder_output_dim), 
            #        reference_image_gt.view(batch_size, channel_dim, encoder_output_dim, encoder_output_dim))
            #    nerf_loss = nerf_loss * 0.85 + nerf_ssim * 0.15

            # concat cam params to latents after flattening, maybe not needed since nerf already has cam params
            # reference and target latents put together, or run through clip
            # not using nerf output for reference latent
            #if channel_dim == 4:
            #    target_latent_cat_cam = pred_target_latent.view(batch_size, -1)
            #    reference_latent_cat_cam = reference_image.view(batch_size, -1)
            #    if cat_cam:
            #        target_latent_cat_cam = torch.cat([target_latent_cat_cam, target_pose], dim=-1)
            #        reference_latent_cat_cam = torch.cat([reference_latent_cat_cam, reference_pose], dim=-1)
            #    image_embeds = torch.cat([target_latent_cat_cam, reference_latent_cat_cam], dim=0)
            #else:
                # resizing (upscaled from 128x128 to clip's 224x224) is inside method, but could be moved somewhere else
                # OR concat images like they are one image
            #    with torch.no_grad():
            #        cond_image = sd.clip_encode_images(torch.cat([reference_image, target_image], dim=2))
                    # reference_embed = sd.clip_encode_images(reference_image)  
                    # target_embed = sd.clip_encode_images(pred_target_latent)
            #    if cat_cam:
            #        cond_image = torch.cat([cond_image, reference_pose, target_pose], dim=-1)
                    # reference_embed = torch.cat([reference_embed, reference_pose], dim=-1)
                    # target_embed = torch.cat([target_embed, target_pose], dim=-1)
            #    image_embeds = cond_image
                # image_embeds = torch.cat([target_embed, reference_embed], dim=0)
            
            target_latent_cat_cam = pred_target_latent.view(batch_size, -1)
            reference_latent_cat_cam = reference_image.view(batch_size, -1)
            if cat_cam:
                target_latent_cat_cam = torch.cat([target_latent_cat_cam, target_pose], dim=-1).view(batch_size, 1, -1)
                reference_latent_cat_cam = torch.cat([reference_latent_cat_cam, reference_pose], dim=-1).view(batch_size, 1, -1)
            image_embeds = torch.cat([target_latent_cat_cam, reference_latent_cat_cam], dim=1)

            # input to unet
            noise = torch.randn_like(reference_image)

            timesteps = torch.randint(0, sd.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = sd.noise_scheduler.add_noise(reference_image, noise, timesteps)

            # dummy_text_embeds = torch.zeros(batch_size, sd.num_tokens, clip_text_output_dim, device=device)
            add_time_ids = [
                torch.tensor([[512, 512]]).to(device),
                torch.tensor([[0, 0]]).to(device),
                torch.tensor([[512, 512]]).to(device),
            ]
            add_time_ids = torch.cat(add_time_ids, dim=1).to(device).repeat(batch_size,1)
            added_cond_kwargs = {"text_embeds":sd.pooled_empty_text_embeds.repeat(batch_size,1).to(device), "time_ids":add_time_ids}
            noise_pred = sd(noisy_latents, timesteps, added_cond_kwargs, image_embeds)

            # sd loss on denoising latent reference + noise
            sd_loss = mse_loss(noise_pred.float(), noise.float())
            print(nerf_loss, sd_loss)
            
            total_loss = sd_loss + nerf_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    arg_1, arg_2, arg_3 = None, None, None
    training(arg_1, arg_2, arg_3)