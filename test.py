
# stable diffusion-nerf pipeline

# NOTE
    # encode using stable diffusion encoder
    # train on the nerf
    # run through the u-net
        # somehow incorporate pose encodings
    # decode

    # get this to work any way possible

# NOTE
    # end goals
        # get a 3d visualization of the encoded nerf

# NOTE
    # intermediate steps

    # get stable diffusion to generate a simple image
    # train a simple nerf to give a novel view



# # NOTE: get stable diffusion to generate a simple image

# import torch
# from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
# from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
# from torchvision.transforms.functional import to_pil_image
# from utils.sd import encode_prompt

# def test_stable_diffusion():
#     """
#     Generate a simple image.
#     """

#     # initialize models
#     model_id = "stabilityai/stable-diffusion-xl-base-1.0"
#     scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     print("using device: ", device)

#     vae = AutoencoderKL.from_pretrained(
#         model_id, subfolder="vae", torch_dtype=torch.float32
#     ).to(device)
#     unet = UNet2DConditionModel.from_pretrained(
#         model_id, subfolder="unet", torch_dtype=torch.float32
#     ).to(device)

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_id, subfolder="tokenizer", use_fast=False
#     )
#     tokenizer_2 = AutoTokenizer.from_pretrained(
#         model_id, subfolder="tokenizer_2", use_fast=False
#     )
#     text_encoder = CLIPTextModel.from_pretrained(
#         model_id, subfolder="text_encoder", torch_dtype=torch.float32
#     ).to(device)
#     text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
#         model_id, subfolder="text_encoder_2", torch_dtype=torch.float32
#     ).to(device)









#     prompt = "A futuristic cityscape at sunset, highly detailed, vibrant colors"
#     prompt_2 = "Cyberpunk style, glowing neon lights, wide-angle perspective"
#     negative_prompt = "tall buildings"
#     negative_prompt_2 = "rainy day"







#     # [CH] Encode text prompts
#     ( prompt_embeds, 
#       negative_prompt_embeds, 
#       pooled_prompt_embeds, 
#       negative_pooled_prompt_embeds
#     ) = encode_prompt(prompt=prompt,
#                       prompt_2=prompt_2,
#                       device=device,
#                       negative_prompt=negative_prompt,
#                       negative_prompt_2=negative_prompt_2,
#                       tokenizer = tokenizer,
#                       tokenizer_2 = tokenizer_2,
#                       text_encoder = text_encoder,
#                       text_encoder_2 = text_encoder_2,
#                       )

#     add_text_embeds = pooled_prompt_embeds

#     # logging
#     print("encoded prompts")

#     ''' 
#     [CH] This part is a mystery... mystery start
#     '''
#     resolution = 256
#     crops_coords_top_left = (0,0)
#     original_sizes = (resolution, resolution)
#     crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)
#     original_sizes = torch.tensor(original_sizes, dtype=torch.long)
#     crops_coords_top_left = crops_coords_top_left.repeat(len(prompt_embeds), 1)
#     original_sizes = original_sizes.repeat(len(prompt_embeds), 1)

#     target_size = (resolution, resolution)
#     add_time_ids = list(target_size)
#     add_time_ids = torch.tensor([add_time_ids])
#     add_time_ids = add_time_ids.repeat(len(prompt_embeds), 1)
#     add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
#     add_time_ids = add_time_ids.to(device, dtype=torch.float32)
#     # [CH] print("add_time_ids",add_time_ids) # [[1024., 1024.,    0.,    0., 1024., 1024.]]
#     negative_add_time_ids = add_time_ids
#     '''
#     [CH] Mystery end
#     '''

#     # do_classifier_free_guidance:
#     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#     add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
#     add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    
#     prompt_embeds = prompt_embeds.to(device, dtype=torch.float32)
#     add_text_embeds = add_text_embeds.to(device)
#     added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

#     # Initialize latent variables
#     latent_shape = (
#         1,
#         unet.in_channels,
#         unet.sample_size,
#         unet.sample_size,
#     )
#     latents = torch.randn(latent_shape, device=device, dtype=torch.float32)








#     # denoise image
#     num_steps = 10
#     guidance_scale = 10.0
#     scheduler.set_timesteps(num_steps)
#     timesteps = scheduler.timesteps.to(device)
#     for t in timesteps:
#         with torch.no_grad():
#             print("starting denoise step") # logging
#             latents_model_input = torch.cat([latents] * 2)

#             print("starting unet") # logging 
#             noise_pred = unet(
#                 latents_model_input, 
#                 t, 
#                 timestep_cond=None,
#                 encoder_hidden_states=prompt_embeds,
#                 added_cond_kwargs=added_cond_kwargs,
#             ).sample
#             print("completed unet") # logging

#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#         latents = scheduler.step(noise_pred, t, latents).prev_sample
#     latents = latents.float()

#     # Decode latents into an image
#     scaling_factor = vae.config.scaling_factor
#     with torch.no_grad():
#         # Scaling factor for unit variance
#         latents = latents / scaling_factor 
#         image_tensor = vae.decode(latents).sample[0]

#     # Convert to PIL image and save
#     image = to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
#     image.save("cache/sd_test_0.png")

# if __name__ == "__main__":
#     test_stable_diffusion()















# NOTE: figure out basic nerf 

# just get it to run and train on something basic?
# is it even necessary?

# not too sure... s

import gc
import torch
import matplotlib.pyplot as plt
from nerf.network import NeRFNetwork
from tqdm import tqdm
from utils.graphics_utils import *
from utils.loss_utils import *
from datasets.dataset import StableNeRFDataset, collate_fn
from torchviz import make_dot

def train_nerf():
    torch.autograd.set_detect_anomaly(True)

    device = 'cuda'
    nerf = NeRFNetwork().to(device)
    nerf.train()

    H, W = 128, 128
    name = 'nerf'
    # name = 'objaverse'
    dataset = StableNeRFDataset(dataset_name=name, shape=(H, W), encoded_shape=(H, W), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], generate_cuda_ray=True, percent_objects=0.0001)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(nerf.get_params(1e-2), betas=(0.9, 0.99), eps=1e-15)

    bg_color = 0
    epochs = 50

    nerf.mark_untrained_grid(dataset.reference_poses, dataset.intrinsic)

    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        nerf.update_extra_state()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            if name == 'objaverse' and i > 0:
                break
            reference_rays_o = batch['reference_rays_o'].to(device)
            reference_rays_d = batch['reference_rays_d'].to(device)
            reference_image = batch['reference_image'].to(device)
            curr_batch_size = reference_image.shape[0]

            # print(curr_batch_size)

            reference_image_gt = reference_image.permute(0, 2, 3, 1).view(curr_batch_size, -1, 3)
            # reference_image_gt = reference_image.view(curr_batch_size, -1, 3)
            pred = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=300)['image']

            # save reference_image_gt and pred to /debug_out
            if name == 'objaverse' and i == 0 or name == 'nerf' and (i + 1) % 101:
                with torch.no_grad():
                    # print(pred.mean(dim=1))
                    plt.imsave(f"cache/nerf/reference_image_gt_{i}.png", (reference_image_gt[0].detach().view(H, W, 3)).cpu().numpy())
                    # plt.imsave(f"debug_out/reference_image_{i}.png", (reference_image[0].detach().permute(1, 2, 0)).cpu().numpy())
                    plt.imsave(f"cache/nerf/pred_{i}.png", pred[0].detach().view(H, W, 3).cpu().numpy())

            loss = l1_loss(pred, reference_image_gt) # + 0.2 * ssim(pred.permute(0, 2, 1).view(curr_batch_size, 3, H, W), reference_image_gt.permute(0, 2, 1).view(curr_batch_size, 3, H, W))
            # make_dot(loss, params=dict(nerf.named_parameters())).render("debug_out/computation_graph", format="png")
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

        total_loss /= len(dataloader)
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")


if __name__ == "__main__":
    # test_nerf()  
    # test_multi_channel_nerf() 
    train_nerf()

