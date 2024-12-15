

# import torch
# import matplotlib.pyplot as plt
# from notes_1 import latent_to_image

# with torch.no_grad():



#     device = "mps"

#     choice = 0

#     pred = torch.load(f"visualizations/notes_3/pred_{choice:04d}.pt", map_location=torch.device(device))

#     print(torch.max(pred))
#     print(torch.min(pred))

#     pred = pred.view(1, 64, 64, 4).permute(0, 3, 1, 2)

#     pred -= 0.45
#     pred *= 4

#     print(torch.max(pred))
#     print(torch.min(pred))

#     latent = latent_to_image(pred, 1, 64, 64)
#     plt.imsave(f"test.png", latent)





# print("DERPP")



# import torch
# from PIL import Image
# import numpy as np
# from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler


# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
# # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# # print("using device: ", device)

# device = "cpu"

# vae = AutoencoderKL.from_pretrained(
#     model_id, subfolder="vae", torch_dtype=torch.float32
# ).to(device)

# image = Image.open(f"visualizations/notes_3/reference_image_0000.png")
# image = torch.tensor(np.array(image.convert("RGB")) / 255., dtype=torch.float32)

# print(image.shape)

# image = image.permute(2, 1, 0)[None,:]

# print(image.shape)

# encoding = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

# print(encoding.shape)

# print(torch.max(encoding))
# print(torch.min(encoding))








# NOTE
    # run the unet on the prediction
    # the should be it though...

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from torchvision.transforms.functional import to_pil_image
from utils.sd import encode_prompt
from PIL import Image
import numpy as np

def test_stable_diffusion():
    """
    Generate a simple image.
    """

    # initialize models
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("using device: ", device)

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float32
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch.float32
    ).to(device)









    prompt = "Yellow lego tractor, black background"
    prompt_2 = "Yellow lego tractor, black background"
    negative_prompt = "flat"
    negative_prompt_2 = "low quality"







    # [CH] Encode text prompts
    ( prompt_embeds, 
      negative_prompt_embeds, 
      pooled_prompt_embeds, 
      negative_pooled_prompt_embeds
    ) = encode_prompt(prompt=prompt,
                      prompt_2=prompt_2,
                      device=device,
                      negative_prompt=negative_prompt,
                      negative_prompt_2=negative_prompt_2,
                      tokenizer = tokenizer,
                      tokenizer_2 = tokenizer_2,
                      text_encoder = text_encoder,
                      text_encoder_2 = text_encoder_2,
                      )

    add_text_embeds = pooled_prompt_embeds

    # logging
    print("encoded prompts")

    ''' 
    [CH] This part is a mystery... mystery start
    '''
    resolution = 512
    crops_coords_top_left = (0,0)
    original_sizes = (resolution, resolution)
    crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)
    original_sizes = torch.tensor(original_sizes, dtype=torch.long)
    crops_coords_top_left = crops_coords_top_left.repeat(len(prompt_embeds), 1)
    original_sizes = original_sizes.repeat(len(prompt_embeds), 1)

    target_size = (resolution, resolution)
    add_time_ids = list(target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.repeat(len(prompt_embeds), 1)
    add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
    add_time_ids = add_time_ids.to(device, dtype=torch.float32)
    # [CH] print("add_time_ids",add_time_ids) # [[1024., 1024.,    0.,    0., 1024., 1024.]]
    negative_add_time_ids = add_time_ids
    '''
    [CH] Mystery end
    '''

    # do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    
    prompt_embeds = prompt_embeds.to(device, dtype=torch.float32)
    add_text_embeds = add_text_embeds.to(device)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # Initialize latent variables
    # latent_shape = (
    #     1,
    #     unet.in_channels,
    #     unet.sample_size,
    #     unet.sample_size,
    # )




    # adding noise
    latents_noise = torch.randn((1,4,64,64), device=device, dtype=torch.float32)

    
        

    



    choice = 0
    
    pred = torch.load(f"visualizations/notes_3/pred_{choice:04d}.pt", map_location=torch.device(device))
    latents_pred = pred.view(1, 64, 64, 4).permute(0, 3, 1, 2)
    latents_pred = 4. * (latents_pred - 0.45)
    






    image = Image.open(f"visualizations/notes_3/reference_image_0000.png")
    image = torch.tensor(np.array(image.convert("RGB")) / 255., dtype=torch.float32, device=device)
    image = image.permute(2, 0, 1)[None,:]
    latents_true = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    latents_true = latents_true.to(device)




    # latents = 0.25 * latents_noise + 0.25 * latents_pred + 0.50 * latents_true
    latents = 0.25 * latents_pred + 0.75 * latents_true

    # testing with the latents...




    # denoise image
    num_steps = 1
    guidance_scale = 10.0
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)
    for t in timesteps:
        with torch.no_grad():
            print("starting denoise step") # logging
            latents_model_input = torch.cat([latents] * 2)

            print("starting unet") # logging 
            noise_pred = unet(
                latents_model_input, 
                t, 
                timestep_cond=None,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            print("completed unet") # logging

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample
    latents = latents.float()








    # Decode latents into an image
    scaling_factor = vae.config.scaling_factor
    with torch.no_grad():
        # Scaling factor for unit variance
        latents = latents / scaling_factor 
        image_tensor = vae.decode(latents).sample[0]

    # Convert to PIL image and save
    image = to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
    image.save(f"final_{choice:04d}.png")

if __name__ == "__main__":
    test_stable_diffusion()
