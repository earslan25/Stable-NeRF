
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



# NOTE: get stable diffusion to generate a simple image

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from torchvision.transforms.functional import to_pil_image
from utils import encode_prompt

def test_stable_diffusion():
    """
    Generate a simple image.
    """

    # with autocast(dtype=torch.float16):
    # with torch.amp.autocast('cuda', args...):

    # with torch.autocast("cpu", torch.float16):
    

    # initialize models
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch.float16
    ).to(device)

    # random noise
    input_image = torch.rand((1,3,512,512))

    # encode input image
    latents = vae.encode(input_image).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    prompt = ""
    prompt_2 = ""
    negative_prompt = ""
    negative_prompt_2 = ""

    prompt_embeds, _, _, _ = encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        tokenizer = tokenizer,
        tokenizer_2 = tokenizer_2,
        text_encoder = text_encoder,
        text_encoder_2 = text_encoder_2,
    )


    # denoise image
    num_steps = 3
    guidance_scale = 1.0
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    for t in timesteps:
        with torch.no_grad():
            latents_model_input = torch.cat([latents] * 2) # NOTE: why?

            latents_model_input = latents_model_input.to(dtype=torch.float32)
            # latents_model_input = torch.tensor()
            # print(latents_model_input)

            noise_pred = unet(
                latents_model_input, 
                t, 
                timestep_cond=None,
                encoder_hidden_states=prompt_embeds,
                # added_cond_kwargs=added_cond_kwargs,
            ).sample

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
    image.save("cache/sd_test.png")

if __name__ == "__main__":
    test_stable_diffusion()
