import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from torchvision.transforms.functional import to_pil_image
from utils.sd import encode_prompt

def test_stable_diffusion():
    """
    Generate a simple image.
    """

    # initialize models
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    device = torch.device("cuda")
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

    #prompt = "A futuristic cityscape at sunset, highly detailed, vibrant colors"
    #prompt_2 = "Cyberpunk style, glowing neon lights, wide-angle perspective, sunny day"
    #negative_prompt = "tall buildings"
    #negative_prompt_2 = "american style"

    prompt = "beautiful model"
    prompt_2 = "very curvy, hot, actress, black dress, 4k, high quality"
    negative_prompt = "ugly"
    negative_prompt_2 = "naked"

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
    resolution = 1024
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
    #prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    #add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    #add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    prompt_embeds = torch.cat([prompt_embeds], dim=0)
    add_text_embeds = torch.cat([add_text_embeds], dim=0)
    add_time_ids = torch.cat([add_time_ids], dim=0)
    
    prompt_embeds = prompt_embeds.to(device, dtype=torch.float32)
    add_text_embeds = add_text_embeds.to(device)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # Initialize latent variables
    latent_size = 64

    latent_shape = (
        1,
        unet.in_channels,
        latent_size, # unet.sample_size,
        latent_size  # unet.sample_size,
    )
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    # denoise image
    num_steps = 50
    guidance_scale = 1.0
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)
    for t in timesteps:
        with torch.no_grad():
            print("starting denoise step") # logging
            #latents_model_input = torch.cat([latents] * 2)
            latents_model_input = latents

            print("starting unet") # logging 
            noise_pred = unet(
                latents_model_input, 
                t, 
                timestep_cond=None,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            print("completed unet") # logging

            #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample
    latents = latents.float()

    print(latents.shape) # 1,4,128,128


    # Decode latents into an image
    scaling_factor = vae.config.scaling_factor
    with torch.no_grad():
        # Scaling factor for unit variance
        latents = latents / scaling_factor 
        image_tensor = vae.decode(latents).sample[0] # 3,1024,1024

    print(image_tensor.shape)
    # Convert to PIL image and save
    image = to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
    image.save(f"sd_test_noneg_g{guidance_scale}_res{latent_size*8}_lat{latent_size}_crop{resolution}.png")

if __name__ == "__main__":
    test_stable_diffusion()

'''
import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

# from diffusers import UNet2DModel

repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
torch_device = "cuda"

scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    repo_id, subfolder="text_encoder", use_safetensors=True
)
tokenizer_2 = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer_2")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(repo_id, subfolder="text_encoder_2")

model = UNet2DConditionModel.from_pretrained(
    repo_id, subfolder="unet", use_safetensors=True
)
vae.to(torch_device)
text_encoder.to(torch_device)
text_encoder_2.to(torch_device)
model.to(torch_device)

prompt = ["a photograph of an astronaut riding a horse"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 15  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.mps.manual_seed(
    0
)  # Seed generator to create the initial latent noise
batch_size = len(prompt)

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device),output_hidden_states=True)
    text_embeddings = text_embeddings.hidden_states[-2]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device),output_hidden_states=True)
uncond_embeddings = uncond_embeddings.hidden_states[-2]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# 3. Prepare condition stuff
conditioning_text = ["happy face"]
text_input_ids_2 = tokenizer_2(
    conditioning_text,
    max_length=tokenizer_2.model_max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
).input_ids.to(torch_device)
encoder_output_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
pooled_text_embeds = encoder_output_2[0]
print(pooled_text_embeds.shape)
add_time_ids = [
                torch.tensor([[512, 512]]).to(torch_device),
                torch.tensor([[0, 0]]).to(torch_device),
                torch.tensor([[512, 512]]).to(torch_device),
            ]
add_time_ids = torch.cat(add_time_ids, dim=1).to(torch_device).repeat(batch_size,1)
added_cond_kwargs = {"text_embeds":pooled_text_embeds.repeat(batch_size,1).to(torch_device), "time_ids":add_time_ids}

latents = torch.randn(
    (batch_size, model.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)
latents = latents * scheduler.init_noise_sigma

from tqdm.auto import tqdm
scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = model(
            latent_model_input, t, encoder_hidden_states=text_embeddings, added_cond_kwargs=added_cond_kwargs
        ).sample

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
images = (image * 255).round().astype("uint8")
image = Image.fromarray(image)
image.save("generated_image.png")
'''
