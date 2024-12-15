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
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = model(
            latent_model_input, t, encoder_hidden_states=text_embeddings, added_cond_kwargs=added_cond_kwargs
        ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

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
