import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection, CLIPTextModel, \
    CLIPTokenizer, CLIPTextModelWithProjection, CLIPImageProcessor

from torchvision.utils import save_image
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
clip_encoder_path = 'openai/clip-vit-large-patch14'

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe = pipe.to("cuda")

# Load the SDXL pipeline
#unet = UNet2DConditionModel.from_pretrained(pretrained_models_path, subfolder="unet")
unet = pipe.unet
unet = unet.to(device)

# 1. Generate or load a starting image (e.g., random noise or an image)
batch_size = 1
height, width = 64, 64  # adjust based on your requirements
latents = torch.randn((batch_size, 4, height, width), device=device)  # Typically SDXL uses 4 channels for latents

# 2. Load sdxl stuff
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
tokenizer_2 = pipe.tokenizer_2
text_encoder_2 = pipe.text_encoder_2
#tokenizer = CLIPTokenizer.from_pretrained(pretrained_models_path, subfolder="tokenizer")
#text_encoder = CLIPTextModel.from_pretrained(pretrained_models_path, subfolder="text_encoder").to(device)
#tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_models_path, subfolder="tokenizer_2")
#text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_models_path, subfolder="text_encoder_2").to(device)

# 3. Prepare condition stuff
conditioning_text = ["happy face, 4k high quality"]

text_input_ids = tokenizer(conditioning_text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
encoder_output = text_encoder(text_input_ids, output_hidden_states=True)
text_embeds = encoder_output.hidden_states[-2]

text_input_ids_2 = tokenizer_2(
    [""],
    max_length=tokenizer_2.model_max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
).input_ids.to(device)
encoder_output_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
pooled_text_embeds = encoder_output_2[0]
text_embeds_2 = encoder_output_2.hidden_states[-2]
text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) 

add_time_ids = [
                torch.tensor([[512, 512]]).to(device),
                torch.tensor([[0, 0]]).to(device),
                torch.tensor([[512, 512]]).to(device),
            ]
add_time_ids = torch.cat(add_time_ids, dim=1).to(device).repeat(batch_size,1)
added_cond_kwargs = {"text_embeds":pooled_text_embeds.repeat(batch_size,1).to(device), "time_ids":add_time_ids}

# 3. Run the denoising loop with the UNet and the text conditioning
#scheduler = DDIMScheduler.from_pretrained(pretrained_models_path, subfolder="scheduler")
scheduler = pipe.scheduler
scheduler.set_timesteps(50)  # Ensure you have 50 timesteps
timesteps = scheduler.timesteps  # Default scheduler for SDXL
latents = latents * scheduler.init_noise_sigma  # Initialize latents with noise

'''
print(latents.shape)
print(timesteps.shape)
print(text_embeds.shape)
print(added_cond_kwargs["text_embeds"].shape)
print(added_cond_kwargs["time_ids"].shape)
torch.Size([1, 4, 64, 64])
torch.Size([1000])
torch.Size([1, 77, 2048])
torch.Size([1, 1280])
torch.Size([1, 6])
'''

# Denoising loop
for t in tqdm(timesteps):
    latents = scheduler.scale_model_input(latents, timestep=t)
    with torch.no_grad():
        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=text_embeds,  # Text condition here
            added_cond_kwargs=added_cond_kwargs
        ).sample
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# 4. Save the final denoised image
# vae = AutoencoderKL.from_pretrained(pretrained_models_path, subfolder="vae").to(device)
vae = pipe.vae
print(vae.config.scaling_factor)
latents = latents / vae.config.scaling_factor
image = vae.decode(latents).sample

image = torch.clamp(image, -1, 1)  # Ensure values are between -1 and 1
image = (image + 1) / 2  # Normalize to [0, 1] for proper visualization
save_image(image, 'output_image.png')  # Saves as 'output_image.png'

