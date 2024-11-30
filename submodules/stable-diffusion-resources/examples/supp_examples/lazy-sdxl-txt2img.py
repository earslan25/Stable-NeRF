import torch
from diffusers import StableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda") 

prompt_a = "an apple"
prompt_b = "a banana"
seed = 1234
num_images_per_prompt = 1 # batch size, increase if you have better GPU

generator = torch.Generator(device='cuda').manual_seed(seed)
images = pipe(prompt=prompt_a, prompt_2=prompt_b, num_images_per_prompt=num_images_per_prompt, generator=generator).images
