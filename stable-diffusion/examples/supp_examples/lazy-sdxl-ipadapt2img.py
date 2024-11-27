from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

scale = 1.0

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipe.set_ip_adapter_scale(scale)
pipe = pipe.to("cuda")

image = load_image("1234.png")
generator = torch.Generator(device="cuda").manual_seed(0)
image = pipe(
    prompt="",
    ip_adapter_image=image,
    negative_prompt="",
    num_inference_steps=100,
    generator=generator,
).images[0]

image.save(f"output_np_ipadapt2img_scale={scale}.jpg")
