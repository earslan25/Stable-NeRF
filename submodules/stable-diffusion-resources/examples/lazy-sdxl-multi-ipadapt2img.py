import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

# 1. Specify ip_adapter model
ip_adapter_name = "ip-adapter-plus_sdxl_vit-h.safetensors" # "ip-adapter_sdxl.bin"

# 2. Load some cartoon images as style
style_folder = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy"
style_images = [load_image(f"{style_folder}/img{i}.png") for i in range(0,10,3)]
for idx, style_image in enumerate(style_images):
    style_image.save(f"style_image_{idx}.png")
print("Style images saved successfully!")

# 3. Load our own image then add it to style_images 
style_cond_image = load_image("style_condition.png")
style_images.append(style_cond_image)

# 3.a example for (robot + cyberpunk)
# style_images = [style_images[1].copy(), style_images[-1].copy()] 

# 4. Initialize image_encoder, only required for "ip-adapter-plus_sdxl_vit-h.safetensors"
if "vit" in ip_adapter_name:
    print("loading image encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
else:
    image_encoder = None

# 5. Load pipe
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name=ip_adapter_name
)
pipe = pipe.to("cuda")
pipe.set_ip_adapter_scale(1.0) 
pipe.enable_model_cpu_offload()

# print(pipe.unet.config.encoder_hid_dim_type) # ip_image_proj

# 6. Generate image
generator = torch.Generator(device="cuda").manual_seed(0)
image = pipe(
    prompt="",
    ip_adapter_image=[style_images],
    negative_prompt="",
    num_inference_steps=100,
    num_images_per_prompt=1,
    generator=generator,
).images[0]
image.save(f"output_{len(style_images)}multistyle_ipadapt2img.jpg")
