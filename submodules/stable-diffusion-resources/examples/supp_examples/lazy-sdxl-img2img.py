from diffusers import StableDiffusionXLImg2ImgPipeline, DiffusionPipeline
import torch



'''
Image-to-Image Process in Stable Diffusion
1. Encode the Input Image:
    - The input image is passed through an encoder (typically a pretrained Variational Autoencoder, VAE).
    - This step maps the image into a lower-dimensional latent space representation, which is suitable for processing by the Stable Diffusion model.
2. Modify Latent Representations:
    - The latent representation of the input image is noised up to a certain level (defined by a diffusion strength or noise scale parameter).
    - This adds flexibility to the output while still retaining structural elements of the input image.
3. Guided Denoising:
    - Using the noised latent as the starting point, the Stable Diffusion model iteratively denoises the representation while incorporating the guidance provided by:
Text Prompts: Input to a text encoder (like CLIP or a similar transformer) that generates conditioning information.
Other Inputs: Such as style conditioning or mask information for partial editing.
Decode the Latents:
The resulting denoised latent is passed back through the VAE decoder to produce the final output image.
'''


# Load the pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16"
) 
pipe = pipe.to("cuda") # Use GPU if available

# Load an image
from PIL import Image
init_image = Image.open("1234.png").convert("RGB")

# Generate the new image
# prompt = "A futuristic cityscape at sunset, highly detailed, vibrant colors"
prompt = ""
image = pipe(prompt=prompt, image=init_image, strength=0.1).images[0]
'''
strength: Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`. Note that in the case of
                `denoising_start` being declared as an integer, the value of `strength` will be ignored.
'''

# Save the image
image.save("output_img2img.jpg")
