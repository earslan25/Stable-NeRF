import torch
from stable_diffusion.network import SDNetwork
from datasets.dataset import StableNeRFDataset, collate_fn
from PIL import Image
from tqdm import tqdm
import numpy as np

from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from torchvision.transforms.functional import to_pil_image



def save_image(tensor_image, filename):
    image = tensor_image.squeeze(0)  # Shape: (3, 512, 512)    
    image = (image + 1) / 2 * 255  # This maps [-1, 1] to [0, 255]
    image = image.clamp(0, 255).to(torch.uint8)
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Shape: (512, 512, 3)
    pil_image = Image.fromarray(image_np)
    pil_image.save(filename)


def test_sd():
    device = 'cuda'
    # device = 'cpu'
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    network = SDNetwork(pretrained_models_path, image_encoder_path, embed_cache_device=device).to(device)
    # print(network.vae.config)
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 512, 512, device=device)
        dummy_latent = network.encode_images(dummy_img)
        print(dummy_latent.shape)
        dummy_img_recon = network.decode_latents(dummy_latent)
        print(dummy_img_recon.shape)

    print(dummy_img.min(), dummy_img.max(), dummy_img.mean())
    print(dummy_img_recon.min(), dummy_img_recon.max(), dummy_img_recon.mean())


def test_sd_reconstruction():
    device = torch.device("cuda")
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    print("initializing sd network")
    sd = SDNetwork(pretrained_models_path, image_encoder_path, embed_cache_device=device).to(device)
    sd.eval()
    print("sd network initialized and moved to device")

    encoder_input_dim = 512  
    encoder_output_dim = 64  
    clip_text_output_dim = 768

    dataset_name = 'nerf'
    batch_size = 1
    print(f"initiating dataset {dataset_name} with batch_size={batch_size}, encoder_input_dim={encoder_input_dim}, encoder_output_dim={encoder_output_dim}")
    dataset = StableNeRFDataset(dataset_name, shape=encoder_input_dim, encoded_shape=encoder_output_dim)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print("dataset and dataloader initiated")

    # Load the image using PIL
    image_path = "ai_face.png"
    image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format

    # Define the transformation to normalize the image
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts the image to a tensor with shape (C, H, W) and values in [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Apply the transformation
    normalized_image = transform(image)
    normalized_image = normalized_image.unsqueeze(0).to(device)
    save_image(normalized_image, "debug_out/batch_ai_face.png")
    normalized_image = sd.encode_images(normalized_image)  # latent space
    recon_normalized_image =  sd.decode_latents(normalized_image)

    recon_normalized_image = recon_normalized_image.squeeze().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0)  # Convert tensor to NumPy array
    recon_normalized_image = Image.fromarray(((recon_normalized_image+1)/2 * 255).astype('uint8'))  # Convert to PIL Image
    recon_normalized_image.save("debug_out/batch_recon_ai_face.png")

    for i, batch in enumerate(dataloader):
        target_image = batch['target_image'].to(device)
        reference_image = batch['reference_image'].to(device)
        save_image(target_image, "batch_target.png")
        save_image(reference_image, "batch_reference.png")

        with torch.no_grad():
            target_image = sd.encode_images(target_image)  # latent space
            reference_image = sd.encode_images(reference_image)  # latent space
            print("target_image latent shape", target_image.shape)
            print("reference_image latent shape", reference_image.shape)
        
        recon_target_image = sd.decode_latents(target_image)
        recon_reference_image =  sd.decode_latents(reference_image)
        save_image(recon_target_image, "debug_out/batch_recon_target.png")
        save_image(recon_reference_image, "debug_out/batch_recon_reference.png")

        exit(0)


def test_sd_denoise():
    device = torch.device("cuda")
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    print("initializing sd network")
    sd = SDNetwork(pretrained_models_path, image_encoder_path)
    sd = sd.to(device)
    sd.eval()
    print("sd network initialized and moved to device")

    encoder_input_dim = 512  
    encoder_output_dim = 64  
    clip_text_output_dim = 768

    # Load the image using PIL
    image_path = "ai_face.png"
    image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format

    # Define the transformation to normalize the image
    transform = transforms.Compose([
        transforms.Resize((encoder_input_dim, encoder_input_dim)), 
        transforms.ToTensor(),  # Converts the image to a tensor with shape (C, H, W) and values in [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Apply the transformation
    normalized_image = transform(image)
    normalized_image = normalized_image.unsqueeze(0).to(device)
    save_image(normalized_image, "debug_out/batch_ai_face.png")
    normalized_image = sd.encode_images(normalized_image)  # latent space
    normalized_image = torch.cat([normalized_image, normalized_image], dim=0)

    curr_batch_size = 2

    # input to unet
    latents = torch.randn_like(normalized_image)

    prompt_embeds = sd.prompt_embeds.repeat(curr_batch_size, 1, 1)  # (bzs, 77, 2048)
    add_text_embeds = sd.add_text_embeds.repeat(curr_batch_size, 1)
    add_time_ids = sd.add_time_ids.repeat(curr_batch_size, 1)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    print("shapes: ", latents.shape, prompt_embeds.shape, add_text_embeds.shape, add_time_ids.shape)
    # (bzs*2, seq, 2048) vs (bzs, seq*2, 2048)

    num_steps = 50
    guidance_scale = 10.0
    sd.noise_scheduler.set_timesteps(num_steps)
    timesteps = sd.noise_scheduler.timesteps.to(device)
    for t in tqdm(timesteps, desc="Denoising"):
        with torch.no_grad():
            # latents_model_input = torch.cat([latents] * 2)
            latents_model_input = latents

            noise_pred = sd.unet(latents_model_input, t, prompt_embeds, added_cond_kwargs=added_cond_kwargs, timestep_cond=None).sample

            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = sd.noise_scheduler.step(noise_pred, t, latents).prev_sample
    latents = latents.float()

    recon_normalized_image = sd.decode_latents(latents)[0]
    image = to_pil_image(recon_normalized_image.add(1).div(2).clamp(0, 1))
    image.save(f"debug_out/noise_denoised.png")


if __name__ == '__main__':
    test_sd_denoise()

