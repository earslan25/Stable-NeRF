import torch
from stable_diffusion.network import SDNetwork
from datasets.dataset import StableNeRFDataset, collate_fn
from PIL import Image
import numpy as np

from torchvision import transforms




def save_image(tensor_image, filename):
    image = tensor_image.squeeze(0)  # Shape: (3, 512, 512)    
    image = (image + 1) / 2 * 255  # This maps [-1, 1] to [0, 255]
    image = image.clamp(0, 255).to(torch.uint8)
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Shape: (512, 512, 3)
    pil_image = Image.fromarray(image_np)
    pil_image.save(filename)


def test_sd():
    # device = 'cuda'
    device = 'cpu'
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    channel_dim = 4
    network = SDNetwork(pretrained_models_path, image_encoder_path, channel_dim=channel_dim, cat_cam=False).to(device)
    print(network.vae.config)
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 512, 512, device=device)
        dummy_latent = network.encode_images(dummy_img)
        print(dummy_latent.shape)
        dummy_img_recon = network.decode_latents(dummy_latent)
        print(dummy_img_recon.shape)

    print(dummy_img.min(), dummy_img.max(), dummy_img.mean())
    print(dummy_img_recon.min(), dummy_img_recon.max(), dummy_img_recon.mean())


def test_sd_reconstruction(channel_dim=4):
    device = torch.device("cuda")
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    cat_cam = True # can be 0 to not concat cam to latent
    print(f"initiating sd network with channel_dim={channel_dim} and cat_cam={cat_cam}")
    sd = SDNetwork(pretrained_models_path, image_encoder_path, channel_dim=channel_dim, cat_cam=cat_cam, from_pretrained=True).to(device)
    print("sd network initiated and moved to device")

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
    save_image(normalized_image, "batch_ai_face.png")
    normalized_image = sd.encode_images(normalized_image)  # latent space
    recon_normalized_image =  sd.decode_latents(normalized_image)

    recon_normalized_image = recon_normalized_image.squeeze().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0)  # Convert tensor to NumPy array
    recon_normalized_image = Image.fromarray(((recon_normalized_image+1)/2 * 255).astype('uint8'))  # Convert to PIL Image
    recon_normalized_image.save("batch_recon_ai_face.png")

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
        save_image(recon_target_image, "batch_recon_target.png")
        save_image(recon_reference_image, "batch_recon_reference.png")

        exit(0)

if __name__ == '__main__':
    test_sd_reconstruction(channel_dim=4)

