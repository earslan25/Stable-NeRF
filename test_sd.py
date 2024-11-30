import torch
from stable_diffusion.network import SDNetwork


def test_sd():
    device = 'cuda'
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    network = SDNetwork(pretrained_models_path, image_encoder_path).to(device)

    dummy_img = torch.randn(1, 3, 512, 512, device=device)
    dummy_latent = network.encode_images(dummy_img)
    print(dummy_latent.shape)
    dummy_img_recon = network.decode_latents(dummy_latent)
    print(dummy_img_recon.shape)

    print(dummy_img.min(), dummy_img.max(), dummy_img.mean())
    print(dummy_img_recon.min(), dummy_img_recon.max(), dummy_img_recon.mean())


if __name__ == '__main__':
    test_sd()

