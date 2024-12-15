
# NOTE
    # the end goal here is fakery
    # we need convincing visuals more than anything

import torch
import matplotlib.pyplot as plt
import numpy as np

from datasets.dataset import StableNeRFDataset, collate_fn
from diffusers import AutoencoderKL
# from stable_diffusion.network import SDNetwork


def latent_to_image(image: np.ndarray, b: int, W: int, H: int) -> np.ndarray:
    """
    Converts 4 channel latents into image representations
    """

    image = image.permute(0, 2, 3, 1).view(b, -1, 4)[0].detach().view(H, W, 4).cpu().numpy()
    image = image / max(np.abs(np.max(image)), np.abs(np.min(image)))
    image = np.sum(image, -1) / 4.

    return image


def visualize_latents():
    """
    visualize latents of the nerf dataset through the sd encoding
    """

    print("visualizing latents")

    device = "cpu"

    H, W = 512, 512
    LH, LW = 64, 64
    name = 'nerf'
    # name = 'objaverse'
    dataset = StableNeRFDataset(dataset_name=name, shape=(H, W), encoded_shape=(H, W), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], generate_cuda_ray=device=="cuda", percent_objects=0.0001)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        for i, batch in enumerate(dataloader):

            # batch size
            b = 1

            # images (b, 3, W, H)
            target_image = batch["target_image"]
            reference_image = batch["reference_image"]
            print("image shape ", target_image.shape)

            # encodings (b, 4, 16, 16)
            target_latent = vae.encode(target_image).latent_dist.sample() * vae.config.scaling_factor
            reference_latent = vae.encode(reference_image).latent_dist.sample() * vae.config.scaling_factor
            print("encodings shape ", target_latent.shape)

            # image visualization
            plt.imsave(f"visualizations/target_image.png", (target_image.permute(0, 2, 3, 1).view(b, -1, 3)[0].detach().view(H, W, 3)).cpu().numpy())
            plt.imsave(f"visualizations/reference_image.png", (reference_image.permute(0, 2, 3, 1).view(b, -1, 3)[0].detach().view(H, W, 3)).cpu().numpy())

            # latent visualization
            target_latent_img = latent_to_image(target_latent, b, LW, LH)
            reference_latent_img = latent_to_image(reference_latent, b, LW, LH)
            plt.imsave(f"visualizations/target_latent.png", target_latent_img)
            plt.imsave(f"visualizations/reference_latent.png", reference_latent_img)


            # run the latent through nerf
                # train the nerf
                # what does the nerf do to these images?
                # what does it look like after running through the nerf...
            # run the latent through the unet 
            # run decode the latent

            break


if __name__ == "__main__":
    visualize_latents()
