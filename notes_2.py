
import torch
import matplotlib.pyplot as plt
from nerf.network import NeRFNetwork
from tqdm import tqdm
from utils.graphics_utils import *
from utils.loss_utils import *
from datasets.dataset import StableNeRFDataset, collate_fn
from diffusers import AutoencoderKL


def latent_to_image(image: np.ndarray, b: int, W: int, H: int) -> np.ndarray:
    """
    Converts 4 channel latents into image representations
    """

    image = image[0].detach().view(H, W, 4).cpu().numpy()
    image = image / max(np.abs(np.max(image)), np.abs(np.min(image)))
    image = np.sum(image, -1) / 4.

    return image


torch.autograd.set_detect_anomaly(True)

device = "cuda" # "cuda"
nerf = NeRFNetwork(channel_dim=4).to(device)
nerf.train()

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
vae = AutoencoderKL.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
).to(device)

H, W = 512, 512
LH, LW = 64, 64

name = "objaverse" # "nerf"
dataset = StableNeRFDataset(dataset_name=name, shape=(H, W), encoded_shape=(LH, LW), generate_cuda_ray=device=="cuda", percent_objects=0.01)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

# for i, batch in enumerate(dataloader):
#     reference_image = batch['reference_image'].to(device)
#     print(torch.max(reference_image))
#     print(torch.min(reference_image))
#     break

optimizer = torch.optim.Adam(nerf.get_params(1e-2), betas=(0.9, 0.99), eps=1e-15)

bg_color = 0
epochs = 100

nerf.mark_untrained_grid(dataset.reference_poses, dataset.intrinsic)

progress_bar = tqdm(range(epochs))
for epoch in progress_bar:
    nerf.update_extra_state()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        if name == 'objaverse' and i > 0:
            break
        reference_rays_o = batch['reference_rays_o'].to(device)
        reference_rays_d = batch['reference_rays_d'].to(device)

        reference_image = batch['reference_image'].to(device)
        reference_image_latent = vae.encode(reference_image).latent_dist.sample() * vae.config.scaling_factor

        # NOTE: fake normalization
        reference_image_latent = (reference_image_latent + 3.) / 6.

        curr_batch_size = reference_image.shape[0]

        reference_image_gt = reference_image_latent.permute(0, 2, 3, 1).view(curr_batch_size, -1, 4)
        pred = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=512)['image']

        # save reference_image_gt and pred to /debug_out
        if name == 'objaverse' and i == 0 or name == 'nerf' and (i + 1) % 101:
            with torch.no_grad():

                # print(reference_image_gt.shape)
                # print(pred.shape)

                mean = 0.5
                std = 0.5

                ref_img = latent_to_image(reference_image_gt, curr_batch_size, LW, LH)
                pred_img = latent_to_image(pred, curr_batch_size, LW, LH)

                reference_image = reference_image * std + mean

                torch.save(pred, f"visualizations/notes_4/pred_{i:04d}.pt")
                plt.imsave(f"visualizations/notes_4/reference_image_{i:04d}.png", (reference_image.permute(0, 2, 3, 1).view(curr_batch_size, -1, 3)[0].detach().view(H, W, 3)).cpu().numpy())
                plt.imsave(f"visualizations/notes_4/reference_latent_{i:04d}.png", ref_img)
                plt.imsave(f"visualizations/notes_4/pred_latent_{i:04d}.png", pred_img)

        loss = l1_loss(pred, reference_image_gt)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss /= len(dataloader)
    progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")

