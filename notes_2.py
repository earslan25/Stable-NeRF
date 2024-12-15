
import torch
import matplotlib.pyplot as plt
from nerf.network import NeRFNetwork
from tqdm import tqdm
from utils.graphics_utils import *
from utils.loss_utils import *
from datasets.dataset import StableNeRFDataset, collate_fn
from diffusers import AutoencoderKL

from notes_1 import latent_to_image


torch.autograd.set_detect_anomaly(True)

device = "cuda"
nerf = NeRFNetwork(channel_dim=4).to(device)
nerf.train()

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
vae = AutoencoderKL.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
).to(device)

H, W = 512, 512
LH, LW = 64, 64

name = "nerf" # 'objaverse'
dataset = StableNeRFDataset(dataset_name=name, shape=(H, W), encoded_shape=(H, W), generate_cuda_ray=True, percent_objects=0.0001)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

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

        # NOTE: should the latent be normalized? probably... tbh

        curr_batch_size = reference_image.shape[0]

        reference_image_gt = reference_image_latent.permute(0, 2, 3, 1).view(curr_batch_size, -1, 4)
        pred = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=512)['image']

        # save reference_image_gt and pred to /debug_out
        if name == 'objaverse' and i == 0 or name == 'nerf' and (i + 1) % 101:
            with torch.no_grad():

                ref_img = latent_to_image(reference_image_gt, curr_batch_size, LW, LH)
                pred_img = latent_to_image(pred, curr_batch_size, LW, LH)

                plt.imsave(f"visualizations/notes_2/reference_image_gt_{i}.png", ref_img)
                plt.imsave(f"visualizations/notes_2/pred_{i}.png", pred_img)

        loss = l1_loss(pred, reference_image_gt)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss /= len(dataloader)
    progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")

