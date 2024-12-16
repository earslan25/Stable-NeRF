
import torch
import matplotlib.pyplot as plt
from nerf.network import NeRFNetwork
from tqdm import tqdm
from utils.graphics_utils import *
from utils.loss_utils import *
from datasets.dataset import StableNeRFDataset, collate_fn
from diffusers import AutoencoderKL


torch.autograd.set_detect_anomaly(True)

device = "cuda" # "cuda"
nerf = NeRFNetwork().to(device)
nerf.train()

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

H, W = 256, 256
# LH, LW = 64, 64

name = "objaverse" # "nerf"
dataset = StableNeRFDataset(dataset_name=name, shape=(H, W), encoded_shape=(H, W), generate_cuda_ray=device=="cuda", percent_objects=0.01)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

# for i, batch in enumerate(dataloader):
#     reference_image = batch['reference_image'].to(device)
#     print(torch.max(reference_image))
#     print(torch.min(reference_image))
#     break

optimizer = torch.optim.Adam(nerf.get_params(1e-2), betas=(0.9, 0.99), eps=1e-15)

bg_color = 0
epochs = 201

nerf.mark_untrained_grid(dataset.reference_poses, dataset.intrinsic)




path = 16






progress_bar = tqdm(range(epochs))
for epoch in progress_bar:
    nerf.update_extra_state()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        # if name == 'objaverse' and i > 0:
        #     break
        reference_rays_o = batch['reference_rays_o'].to(device)
        reference_rays_d = batch['reference_rays_d'].to(device)

        reference_image = batch['reference_image'].to(device)
        # reference_image_latent = vae.encode(reference_image).latent_dist.sample() * vae.config.scaling_factor

        # NOTE: fake normalization
        # reference_image_latent = (reference_image_latent + 3.) / 6.

        curr_batch_size = reference_image.shape[0]

        reference_image_gt = reference_image.permute(0, 2, 3, 1).view(curr_batch_size, -1, 3)
        pred = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=512)['image']




        # save reference_image_gt and pred to /debug_out
        # if name == 'objaverse' and i == 0 or name == 'nerf' and (i + 1) % 101:
        if epoch % 10 == 0:
            with torch.no_grad():

                mean = 0.5
                std = 0.5

                reference_image = reference_image * std + mean

                torch.save(pred, f"visualizations/notes_{path}/pred_{epoch:04d}_{i:04d}.pt")
                plt.imsave(f"visualizations/notes_{path}/reference_image_{i:04d}.png", (pred[0].detach().view(H, W, 3).cpu().numpy()))
                plt.imsave(f"visualizations/notes_{path}/reference_image_{i:04d}.png", (reference_image.permute(0, 2, 3, 1).view(curr_batch_size, -1, 3)[0].detach().view(H, W, 3)).cpu().numpy())
                # plt.imsave(f"visualizations/notes_{path}/reference_latent_{i:04d}.png", ref_img)
                # plt.imsave(f"visualizations/notes_{path}/pred_latent_{epoch:04d}_{i:04d}.png", pred_img)

        loss = l1_loss(pred, reference_image_gt)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss /= len(dataloader)
    progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")