import itertools
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from nerf.network import NeRFNetwork
from stable_diffusion.network import SDNetwork
from datasets.dataset import StableNeRFDataset, collate_fn
from utils.graphics_utils import *
from utils.loss_utils import *
from utils.system_utils import get_memory_usage
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader, random_split


# TODO
# argparser -> daniel
# distributed training setup -> emre
# prep dataset for objaverse -> emre
# visualize/plot results -> daniel

def forward_iteration(sd, nerf, batch, device, model_args):
    # in model args: encoder_output_dim, channel_dim, max_steps, bg_color
    channel_dim = 4
    bg_color = 1
    max_steps = 256 
    encoder_output_dim = 64  

    sd_module = sd
    if isinstance(sd, torch.nn.parallel.DistributedDataParallel):
        sd_module = sd.module

    target_image = batch['target_image'].to(device)
    reference_image = batch['reference_image'].to(device)
    curr_batch_size = target_image.shape[0]

    with torch.no_grad():
        # latent space
        target_image_lt = sd_module.encode_images(target_image) 
        reference_image_lt = sd_module.encode_images(reference_image) 

    target_rays_o = batch['target_rays_o'].to(device)
    target_rays_d = batch['target_rays_d'].to(device)

    reference_rays_o = batch['reference_rays_o'].to(device)
    reference_rays_d = batch['reference_rays_d'].to(device)

    target_image_gt = target_image_lt.permute(0, 2, 3, 1).view(curr_batch_size, -1, channel_dim)
    reference_image_gt = reference_image_lt.permute(0, 2, 3, 1).view(curr_batch_size, -1, channel_dim)

    # novel for conditioning and ip adaptor
    pred_target_latent = nerf.render(target_rays_o, target_rays_d, bg_color=bg_color, max_steps=max_steps)['image']

    # reference for reconstruction loss for nerf
    pred_reference_latent = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=max_steps)['image']

    # nerf loss on target and reference for reconstruction
    nerf_loss = l1_loss(pred_target_latent, target_image_gt) + l1_loss(pred_reference_latent, reference_image_gt)

    # # reference and target latents put together; not using nerf output for reference latent
    # # NOTE: currently appending cam like this -> we have 4x64x64 + 3x64x64 = 7x64x64, instead of [flat(4x64x64), flat(3x64x64)])
    pred_target_latent = pred_target_latent.view(curr_batch_size, channel_dim, encoder_output_dim, encoder_output_dim)
    target_rays_d = target_rays_d.permute(0, 2, 1).view(curr_batch_size, 3, encoder_output_dim, encoder_output_dim)
    reference_rays_d = reference_rays_d.permute(0, 2, 1).view(curr_batch_size, 3, encoder_output_dim, encoder_output_dim)
    
    # cat and flatten
    target_latent_cat_cam = torch.cat([pred_target_latent, target_rays_d], dim=1).view(curr_batch_size, 1, -1)  # batch_size, 1, 7*64*64
    reference_latent_cat_cam = torch.cat([reference_image_lt, reference_rays_d], dim=1).view(curr_batch_size, 1, -1)  # batch_size, 1, 7*64*64
    image_embeds = torch.cat([target_latent_cat_cam, reference_latent_cat_cam], dim=1)  # batch_size, 2, 7*64*64 for seqlen

    # input to unet
    noise = torch.randn_like(reference_image_lt)

    timesteps = torch.randint(0, sd_module.noise_scheduler.config.num_train_timesteps, (curr_batch_size,), device=device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = sd_module.noise_scheduler.add_noise(reference_image, noise, timesteps)

    # dummy_text_embeds = torch.zeros(batch_size, sd.num_tokens, clip_text_output_dim, device=device)
    add_time_ids = [
        torch.tensor([[512, 512]]).to(device),
        torch.tensor([[0, 0]]).to(device),
        torch.tensor([[512, 512]]).to(device),
    ]
    add_time_ids = torch.cat(add_time_ids, dim=1).to(device).repeat(curr_batch_size,1)
    added_cond_kwargs = {"text_embeds": sd_module.pooled_empty_text_embeds.repeat(curr_batch_size,1).to(device), "time_ids": add_time_ids}
    noise_pred = sd(noisy_latents, timesteps, added_cond_kwargs, image_embeds)
    # noise_pred = sd_module(noisy_latents.to(device), timesteps, added_cond_kwargs, image_embeds.to(device))

    # sd loss on denoising latent reference + noise
    sd_loss = mse_loss(noise_pred.float(), noise.float())

    return sd_loss, nerf_loss


def training(data_args, model_args, opt_args):
    torch.autograd.set_detect_anomaly(True)

    output_dir = Path("output")
    logging_dir = Path("output", "logs")
    save_models = False

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        # mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print(f"Number of processes: {accelerator.num_processes}")
    else:
        print("Using a single device.")

    device = accelerator.device
    print("using device:", device)
    pretrained_models_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'openai/clip-vit-large-patch14'

    print("initializing sd network")
    sd = SDNetwork(pretrained_models_path, image_encoder_path).to(device)
    print("sd network initialized and moved to device")
    channel_dim = 4
    print("initializing nerf network with channel_dim =", channel_dim)
    nerf = NeRFNetwork(channel_dim=channel_dim).to(device)
    nerf.train()
    print("nerf network initialized and moved to device")

    # hardcoded, could not find where to get these from
    # [CH] default latent dimension encoded by sd's vae is: 4x(H/8)x(W/8)
    encoder_input_dim = 512  
    encoder_output_dim = 64  

    # dataset_name = 'objaverse'
    dataset_name = 'nerf'
    generate_cuda_ray = True
    batch_size = 1
    percent_objects = 0.0001
    print(f"initializing dataset {dataset_name} with batch_size={batch_size}, encoder_input_dim={encoder_input_dim}, encoder_output_dim={encoder_output_dim}")
    dataset = StableNeRFDataset(dataset_name, shape=encoder_input_dim, encoded_shape=encoder_output_dim, generate_cuda_ray=generate_cuda_ray, percent_objects=percent_objects)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% training
    val_size = int(0.1 * total_size)   # 10% validation
    test_size = total_size - train_size - val_size  # Remaining 10% for testing

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print("dataset and dataloader initialized")

    epochs = 200
    lr = 1e-4
    weight_decay = 1e-4
    params_to_opt = itertools.chain(sd.image_proj_model.parameters(),  sd.adapter_modules.parameters(), nerf.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)

    sd, nerf, optimizer, train_loader, val_loader = accelerator.prepare(sd, nerf, optimizer, train_loader, val_loader)

    # not using forward method for nerf, can't refactor
    nerf = accelerator.unwrap_model(nerf)

    nerf.mark_untrained_grid(torch.cat([dataset.reference_poses, dataset.target_poses], dim=0).to(device), dataset.intrinsic)

    # note: could be optimized by batching ref/target images together
    num_training_steps = epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps), desc="Training progress", disable=not accelerator.is_main_process)
    losses = []
    for epoch in range(epochs):
        nerf.update_extra_state()
        if accelerator.is_main_process:
            total_loss_train = 0
            total_sd_loss_train = 0
            total_nerf_loss_train = 0
        for i, batch in enumerate(train_loader):
            with accelerator.accumulate(sd, nerf):
                # print(f"batch {i} of epoch {epoch} with {get_memory_usage()} memory usage and batch size {len(batch['target_image'])} on process {accelerator.process_index}")
                
                sd_loss, nerf_loss = forward_iteration(sd, nerf, batch, device, model_args)
                
                total_loss = sd_loss + nerf_loss

                if accelerator.is_main_process and i % 10 == 0:
                    # gather loss
                    avg_loss = accelerator.gather_for_metrics(total_loss).mean().item()
                    avg_sd_loss = accelerator.gather_for_metrics(sd_loss).mean().item()
                    avg_nerf_loss = accelerator.gather_for_metrics(nerf_loss).mean().item()

                    total_loss_train += avg_loss
                    total_sd_loss_train += avg_sd_loss
                    total_nerf_loss_train += avg_nerf_loss

                    progress_bar.set_postfix({'loss': avg_loss, 'sd_loss': avg_sd_loss, 'nerf_loss': avg_nerf_loss})
                progress_bar.update(1)

                optimizer.zero_grad()
                # accelerator.backward(nerf_loss)
                # accelerator.backward(sd_loss)
                accelerator.backward(total_loss)
                optimizer.step()

        # validation
        if accelerator.is_main_process:
            total_loss_valid = 0
            sd_loss_valid = 0
            nerf_loss_valid = 0
        with torch.no_grad():
            for batch in val_loader:
                sd_loss, nerf_loss = forward_iteration(sd, nerf, batch, device, model_args)
                
                total_loss = sd_loss + nerf_loss

                if accelerator.is_main_process:
                    # gather loss
                    avg_loss = accelerator.gather_for_metrics(total_loss.repeat(batch_size)).mean().item()
                    avg_sd_loss = accelerator.gather_for_metrics(sd_loss.repeat(batch_size)).mean().item()
                    avg_nerf_loss = accelerator.gather_for_metrics(nerf_loss.repeat(batch_size)).mean().item()

                    total_loss_valid += avg_loss
                    sd_loss_valid += avg_sd_loss
                    nerf_loss_valid += avg_nerf_loss

        if accelerator.is_main_process:
            total_loss_train /= len(train_loader)
            total_sd_loss_train /= len(train_loader)
            total_nerf_loss_train /= len(train_loader)
            total_loss_valid /= len(val_loader)
            sd_loss_valid /= len(val_loader)
            nerf_loss_valid /= len(val_loader)

            loss_data = {
                "total_loss_train": total_loss_train,
                "sd_loss_train": total_sd_loss_train,
                "nerf_loss_train": total_nerf_loss_train,
                "total_loss_valid": total_loss_valid,
                "sd_loss_valid": sd_loss_valid,
                "nerf_loss_valid": nerf_loss_valid,
            }
            losses.append(loss_data)
            print(f"Epoch {epoch} - train loss: {total_loss_train:.4f} - validation loss: {total_loss_valid:.4f}")

    if accelerator.is_main_process and save_models:
        # save model and test dataset
        torch.save(accelerator.unwrap_model(sd).state_dict(), output_dir / "sd.pth")
        torch.save(accelerator.unwrap_model(nerf).state_dict(), output_dir / "nerf.pth")
        torch.save(test_dataset, output_dir / "test_dataset.pth")

    # exit gracefully if using multiple processes
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.print("Training finished, exiting...")

        exit()

    return sd, nerf, test_dataset, losses


# not validation, full inference
def evaluate(sd, nerf, test_data, model_args):
    output_dir = Path("output")
    logging_dir = Path("output", "logs")
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    device = 'cuda'
    print("using device:", device)

    sd = sd.to(device)
    sd.eval()
    channel_dim = 4
    noise_time_steps = 64
    nerf = nerf.to(device)
    nerf.eval()

    bg_color = torch.ones(channel_dim, device=device)
    max_steps = 4  # reduced from 1024 for memory reasons during testing

    encoder_input_dim = 512  
    encoder_output_dim = 64  

    batch_size = 2
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss = 0
    total_psnr = 0
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            curr_batch_size = len(batch['target_image'])
            target_image = batch['target_image']  # THIS IS THE GT TARGET WE WANT TO PREDICT
            reference_image = batch['reference_image']

            # cond 1 for eval: latent features of reference image
            reference_image = sd.encode_images(reference_image) 
            reference_rays_d = batch['reference_rays_d'].to(device)

            # cond 2 for eval: novel/target features from nerf
            target_rays_o = batch['target_rays_o'].to(device)
            target_rays_d = batch['target_rays_d'].to(device)
            pred_target_latent = nerf.render(target_rays_o, target_rays_d, bg_color=bg_color, max_steps=max_steps)['image']

            pred_target_latent = pred_target_latent.view(curr_batch_size, channel_dim, encoder_output_dim, encoder_output_dim)
            target_rays_d = target_rays_d.view(curr_batch_size, 3, encoder_output_dim, encoder_output_dim)
            reference_rays_d = reference_rays_d.view(curr_batch_size, 3, encoder_output_dim, encoder_output_dim)
            
            # cond 2.5 for eval: plucker coordinates concatenated to latent features
            target_latent_cat_cam = torch.cat([pred_target_latent, target_rays_d], dim=1).view(curr_batch_size, 1, -1)  # batch_size, 1, 7*64*64
            reference_latent_cat_cam = torch.cat([reference_image, reference_rays_d], dim=1).view(curr_batch_size, 1, -1)  # batch_size, 1, 7*64*64
            image_embeds = torch.cat([target_latent_cat_cam, reference_latent_cat_cam], dim=1)  # batch_size, 2, 7*64*64 for seqlen

            # input to unet
            noise = torch.randn_like(reference_image)

            # TODO: not sure if this is correct
            timesteps = torch.arange(0, noise_time_steps, device=device, dtype=torch.float32) / noise_time_steps
            timesteps = timesteps.long()

            # dummy_text_embeds = torch.zeros(batch_size, sd.num_tokens, clip_text_output_dim, device=device)
            add_time_ids = [
                torch.tensor([[512, 512]]).to(device),
                torch.tensor([[0, 0]]).to(device),
                torch.tensor([[512, 512]]).to(device),
            ]
            add_time_ids = torch.cat(add_time_ids, dim=1).to(device).repeat(curr_batch_size,1)
            added_cond_kwargs = {"text_embeds":sd.pooled_empty_text_embeds.repeat(curr_batch_size,1).to(device), "time_ids":add_time_ids}

            # forward diffusion process
            # TODO: idk how to do this
            

if __name__ == "__main__":
    # TODO
    # resolve hacks (fit latents to clip/image projector dims) -> chia
    # argparser -> daniel
    # do training -> chia

    arg_1, arg_2, arg_3 = None, None, None
    train = True
    load_path = 'output/'
    if train:
        sd, nerf, test_dataset, losses = training(arg_1, arg_2, arg_3)
    else:
        assert os.path.exists(load_path), "Path does not exist"
        sd = torch.load(load_path + 'sd.pth')
        nerf = torch.load(load_path + 'nerf.pth')
        test_dataset = torch.load(load_path + 'test_dataset.pth')

    # evaluate does not use accelerator, if accelerator is used during training, program will exit after saving models
    # evaluate(sd, nerf, test_dataset, arg_2)
