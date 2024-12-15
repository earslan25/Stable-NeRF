import torch
import matplotlib.pyplot as plt
from nerf.network import NeRFNetwork
from tqdm import tqdm
from utils.graphics_utils import *
from utils.loss_utils import *
from datasets.dataset import StableNeRFDataset, collate_fn
from torchviz import make_dot


# testing main fn
def test_nerf():
    device = 'cuda'
    nerf = NeRFNetwork().to(device)
    nerf.train()
    print(nerf)

    print_metrics = False
    H, W, B = 128, 128, 1
    rand_image = torch.zeros((B, H, W, 3)).to(device)
    rand_cam = rand_poses(B, device, radius=5)
    fx, fy, cx, cy = 64, 64, W // 2, H // 2
    nerf.mark_untrained_grid(rand_cam, (fx, fy, cx, cy))
    nerf.update_extra_state()
    with torch.no_grad():
        rays = get_rays(rand_cam, (fx, fy, cx, cy), H, W)
        rand_image = torch.gather(rand_image.view(B, -1, 3), 1, torch.stack(3 * [rays['inds']], -1)) # for training
        res = nerf.render(rays['rays_o'], rays['rays_d'])

    if print_metrics:
        res_image = res['image'].view(-1, H, W, 3)
        gt_image = rand_image.view(-1, H, W, 3)
        print(res['depth'].max(), res['depth'].min(), res['depth'].shape)
        print(res_image.min(), res_image.max(), res_image.shape)
        print(l1_loss(res_image, gt_image))
        print(l2_loss(res_image, gt_image))
        print(mse(res_image, gt_image))
        print(psnr(res_image, gt_image))

        print(ssim(res_image.reshape((-1, 3, H, W)), gt_image.reshape((-1, 3, H, W))))

    data = {}
    rays = get_rays(rand_cam, (fx, fy, cx, cy), H, W)
    data['rays_o'] = rays['rays_o']
    data['rays_d'] = rays['rays_d']
    data['inds'] = rays['inds']
    data['images'] = rand_image
    data['H'] = H
    data['W'] = W
    loss_fns = {'l1': l1_loss, 'l2': l2_loss}

    optimizer = torch.optim.Adam(nerf.get_params(1e-4), betas=(0.9, 0.99), eps=1e-15)

    pred_rgb, gt_rgb, losses = nerf.train_step(data, loss_fns=loss_fns)
    print(losses)

    optimizer.zero_grad()
    loss = torch.sum(losses['l1'] * 0.5 + losses['l2'] * 0.5)
    loss.backward()

    for name, param in nerf.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient: {param.grad.norm().item()}")
            assert param.requires_grad
        else:
            print(f"{name} gradient: None")

    optimizer.step()

    pred_rgb_new, gt_rgb_new, losses_new = nerf.train_step(data, loss_fns=loss_fns)
    print(losses_new)

    # assert different results after optimizer step
    assert not torch.allclose(pred_rgb, pred_rgb_new)
    assert not torch.allclose(losses['l1'], losses_new['l1'])
    assert not torch.allclose(losses['l2'], losses_new['l2'])


def test_multi_channel_nerf():
    device = 'cuda'
    channel_dim = 4
    nerf = NeRFNetwork(channel_dim=channel_dim).to(device)
    nerf.train()
    print(nerf.sigma_net)
    print(nerf.encoder_dir)
    print(nerf.color_net)

    print_metrics = False
    H, W, B = 64, 64, 1
    rand_image = torch.rand((B, H, W, channel_dim)).to(device)
    rand_cam = rand_poses(B, device, radius=5)
    fx, fy, cx, cy = 128.0, 128.0, W // 2, H // 2

    nerf.mark_untrained_grid(rand_cam, (fx, fy, cx, cy))
    nerf.update_extra_state()

    data = {}
    rays = get_rays(rand_cam, (fx, fy, cx, cy), H, W)
    rand_image = torch.gather(rand_image.view(B, -1, channel_dim), 1, torch.stack(channel_dim * [rays['inds']], -1)) # for training
    data['rays_o'] = rays['rays_o']
    data['rays_d'] = rays['rays_d']
    data['inds'] = rays['inds']
    data['images'] = rand_image
    data['H'] = H
    data['W'] = W
    loss_fns = {'l1': l1_loss, 'l2': l2_loss}

    optimizer = torch.optim.Adam(nerf.get_params(1e-4), betas=(0.9, 0.99), eps=1e-15)

    pred_rgb, gt_rgb, losses = nerf.train_step(data, loss_fns=loss_fns)

    optimizer.zero_grad()
    loss = torch.sum(losses['l1'] * 0.5 + losses['l2'] * 0.5)
    print(loss)
    loss.backward()

    for name, param in nerf.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient: {param.grad.norm().item()}")
            assert param.requires_grad
        else:
            print(f"{name} gradient: None")

    optimizer.step()

    pred_rgb_new, gt_rgb_new, losses_new = nerf.train_step(data, loss_fns=loss_fns)

    optimizer.zero_grad()
    new_loss = torch.sum(losses_new['l1'] * 0.5 + losses_new['l2'] * 0.5)
    print(new_loss)
    new_loss.backward()

    for name, param in nerf.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient: {param.grad.norm().item()}")
            assert param.requires_grad
        else:
            print(f"{name} gradient: None")

    # assert different results after optimizer step
    assert not torch.allclose(pred_rgb, pred_rgb_new)
    assert not torch.allclose(losses['l1'], losses_new['l1'])
    assert not torch.allclose(losses['l2'], losses_new['l2'])


def train_nerf():
    torch.autograd.set_detect_anomaly(True)

    device = 'cuda'
    nerf = NeRFNetwork().to(device)
    nerf.train()

    H, W = 128, 128
    name = 'nerf'
    # name = 'objaverse'
    dataset = StableNeRFDataset(dataset_name=name, shape=(H, W), encoded_shape=(H, W), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], generate_cuda_ray=True, percent_objects=0.0001)
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
            curr_batch_size = reference_image.shape[0]
 
            reference_image_gt = reference_image.permute(0, 2, 3, 1).view(curr_batch_size, -1, 3)
            pred = nerf.render(reference_rays_o, reference_rays_d, bg_color=bg_color, max_steps=256)['image']

            # save reference_image_gt and pred to /debug_out
            if (name == 'objaverse' and i == 0) or (name == 'nerf' and (i + 1) % 101 == 0):
                with torch.no_grad():
                    plt.imsave(f"debug_out/reference_image_gt_{i}.png", (reference_image_gt[0].detach().view(H, W, 3)).cpu().numpy())
                    # plt.imsave(f"debug_out/reference_image_{i}.png", (reference_image[0].detach().permute(1, 2, 0)).cpu().numpy())
                    plt.imsave(f"debug_out/pred_{i}.png", pred[0].detach().view(H, W, 3).cpu().numpy())

            loss = l1_loss(pred, reference_image_gt)
            # make_dot(loss, params=dict(nerf.named_parameters())).render("debug_out/computation_graph", format="png")
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss /= len(dataloader)
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")


if __name__ == "__main__":
    # test_nerf()  
    # test_multi_channel_nerf() 
    train_nerf()