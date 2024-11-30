import torch
from nerf.network import NeRFNetwork
from utils.graphics_utils import *
from utils.loss_utils import *


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

    nerf.mark_untrained_grid(rand_cam, (fx, fy, cx, cy))
    nerf.update_extra_state()

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
    # print(nerf)

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

if __name__ == "__main__":
    # test_nerf()  # requires compiling with num channels 3
    test_multi_channel_nerf()  # requires compiling with num channels 4