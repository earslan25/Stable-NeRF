import torch
from nerf.network import NeRFNetwork
from utils.graphics_utils import *
from utils.loss_utils import *


# testing main fn
def test_nerf():
    device = 'cuda'
    nerf = NeRFNetwork()
    print(nerf)
    print(nerf.summary())

    H, W = 256, 256
    rand_image = torch.rand((1, 3, W, H)).to(device)
    rand_cam = rand_poses(1, device)
    fx, fy, cx, cy = 1, 1, 0, 0
    rays = get_rays(rand_cam, (fx, fy, cx, cy), H, W)
    res = nerf.render(rays['rays_o'], rays['rays_d'])

    print(res['depth'])
    print(res['image'])
    print(rand_image)
    print(ssim(res['image'], rand_image))
    print(l1_loss(res['image'], rand_image))
    print(l2_loss(res['image'], rand_image))
    print(mse(res['image'], rand_image))
    print(psnr(res['image'], rand_image))


if __name__ == "__main__":
    test_nerf()