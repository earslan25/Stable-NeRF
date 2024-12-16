import torch
from .preprocess import load_data
from utils.graphics_utils import get_rays
import math


class StableNeRFDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        dataset_name, 
        shape=(512, 512), 
        encoded_shape=(128, 128), 
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5],
        generate_cuda_ray=False,
        fix_choices=(0, 1),
        percent_objects=0.1
    ):
        
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(encoded_shape, int):
            encoded_shape = (encoded_shape, encoded_shape)

        self.H, self.W = shape
        self.encoded_H, self.encoded_W = encoded_shape
        
        # intrinsic/focal ?
        # ideally this would be a set of objects from different datasets, that is we shuffle images to create pairs
        images, poses, intrinsic = load_data(
            dataset=dataset_name, 
            shape=shape, mean=mean, 
            std=std, 
            fix_choices=fix_choices,
            percent_objects=percent_objects
        )
        print(images.shape, poses.shape, intrinsic.shape)
        # self.intrinsic = torch.tensor([intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]])
        # self.intrinsic = torch.tensor([138.0, 138.0, self.encoded_W // 2, self.encoded_H // 2])


        fov = 47.1
        FovY = self.H / (2 * math.tan(fov / 2))
        FovX = self.W / (2 * math.tan(fov / 2))
        self.intrinsic = torch.tensor([FovX, FovY, self.encoded_W // 2, self.encoded_H // 2])



        if len(images.shape) == 4:
            # single object dummy nerf data (num_images, 3, W, H)
            shuffle_indices = torch.randperm(images.shape[0])
            self.reference_images = images
            self.target_images = images[shuffle_indices]
            self.reference_poses = poses
            self.target_poses = poses[shuffle_indices]
        else:
            # objaverse data (num_objects, num_images, 3, W, H)
            self.reference_images = images[:, 0]
            self.target_images = images[:, 1]
            self.reference_poses = poses[:, 0]
            self.target_poses = poses[:, 1]

            # NOTE, multiple images here... 

        # cuda rays and poses, could be cached for faster training but rn gives error
        if generate_cuda_ray:
            self.reference_rays = get_rays(self.reference_poses.to('cuda'), self.intrinsic, self.encoded_H, self.encoded_W)
            self.target_rays = get_rays(self.target_poses.to('cuda'), self.intrinsic, self.encoded_H, self.encoded_W)
            for key in self.reference_rays.keys():
                self.reference_rays[key] = self.reference_rays[key].to('cpu')
                self.target_rays[key] = self.target_rays[key].to('cpu')
        else:
            self.reference_rays = get_rays(self.reference_poses.to('cpu'), self.intrinsic, self.encoded_H, self.encoded_W)
            self.target_rays = get_rays(self.target_poses.to('cpu'), self.intrinsic, self.encoded_H, self.encoded_W)
            for key in self.reference_rays.keys():
                self.reference_rays[key] = self.reference_rays[key].to('cpu')
                self.target_rays[key] = self.target_rays[key].to('cpu')

    def __getitem__(self, idx):
        target_image = self.target_images[idx]
        target_pose = self.target_poses[idx]
        reference_image = self.reference_images[idx]
        reference_pose = self.reference_poses[idx]

        target_rays_o = self.target_rays['rays_o'][idx]
        target_rays_d = self.target_rays['rays_d'][idx]
        target_rays_inds = self.target_rays['inds'][idx]
        reference_rays_o = self.reference_rays['rays_o'][idx]
        reference_rays_d = self.reference_rays['rays_d'][idx]
        reference_rays_inds = self.reference_rays['inds'][idx]

        return {
            "target_image": target_image,
            "reference_image": reference_image,
            "target_pose": target_pose,
            "reference_pose": reference_pose,
            "target_rays_o": target_rays_o,
            "target_rays_d": target_rays_d,
            "target_rays_inds": target_rays_inds,
            "reference_rays_o": reference_rays_o,
            "reference_rays_d": reference_rays_d,
            "reference_rays_inds": reference_rays_inds
        }

    def __len__(self):
        return self.target_images.shape[0]


def collate_fn(data):
    # Initialize a dictionary to hold the batched data
    batched_data = {}
    # Loop through each key in the dictionary of the first sample
    for key in data[0].keys():
        # Stack all the values for the current key across the batch
        batched_data[key] = torch.stack([sample[key] for sample in data], dim=0)

    return batched_data
