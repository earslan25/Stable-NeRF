import torch
from .preprocess import load_data
from utils.graphics_utils import get_rays


class StableNeRFDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset_name, shape=(512, 512), encoded_shape=(128, 128), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(encoded_shape, int):
            encoded_shape = (encoded_shape, encoded_shape)

        self.H, self.W = shape
        self.encoded_H, self.encoded_W = encoded_shape
        
        # intrinsic/focal ?
        # ideally this would be a set of objects from different datasets, that is we shuffle images to create pairs
        images, poses, intrinsic = load_data(dataset=dataset_name, shape=shape, mean=mean, std=std)
        self.intrinsic = torch.tensor([intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]])
        shuffle_indices = torch.randperm(images.shape[0])
        self.intrinsic = torch.tensor([128.0, 128.0, self.encoded_W // 2, self.encoded_H // 2])
        # currently generating rays in cpu, could be optimized to use gpu but will need to be moved back for memory reasons
        self.reference_images = images
        self.reference_poses = poses
        self.reference_rays = get_rays(self.reference_poses, self.intrinsic, self.encoded_H, self.encoded_W)
        self.target_images = images[shuffle_indices]
        self.target_poses = poses[shuffle_indices]
        self.target_rays = get_rays(self.target_poses, self.intrinsic, self.encoded_H, self.encoded_W)

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
