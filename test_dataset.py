import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.dataset import StableNeRFDataset, collate_fn


def test_dataset():
    batch_size = 2
    fix_choices = None
    dataset = StableNeRFDataset(dataset_name='objaverse', shape=(512, 512), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], generate_cuda_ray=True, percent_objects=0.0001)
    dataset = StableNeRFDataset(dataset_name='objaverse', shape=(512, 512), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], generate_cuda_ray=True, fix_choices=fix_choices, percent_objects=0.0001)
    dataset_2 = StableNeRFDataset(dataset_name='nerf', shape=(512, 512), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], generate_cuda_ray=False)
    print(dataset.intrinsic.shape)
    print(dataset.reference_images.shape)
    print(dataset.reference_poses.shape)
    print(dataset.reference_rays['rays_o'].shape)
    print(dataset.reference_rays['rays_d'].shape)
    print(dataset.reference_rays['inds'].shape)
    print(dataset.target_images.shape)
    print(dataset.target_poses.shape)
    print(dataset.target_rays['rays_o'].shape)
    print(dataset.target_rays['rays_d'].shape)
    print(dataset.target_rays['inds'].shape)

    # save sample image to debug_out/
    plt.imsave("debug_out/sample_image_11.png", dataset.reference_images[0].permute(1, 2, 0).numpy())
    plt.imsave("debug_out/sample_image_12.png", dataset.target_images[0].permute(1, 2, 0).numpy())
    plt.imsave("debug_out/sample_image_21.png", dataset_2.reference_images[0].permute(1, 2, 0).numpy())
    plt.imsave("debug_out/sample_image_22.png", dataset_2.target_images[0].permute(1, 2, 0).numpy())

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=1
    )
    for batch in train_dataloader:
        for key in batch.keys():
            print(key, batch[key].shape)
        break


if __name__ == '__main__':
    test_dataset()