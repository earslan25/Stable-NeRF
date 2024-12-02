import torch
from datasets.dataset import StableNeRFDataset, collate_fn


def test_dataset():
    batch_size = 2
    dataset = StableNeRFDataset(dataset_name='nerf', shape=(512, 512), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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