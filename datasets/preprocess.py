
"""
This file takes data from the the subdirectories in the datasets folders and
preprocesses it into img+cam pairs (reference and target).

Author: Daniel Cho
"""


import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import sys
sys.path.append(os.getcwd() + "/utils")
from graphics_utils import nerf_matrix_to_ngp


def preprocess_images(images: np.ndarray, shape=(64, 64), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Takes as input a set of images as a numpy array and applies resizing and normalization.

    inputs:
        images: a set of images as a numpy array.
        shape: the new shape to resize images.
        mean: the mean for normalization.
        std: the std for normalization.

    outputs:
        images: the images with transformations applied.
    """

    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return torch.stack([
        transform(Image.fromarray((image * 255).astype(np.uint8))) for image in images
    ])


def load_nerf_data(shape=(64, 64), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Load in the nerf dataset.

    inputs:
        shape: the new shape to resize images.
        mean: the mean for normalization.
        std: the std for normalization.
    
    outputs:
        images: preprocessed images (image set, images in set, channel, x, y).
        poses: camera matrix as a 4x4, where the last row is the identity (image set, images in set, channel, x, y).
    """

    print("[preprocess.py] WARNING: The NeRF dataset is poorly formatted/documented and has limited examples. For general testing, it is best to use the objaverse dataset. Otherwise, this function will provide 106 images/camera poses of the lego tractor.")

    # ensure directories exist
    if not os.path.isdir(os.getcwd() + "/datasets/nerf"):
        raise Exception("[preprocess.py] The nerf folder in datasets was not found. Either it does not exists or you're in the wrong directory. If the latter, run from the Stable-NeRF directory.")
    if not os.path.exists(os.getcwd() + "/datasets/nerf/tiny_nerf_data.npz"):
        raise Exception("[preprocess.py] Could not find the tiny_nerf_data.npz file. Use the dataset_nerf.ipynb notebook to download the dataset.")

    # load data
    data = np.load('datasets/nerf/tiny_nerf_data.npz', allow_pickle=True)
    images = data['images']
    poses = data['poses']

    # preprocess images
    images = preprocess_images(images, shape, mean, std)

    # preprocess poses
    poses = poses[:,:-1,:] # remove unnecessary last row
    npg_poses = []
    for i in range(len(poses)):
        npg_poses.append(nerf_matrix_to_ngp(poses[i]))
    poses = torch.from_numpy(np.array(npg_poses))

    return images.unsqueeze(0), poses.unsqueeze(0)


def load_objaverse_data(shape=(64, 64), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Load in the objaverse dataset.

    inputs:
        shape: the new shape to resize images.
        mean: the mean for normalization.
        std: the std for normalization.
    
    outputs:
        images: preprocessed images (image set, images in set, channel, x, y).
        poses: camera matrix as a 4x4, where the last row is the identity (image set, images in set, channel, x, y).
    """

    dataset_path = os.getcwd() + "/datasets/objaverse/views_release"

    # ensure dataset directory exists
    if not os.path.isdir(dataset_path):
        raise Exception("[preprocess.py] The objaverse folder in datasets was not found. Either it does not exists or you're in the wrong directory. If the latter, run from the Stable-NeRF directory.")

    # load in data from dataset
    image_sets, pose_sets = [], []
    for image_set_path in os.listdir(dataset_path):
        try:
            images, poses = [], []
            for i in range(12):
                image = Image.open(f"{dataset_path}/{image_set_path}/{i:03d}.png")
                image = np.array(image.convert("RGB")) / 255.
                images.append(image)
                pose = np.load(f"{dataset_path}/{image_set_path}/{i:03d}.npy")
                poses.append(nerf_matrix_to_ngp(pose))
            image_sets.append(np.array(images))
            pose_sets.append(np.array(poses))
        except:
            continue
        
    # preprocess images
    preprocessed_image_sets = []
    for image_set in image_sets:
        preprocessed_image_sets.append(preprocess_images(image_set, shape=shape, mean=mean, std=std))
    preprocessed_image_sets = torch.stack(preprocessed_image_sets)

    # preprocess poses
    pose_sets = np.array(pose_sets)

    return preprocessed_image_sets, pose_sets


def load_data(dataset="objaverse", shape=(64, 64), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Preprocesses data into img+cam pairs (reference and target).
    
    inputs:
        dataset (str): which dataset to get images from ["nerf", "objaverse"]
        shape: the new shape to resize images.
        mean: the mean for normalization.
        std: the std for normalization.
    
    outputs:
        images: preprocessed images as a pytorch tensor (image set, images in set, channel, image).
        poses: camera matrix as a 4x4 pytorch tensor, where the last row is the identity (image set, images in set, camera matrix).
    """

    if dataset == "nerf":
        return load_nerf_data(shape, mean, std)
    elif dataset == "objaverse":
        return load_objaverse_data(shape, mean, std)
    else:
        raise Exception(f'[preprocess.py] Dataset "{dataset} not found. Select from ["nerf", "objaverse"]".')
    

if __name__ == "__main__":
    """
    Testing functions work..
    """

    print("[preprocess.py] Testing")
    images, poses = load_data(dataset="objaverse")
    print(images.shape)
    print(poses.shape)