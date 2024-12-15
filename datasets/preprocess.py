
"""
This file takes data from the the subdirectories in the datasets folders and
preprocesses it into img+cam pairs (reference and target).

Author: Daniel Cho
"""


import os
import psutil
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.append(os.getcwd() + "/utils")
from graphics_utils import nerf_matrix_to_ngp


def construct_normalized_camera_intrinsics(image_shape, focal_length=50., skew=0.):
    """
    Returns a normalized camera intrinsics matrix.

    inputs:
        image_shape: image shape in pixels
        focal_length: focal length in mm
        skew: skew
    """

    sensor_width_mm = 36. # default blender value
    focal_length_pixels = focal_length * (image_shape[0] / sensor_width_mm)

    return torch.tensor([
        [focal_length_pixels / image_shape[0], skew, 0.5],
        [0, focal_length_pixels / image_shape[1], 0.5],
        [0, 0, 1.],
    ])


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
        poses: camera matrix as a 4x4 (image set, images in set, channel, x, y).
        intrinsics: camera intrinsics matrix as a 3x3
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
    focal = data['focal']

    # preprocess images
    images = preprocess_images(images, shape, mean, std)

    # preprocess poses
    poses = poses[:,:-1,:] # remove unnecessary last row
    npg_poses = []
    for i in range(len(poses)):
        npg_poses.append(nerf_matrix_to_ngp(poses[i]))
    poses = torch.from_numpy(np.array(npg_poses))

    # normalized camera intrinsics matrix
    intrinsic = construct_normalized_camera_intrinsics(shape, focal)

    return images, poses, intrinsic


def load_objaverse_data(shape=(64, 64), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], fix_choices=(0, 1), percent_objects=0.001):
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
    assert percent_objects > 0 and percent_objects <= 1, "percent_objects must be between 0 and 1."
    if fix_choices is not None:
        assert len(fix_choices) == 2, "fix_choices must be a tuple of 2 integers."
        assert fix_choices[0] >= 0 and fix_choices[0] < 12, "fix_choices[0] must be between 0 and 11."
        assert fix_choices[1] >= 0 and fix_choices[1] < 12, "fix_choices[1] must be between 0 and 11."

    dataset_path = os.getcwd() + "/datasets/objaverse/views_release"
    # dataset_path = '/users/earslan/scratch/stable_nerf_data/objaverse/views_release'

    # ensure dataset directory exists
    if not os.path.isdir(dataset_path):
        raise Exception("[preprocess.py] The objaverse folder in datasets was not found. Either it does not exists or you're in the wrong directory. If the latter, run from the Stable-NeRF directory.")

    # load in data from dataset
    def process_image_set(image_set_path, fix_choices):
        try:
            images, poses = [], []
            # only choose 2 random poses out of 12 to load per object
            # if fix_choices is not None:
            #     choices = np.array(fix_choices)
            # else:
            choices = np.random.choice(12, 2, replace=False)
            for i in choices:
                image = Image.open(f"{dataset_path}/{image_set_path}/{i:03d}.png")
                image = np.array(image.convert("RGB")) / 255.
                images.append(image)
                pose = np.load(f"{dataset_path}/{image_set_path}/{i:03d}.npy")
                poses.append(nerf_matrix_to_ngp(pose))
            images = preprocess_images(np.array(images), shape=shape, mean=mean, std=std)
            return images, np.array(poses)
        except:
            return None, None

    image_sets, pose_sets = [], []
    process = psutil.Process(os.getpid())  # Current process for memory tracking
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_image_set, image_set_path, fix_choices): image_set_path
            # for image_set_path in os.listdir(dataset_path)[:int(len(os.listdir(dataset_path)) * percent_objects)]
            # for image_set_path in ["0d2872e2a3474ec899c4fdd009701dca", "0d87376dea3240c8a17f0e7b2275ee80"]
            for image_set_path in ["0d87376dea3240c8a17f0e7b2275ee80", "0d87376dea3240c8a17f0e7b2275ee80", "0d87376dea3240c8a17f0e7b2275ee80"]
        }

        initial_memory = process.memory_info().rss / (1024 ** 3)  # Initial memory in GB
        progress_bar = tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True)
        for future in progress_bar:
            images, poses = future.result()

            if images is not None and poses is not None:
                image_sets.append(images)
                pose_sets.append(poses)

            current_memory = process.memory_info().rss / (1024 ** 3)
            memory_diff = current_memory - initial_memory
            progress_bar.set_description(f"Loading data | Memory Increase: {memory_diff:.2f} GB")

    # preprocess poses
    print(f"Loaded {len(image_sets)} objects and {len(pose_sets)} poses with 2 poses each.")
    pose_sets = np.array(pose_sets)
    pose_sets = torch.from_numpy(pose_sets)
    preprocessed_image_sets = torch.stack(image_sets)

    # normalized camera intrinsics matrix
    intrinsic = construct_normalized_camera_intrinsics(shape)

    return preprocessed_image_sets, pose_sets, intrinsic


def load_data(dataset="objaverse", shape=(64, 64), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], fix_choices=(0, 1), percent_objects=0.1):
    """
    Preprocesses data into img+pose groups (used as references and target).
    
    inputs:
        dataset (str): which dataset to get images from ["nerf", "objaverse"]
        shape: shape to resize images.
        mean: mean for image normalization.
        std: std for image normalization.
    
    outputs:
        images: preprocessed images as a pytorch tensor (image set, image, channel, pixels).
        poses: rotation/translation matrix (image set, image, rotation/translation matrix).

    """

    if dataset == "nerf":
        return load_nerf_data(shape, mean, std)
    elif dataset == "objaverse":
        return load_objaverse_data(shape, mean, std, fix_choices, percent_objects)
    else:
        raise Exception(f'[preprocess.py] Dataset "{dataset} not found. Select from ["nerf", "objaverse"]".')
    

if __name__ == "__main__":
    """
    Testing.
    """

    print("[preprocess.py] Testing")
    images, poses = load_data(dataset="objaverse")
    print(images.shape)
    print(poses.shape)