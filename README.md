# Stable-NeRF

## Project Overleaf
[link](https://www.overleaf.com/9285862251qcvdcnhcnbmx#a783ac)

## Datasets
- Original NERF dataset ([link](https://www.kaggle.com/datasets/sauravmaheshkar/nerf-dataset))
- Limited objects, but many views ([link](https://cvg.cit.tum.de/data/datasets/3dreconstruction))
- Github repo of multiple datasets (unclear if there's positional data, [link](https://github.com/KunyuLin/Multi-view-Datasets))
- Large scale data for NERF training (3.2 TB, [link](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet))
- Zero-1-to-3 dataset

## One Page Proposal

What is the exact problem you are trying to solve? 
- We want to address 3D reconstruction using a combination of stable diffusion and NeRF. By inserting a model capable of a 3D representation within stable diffusion, we hope to create a more robust 3D reconstruction. 

What prior works have tried to address it? 
- [Reconstructive Latent-Space Neural Radiance Fields for Efficient 3D Scene Representations](https://arxiv.org/pdf/2310.17880)

How is your approach different? 
- Zero-1-to-3 doesn't have 3D.
- Reconstructive paper doesn't have stable diffusion.

What data will you use? 
- Start with Zero-1-to-3 dataset (drum set, fire hidrant)

What compute will you use? 
- Oscar (Chia)
- Possibly Kaggle, dual T4's

What existing codebases will you use?
- Stable diffusion
- NeRF or tiny-cuda-nn/instant-ngp
