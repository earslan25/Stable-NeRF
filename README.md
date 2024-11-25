# Stable-NeRF

## Conda Env Notes:
- Currently, directly creating from the environment.yml file is not working. Instead, create a new environment with python=3.8 and install the packages manually.
- To install the packages manually, run the following commands:
```bash
conda create -n stable_nerf python=3.8
conda activate stable_nerf
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Project Overleaf
[link](https://www.overleaf.com/9285862251qcvdcnhcnbmx#a783ac)

## Datasets
- Original NERF dataset ([link](https://www.kaggle.com/datasets/sauravmaheshkar/nerf-dataset))
- Limited objects, but many views ([link](https://cvg.cit.tum.de/data/datasets/3dreconstruction))
- Github repo of multiple datasets (unclear if there's positional data, [link](https://github.com/KunyuLin/Multi-view-Datasets))
- Large scale data for NERF training (3.2 TB, [link](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet))
- Zero-1-to-3 dataset

Initially, we will be using a synthetic dataset to assess our method's performance. Given that baseline, we will attempt to extend our approach to generalize to real world views. Our method can benefit from most multiview datasets as we only need 2 views per scene.
