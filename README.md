# Stable-NeRF
This repository contains the code for the research project Stable NeRF. The project aims use 2D priors from Stable Diffusion and the learned 3D forward map from NeRF to perform generalizable novel view synthesis. Our initial proposal and current report can be found in the reports folder.

Contributors: Emre Arslan (earslan25), Chia-Hong Hsu (swimmincatt35), Daniel Cho (hypochoco)

## Conda Env Notes:
- Currently, directly creating from the environment.yml file is not working or is too slow. Instead, create a new environment with python=3.8 and install the packages manually.
- To install the packages manually, run the following commands:
1. When using oscar, run these lines first.
```bash
module purge
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
module load cuda/11.8.0-lpttyok
```
2. Then, run the following lines.
```bash
conda create -n stable_nerf python=3.8
conda activate stable_nerf
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install submodules/raymarching
```
3. Download datasets. They can also be downloaded by running the Jupyter notebooks in the datasets folder.
```bash
curl https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz -o datasets/nerf/tiny_nerf_data.npz
```

## Datasets
- Original NERF dataset ([link](https://www.kaggle.com/datasets/sauravmaheshkar/nerf-dataset))
- Objaverse ([link](https://objaverse.allenai.org/))

Initially, we used a synthetic dataset to assess our method's performance. Given that baseline, we will attempt to extend our approach to generalize to real world views. Our method can benefit from most multiview datasets as we only need 2 views per scene.


## TODO
- [x] Implement argument parser to handle different configurations
- [x] Integrate per-pixel features from encoded images into NeRF


## Credits
We would like to thank the following repositories for their code and inspiration:
- torch-ngp by Jiaxiang Tang ([link](https://github.com/ashawkey/torch-ngp))
- IP-Adapter by Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, Wei Yang ([link](https://ip-adapter.github.io/))






