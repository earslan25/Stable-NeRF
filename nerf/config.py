import numpy as np
from typing import NamedTuple


class EncodingConfigHG(NamedTuple):
    otype: str
    n_levels: int
    n_features_per_level: int
    log2_hashmap_size: int
    base_resolution: int
    per_level_scale: float


class EncodingConfigSH(NamedTuple):
    otype: str
    degree: int


class NetworkConfig(NamedTuple):
    otype: str
    activation: str
    output_activation: str
    n_neurons: int
    n_hidden_layers: int


class NeRFConfig(NamedTuple):
    encoding_sigma: EncodingConfigHG
    network_sigma: NetworkConfig
    encoding_dir: EncodingConfigSH
    network_color: NetworkConfig


class BaseNeRFConfig(NeRFConfig):
    encoding_sigma=EncodingConfigHG(
        otype="HashGrid",
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=np.exp2(np.log2(2048 / 16) / (16 - 1)),
    )
    network_sigma=NetworkConfig(
        otype="FullyFusedMLP",
        activation="ReLU",
        output_activation="None",
        n_neurons=128,
        n_hidden_layers=3, # 4 - 1
    )
    encoding_dir=EncodingConfigSH(
        otype="SphericalHarmonics",
        degree=4,
    )
    network_color=NetworkConfig(
        otype="FullyFusedMLP",
        activation="ReLU",
        output_activation="None",
        n_neurons=128,
        n_hidden_layers=5, # 6 - 1
    )
        


        