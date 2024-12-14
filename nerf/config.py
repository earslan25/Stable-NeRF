import numpy as np
from dataclasses import dataclass, field


@dataclass
class EncodingConfigHG:
    otype: str
    n_levels: int
    n_features_per_level: int
    log2_hashmap_size: int
    base_resolution: int
    per_level_scale: float


@dataclass
class EncodingConfigSH:
    otype: str
    degree: int


@dataclass
class NetworkConfig:
    otype: str
    activation: str
    output_activation: str
    n_neurons: int
    n_hidden_layers: int


@dataclass
class NeRFConfig:
    encoding_sigma: EncodingConfigHG
    network_sigma: NetworkConfig
    encoding_dir: EncodingConfigSH
    network_color: NetworkConfig

    def as_dict(self):
        res = {}
        for k, v in self.__dict__.items():
            res[k] = v.__dict__
        
        return res


@dataclass
class BaseNeRFConfig(NeRFConfig):
    encoding_sigma: EncodingConfigHG = field(default=EncodingConfigHG(
        otype="HashGrid",
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=np.exp2(np.log2(2048 / 16) / (16 - 1)),
    ))
    network_sigma: NetworkConfig = field(default=NetworkConfig(
        otype="FullyFusedMLP",
        activation="ReLU",
        output_activation="None",
        n_neurons=64,
        n_hidden_layers=1, 
    ))
    encoding_dir: EncodingConfigSH = field(default=EncodingConfigSH(
        otype="SphericalHarmonics",
        degree=4,
    ))
    network_color: NetworkConfig = field(default=NetworkConfig(
        otype="FullyFusedMLP",
        activation="ReLU",
        output_activation="None",
        n_neurons=64,
        n_hidden_layers=2,
    ))
        


        