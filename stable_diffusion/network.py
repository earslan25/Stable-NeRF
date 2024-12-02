import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPTextModel, \
    CLIPTokenizer, CLIPTextModelWithProjection, CLIPImageProcessor

from .ip_adapter.ip_adapter import ImageProjModel
from .ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from .ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class SDNetwork(torch.nn.Module):

    def __init__(self, pretrained_models_path, image_encoder_path, channel_dim=4, cat_cam=True):
        super(SDNetwork, self).__init__()
        vae_config = AutoencoderKL.load_config(pretrained_models_path, subfolder="vae")
        vae_config['latent_channels'] = channel_dim
        self.channel_dim = channel_dim
        self.cat_cam = cat_cam
        self.vae = AutoencoderKL.from_config(vae_config)
        self.vae.requires_grad_(False)
        unet_config = UNet2DConditionModel.load_config(pretrained_models_path, subfolder="unet")
        unet_config['in_channels'] = channel_dim
        unet_config['out_channels'] = channel_dim
        unet_config['addition_embed_type'] = "text"
        self.unet = UNet2DConditionModel.from_config(unet_config)
        # self.unet.config.addition_embed_type = "text"
        self.unet.requires_grad_(False)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_models_path, subfolder="scheduler")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        self.image_encoder.requires_grad_(False)
        self.clip_image_processor = T.Resize((self.image_encoder.config.image_size, self.image_encoder.config.image_size), antialias=None)  # CLIPImageProcessor()  
        self.init_ip_modules()

        # tokenizer = CLIPTokenizer.from_pretrained(pretrained_models_path, subfolder="tokenizer")
        # text_encoder = CLIPTextModel.from_pretrained(pretrained_models_path, subfolder="text_encoder")
        # text_encoder.requires_grad_(False)
        # tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_models_path, subfolder="tokenizer_2")
        # text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_models_path, subfolder="text_encoder_2")
        # text_encoder_2.requires_grad_(False)
        # self.init_empty_prompts(tokenizer, text_encoder, tokenizer_2, text_encoder_2)
        
    def init_ip_modules(self):
        self.num_tokens = 1 # idk what this is
        if self.channel_dim == 4:
            proj_dim = self.channel_dim * (64 ** 2)
        else:
            proj_dim = self.image_encoder.config.projection_dim

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=proj_dim + 32 if self.cat_cam else proj_dim,
            clip_extra_context_tokens=self.num_tokens,
        )

        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=self.num_tokens)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())

    @torch.no_grad()
    def init_empty_prompts(self, tokenizer, text_encoder, tokenizer_2, text_encoder_2):
        missing_prompt = [""]
        text_input_ids = tokenizer(
            missing_prompt,
            max_length=self.num_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # text_input_ids_2 = tokenizer_2(
        #     missing_prompt,
        #     max_length=tokenizer_2.model_max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # ).input_ids
        # 
        # encoder_output = text_encoder(text_input_ids, output_hidden_states=True)
        # text_embeds = encoder_output.hidden_states[-2]
        # encoder_output_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
        # self.pooled_empty_text_embeds = encoder_output_2[0]
        # text_embeds_2 = encoder_output_2.hidden_states[-2]
        # self.empty_text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) 
        # self.empty_text_embeds = text_encoder(text_input_ids)[0]

    def encode_images(self, images):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        
        return images

    def clip_encode_images(self, images):
        if images.shape[2] != self.image_encoder.config.image_size and images.shape[3] != self.image_encoder.config.image_size:
            clip_images = self.clip_image_processor(images)
            image_embeds = self.image_encoder(clip_images).image_embeds
        else:
            image_embeds = self.image_encoder(images).image_embeds
        
        return image_embeds

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)

        # (batch, sequence_length, feature_dim), concatenated, the more prompts, the larger sequence_length 
        # encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=-1) 
        encoder_hidden_states = ip_tokens
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return noise_pred