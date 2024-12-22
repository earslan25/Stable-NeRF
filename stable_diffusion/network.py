import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPTextModel, \
    CLIPTokenizer, CLIPTextModelWithProjection, CLIPImageProcessor, AutoTokenizer
from utils.sd_utils import encode_prompt

from .ip_adapter.ip_adapter import ImageProjModel
from .ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from .ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class SDNetwork(torch.nn.Module):

    def __init__(self, pretrained_models_path, image_encoder_path, use_downsampling_layers=False, embed_cache_device='cuda'):
        super(SDNetwork, self).__init__()
        # init vae from pretrained
        self.vae = AutoencoderKL.from_pretrained(pretrained_models_path, subfolder="vae")
        self.vae.requires_grad_(False)
        # init unet from pretrained 
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_models_path, subfolder="unet")
        # self.unet.config.addition_embed_type = "text" 
        self.unet.requires_grad_(False)

        self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_models_path, subfolder="scheduler")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        self.image_encoder.requires_grad_(False)
        self.clip_image_processor = T.Resize((self.image_encoder.config.image_size, self.image_encoder.config.image_size), antialias=None)  # CLIPImageProcessor()  
        self.use_downsampling_layers = use_downsampling_layers
        self.init_ip_modules()

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_models_path, subfolder="tokenizer", use_fast=False
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            pretrained_models_path, subfolder="tokenizer_2", use_fast=False
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_models_path, subfolder="text_encoder", torch_dtype=torch.float32
        ).to(embed_cache_device)
        text_encoder.requires_grad_(False)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_models_path, subfolder="text_encoder_2", torch_dtype=torch.float32
        ).to(embed_cache_device)
        text_encoder_2.requires_grad_(False)

        self.init_empty_prompts(tokenizer, text_encoder, tokenizer_2, text_encoder_2, embed_cache_device)
        
    def init_ip_modules(self):
        self.num_tokens = 2
        proj_dim = (4 + 3) * (64 ** 2)  # 4 from latent image, 3 from plucker coordinates
        # for nerfless tests
        # proj_dim = (4 + 3 + 3) * (64 ** 2)  # 4 from latent image, 3 from plucker coordinates, 3 from plucker coordinates

        self.downsampling_layers = None
        if self.use_downsampling_layers:
            # CNN downsampling before the image projection, from 7x64x64
            print("using downsampling layers")
            self.downsampling_layers = torch.nn.Sequential(
                torch.nn.Conv2d(7, 16, kernel_size=4, stride=2, padding=1),  # [B, 7, 64, 64] -> [B, 16, 32, 32]
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [B, 16, 32, 32] -> [B, 32, 16, 16]
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=4, padding=0),  # [B, 32, 16, 16] -> [B, 64, 4, 4]
                torch.nn.ReLU(),
            )
            proj_dim = 64 * 4 * 4

            # for nerfless tests, hardcoded for now
            # self.downsampling_layers = torch.nn.Sequential(
            #     torch.nn.Conv2d(10, 16, kernel_size=4, stride=2, padding=1),  # [B, 7, 64, 64] -> [B, 16, 32, 32]
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [B, 16, 32, 32] -> [B, 32, 16, 16]
            #     torch.nn.ReLU(),
            #     torch.nn.Conv2d(32, 64, kernel_size=4, stride=4, padding=0),  # [B, 32, 16, 16] -> [B, 64, 4, 4]
            #     torch.nn.ReLU(),
            # )
            # proj_dim = 64 * 4 * 4

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=proj_dim,
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
    def init_empty_prompts(self, tokenizer, text_encoder, tokenizer_2, text_encoder_2, device):
        missing_prompt_pos = ""
        missing_prompt_neg = ""

        (
            prompt_embeds, 
            negative_prompt_embeds, 
            pooled_prompt_embeds, 
            negative_pooled_prompt_embeds
        ) = encode_prompt(
            prompt=missing_prompt_pos,
            prompt_2=missing_prompt_pos,
            device=device,
            negative_prompt=missing_prompt_neg,
            negative_prompt_2=missing_prompt_neg,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
        )

        # no neg prompts

        add_text_embeds = pooled_prompt_embeds

        resolution = 1024
        crops_coords_top_left = (0,0)
        original_sizes = (resolution, resolution)
        crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)
        original_sizes = torch.tensor(original_sizes, dtype=torch.long)
        crops_coords_top_left = crops_coords_top_left.repeat(len(prompt_embeds), 1)
        original_sizes = original_sizes.repeat(len(prompt_embeds), 1)

        target_size = (resolution, resolution)
        add_time_ids = list(target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_embeds), 1)
        add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
        add_time_ids = add_time_ids.to(device, dtype=torch.float32)
        # negative_add_time_ids = add_time_ids

        # no negative prompts
        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        # add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        # add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        
        prompt_embeds = prompt_embeds.to(device, dtype=torch.float32)
        add_text_embeds = add_text_embeds.to(device)

        self.prompt_embeds = prompt_embeds
        self.add_text_embeds = add_text_embeds
        self.add_time_ids = add_time_ids

        print("empty embeddings initialized")

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

    def forward(self, noisy_latents, timesteps, added_cond_kwargs, image_embeds):
        if self.use_downsampling_layers:
            image_embeds = self.downsampling_layers(image_embeds)

        seq = 2
        bs = image_embeds.shape[0] // seq
        hidden_state_dim = image_embeds.shape[-1] * image_embeds.shape[-2] * image_embeds.shape[-3]
        # change from batch * 2, channel, height, width to batch * 2, channel * height * width
        image_embeds = image_embeds.view(-1, hidden_state_dim)
        ip_tokens = self.image_proj_model(image_embeds)  # batch * 2, num tokens, hidden_state_dim

        # commented out for nerfless
        ip_tokens = ip_tokens.view(bs, seq * self.num_tokens, -1)  # batch, num tokens * 2, hidden_state_dim

        # empty prompts not used, only image condition
        # encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1) 
        encoder_hidden_states = ip_tokens 
        
        # predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, timestep_cond=None).sample

        return noise_pred
