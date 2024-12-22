import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from torchvision.transforms.functional import to_pil_image
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

'''
[CH] SDXL text_to_image full pipeline.
[CH] This function shows how each component is loaded, e.g., tokenizer, text_encoder, vae, unet.
[CH] TODO image_to_image full pipeline.
'''
def generate_image_with_sdxl_dual_encoders(
    prompt,
    prompt_2 = None,
    negative_prompt = None,
    negative_prompt_2 = None,
    tokenizer=None,
    tokenizer_2=None,
    text_encoder=None,
    text_encoder_2=None,
    vae=None,
    unet=None,
    scheduler= None,
    output_file = "generated_image_sdxl_dual.png",
    num_steps = 50,
    scaling_factor = None,
    seed = 42,
    device = torch.device("cuda"),
    guidance_scale = 1.0
):

    # Set device and random seed
    torch.manual_seed(seed)

    # [CH] Set VAE scaling factor, *scale up* the latent before passing to vae decoder
    scaling_factor = scaling_factor or vae.config.scaling_factor

    # [CH] Encode text prompts
    ( prompt_embeds, 
      negative_prompt_embeds, 
      pooled_prompt_embeds, 
      negative_pooled_prompt_embeds
    ) = encode_prompt(prompt=prompt,
                      prompt_2=prompt_2,
                      device=device,
                      negative_prompt=negative_prompt,
                      negative_prompt_2=negative_prompt_2,
                      tokenizer = tokenizer,
                      tokenizer_2 = tokenizer_2,
                      text_encoder = text_encoder,
                      text_encoder_2 = text_encoder_2,
                      )

    add_text_embeds = pooled_prompt_embeds

    ''' 
    [CH] This part is a mystery... mystery start
    '''
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
    add_time_ids = add_time_ids.to(device, dtype=torch.float16)
    # [CH] print("add_time_ids",add_time_ids) # [[1024., 1024.,    0.,    0., 1024., 1024.]]
    negative_add_time_ids = add_time_ids
    '''
    [CH] Mystery end
    '''

    # do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    
    prompt_embeds = prompt_embeds.to(device, dtype=torch.float16)
    add_text_embeds = add_text_embeds.to(device)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # Initialize latent variables
    latent_shape = (
        1,
        unet.in_channels,
        unet.sample_size,
        unet.sample_size,
    )
    latents = torch.randn(latent_shape, device=device, dtype=torch.float16)

    # Denoising process
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    for t in timesteps:
        with torch.no_grad():
            latents_model_input = torch.cat([latents] * 2)

            noise_pred = unet(
                latents_model_input, 
                t, 
                timestep_cond=None,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample
    latents = latents.float()

    # Decode latents into an image
    with torch.no_grad():
        # Scaling factor for unit variance
        latents = latents / scaling_factor 
        image_tensor = vae.decode(latents).sample[0]

    # Convert to PIL image and save
    image = to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
    image.save(output_file)
    print(f"Image saved to {output_file}")

'''
[CH] Conditioning on text. Research on conditioning on image later. 
'''
def encode_prompt(
        prompt,
        prompt_2 = None,
        device = None,
        num_images_per_prompt = 1,
        do_classifier_free_guidance = True,
        negative_prompt = None,
        negative_prompt_2 = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        pooled_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
        clip_skip = None,
        tokenizer = None,
        tokenizer_2 = None,
        text_encoder = None,
        text_encoder_2 = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        assert text_encoder is not None
        assert text_encoder_2 is not None
        assert tokenizer is not None
        assert tokenizer_2 is not None

        device = device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [tokenizer, tokenizer_2] 
        text_encoders = (
            [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]

            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                prompt = maybe_convert_prompt(prompt, tokenizer)
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None 
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                negative_prompt = maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=unet.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def maybe_convert_prompt(prompt, tokenizer):  
    if not isinstance(prompt, List):
        prompts = [prompt]
    else:
        prompts = prompt
    prompts = [_maybe_convert_prompt(p, tokenizer) for p in prompts]
    if not isinstance(prompt, List):
        return prompts[0]
    return prompts

def _maybe_convert_prompt(prompt, tokenizer):  
    tokens = tokenizer.tokenize(prompt)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1
            prompt = prompt.replace(token, replacement)
    return prompt

if __name__ == "__main__":

    # Load components
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch.float16
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    
    # Condition 1
    prompt = "A futuristic cityscape at sunset, highly detailed, vibrant colors"
    # Condition 2
    prompt_2 = "Cyberpunk style, glowing neon lights, wide-angle perspective"
    # Condition 3
    negative_prompt = "tall buildings"
    # Condition 4
    negative_prompt_2 = "rainy day"

    # Condition 1,2,3,4
    generate_image_with_sdxl_dual_encoders(
        prompt,
        prompt_2,
        negative_prompt,
        negative_prompt_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        num_steps=50,  # Increased steps for better quality
        scaling_factor=0.18215,  # Example scaling factor
        guidance_scale=10.0,
        device=device,
        output_file="1234.png",
    )
