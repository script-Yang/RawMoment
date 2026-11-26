# SPDX-License-Identifier: Apache-2.0
"""Qwen 2.1 block execution with an independently projected RAW image prefix.

Keep the native blocks, RoPE, attention processors and timestep modulation.
Assemble RAW, LQ and noisy tokens before running the native transformer blocks.
"""

import torch


def _axis_rotary(positions, dimension, theta):
    frequencies = 1 / torch.pow(
        theta,
        torch.arange(0, dimension, 2, device=positions.device, dtype=torch.float32) / dimension,
    )
    angles = torch.outer(positions, frequencies)
    return torch.polar(torch.ones_like(angles), angles)


def spatial_rotary_embedding(transformer, image_mask, text_length, height, width, spatial_shape):
    """Give each RAW patch its center in the full LQ spatial coordinate system."""
    spatial_height, spatial_width = spatial_shape
    shapes = [(1, spatial_height, spatial_width), (1, height, width), (1, height, width)]
    rope = transformer.pos_embed
    native = rope(shapes, image_mask, device=image_mask.device)
    if spatial_shape == (height, width):
        return native
    rows = (
        (torch.arange(spatial_height, device=image_mask.device) + 0.5) * (height / spatial_height)
        - 0.5
        - (height - height // 2)
    )
    columns = (
        (torch.arange(spatial_width, device=image_mask.device) + 0.5) * (width / spatial_width)
        - 0.5
        - (width - width // 2)
    )
    spatial_tokens = spatial_height * spatial_width
    raw_rope = torch.cat(
        [
            native[text_length : text_length + spatial_tokens, : rope.axes_dim[0] // 2],
            _axis_rotary(rows.repeat_interleave(spatial_width), rope.axes_dim[1], rope.theta),
            _axis_rotary(columns.repeat(spatial_height), rope.axes_dim[2], rope.theta),
        ],
        dim=-1,
    )
    return torch.cat([native[:text_length], raw_rope, native[text_length + spatial_tokens :]])


def spatial_dit_forward(
    transformer,
    spatial_tokens,
    lq_tokens,
    noisy_tokens,
    context,
    sigma,
    height,
    width,
    spatial_shape=None,
):
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21FlexAttnProcessor,
        _qwenimage21_prefix_segments,
        build_qwenimage21_block_causal_mask,
    )

    batch_size, image_tokens, _ = lq_tokens.shape
    spatial_shape = spatial_shape or (height, width)
    spatial_height, spatial_width = spatial_shape
    dtype = transformer.img_in.weight.dtype
    text = transformer.txt_in(context.to(dtype))
    lq = transformer.img_in(lq_tokens.to(dtype))
    noisy = transformer.img_in(noisy_tokens.to(dtype))
    hidden = torch.cat([text, spatial_tokens.to(dtype), lq, noisy], dim=1)
    image_mask = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
    image_mask[text.shape[1] :] = True
    shapes = [(1, spatial_height, spatial_width), (1, height, width), (1, height, width)]
    rotary_emb = spatial_rotary_embedding(
        transformer, image_mask, text.shape[1], height, width, spatial_shape
    )
    image_ids, target_mask = transformer.build_token_metadata(image_mask, shapes)
    timestep = sigma.to(dtype)
    modulation_mask = None
    if transformer.config.causal_condition:
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        modulation_mask = target_mask
    temb = transformer.time_text_embed(timestep, hidden)
    modulation = transformer.modulation(temb)
    processors = [block.attn.processor for block in transformer.transformer_blocks]
    attention_mask = (
        build_qwenimage21_block_causal_mask(image_ids, None, batch_size, hidden.device)
        if any((isinstance(processor, QwenImage21FlexAttnProcessor) for processor in processors))
        else None
    )
    prefix_len = hidden.shape[1] - image_tokens
    segments = (
        None
        if all((isinstance(processor, QwenImage21FlexAttnProcessor) for processor in processors))
        else _qwenimage21_prefix_segments(image_ids, prefix_len)
    )
    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            hidden = transformer._gradient_checkpointing_func(
                block,
                hidden,
                modulation,
                rotary_emb,
                attention_mask,
                modulation_mask,
                None,
                None,
                None,
                segments,
                None,
            )
        else:
            hidden = block(
                hidden_states=hidden,
                modulation=modulation,
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
                target_token_mask=modulation_mask,
                segments=segments,
            )
    target = hidden[:, -image_tokens:]
    target_modulation = modulation_mask[-image_tokens:] if modulation_mask is not None else None
    return transformer.proj_out(transformer.norm_out(target, temb, target_modulation))
