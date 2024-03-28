import torch


def get_attention_shapes(
    attention_shapes, max_seq_len, cache_batch_size, n_heads, n_kv_heads, head_dim
):
    if attention_shapes is not None:
        attention_shapes = attention_shapes

    elif n_kv_heads == 0:
        attention_shapes = {
            # following fastertransformer definition
            "cache_v": (
                cache_batch_size,
                n_heads,
                max_seq_len,
                head_dim,
            ),
            # 8: pack 8 fp16 in FT, if fp32 then use 4
            "cache_k": (
                cache_batch_size,
                n_heads,
                head_dim // 8,
                max_seq_len,
                8,
            ),
            "xqkv_view": (-1, n_heads, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0],
            "xk_slice": lambda xqkv: xqkv[:, :, 1],
            "xv_slice": lambda xqkv: xqkv[:, :, 2],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_heads, head_dim),
            "xv_view": (n_heads, head_dim),
            "xk_reshape": (n_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_heads, head_dim),
            "single_xv_view": (n_heads, head_dim),
        }

    else:
        attention_shapes = {
            # following fastertransformer definition
            "cache_v": (
                cache_batch_size,
                n_kv_heads,
                max_seq_len,
                head_dim,
            ),
            # 8: pack 8 fp16 in FT, if fp32 then use 4
            "cache_k": (
                cache_batch_size,
                n_kv_heads,
                head_dim // 8,
                max_seq_len,
                8,
            ),
            "xqkv_view": (n_heads + n_kv_heads * 2, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0:n_heads],
            "xk_slice": lambda xqkv: xqkv[:, :, n_heads : (n_heads + n_kv_heads)],
            "xv_slice": lambda xqkv: xqkv[:, :, -n_kv_heads:],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_kv_heads, head_dim),
            "xv_view": (n_kv_heads, head_dim),
            "xk_reshape": (n_kv_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_kv_heads, head_dim),
            "single_xv_view": (n_kv_heads, head_dim),
        }

    return attention_shapes


def prepare_input_ids(input_ids: torch.Tensor, last_forward_num_tokens: int):
    # NOTE: from transformers 4.35.0, input_ids includes full context during decoding
    num_input_tokens = input_ids.shape[-1]
    num_new_tokens = num_input_tokens

    if num_input_tokens != 1:
        num_new_tokens = num_input_tokens - last_forward_num_tokens

        # after context is processed, slice to latest token
        if num_new_tokens == 1:
            input_ids = input_ids[:, -1:]

    return input_ids, last_forward_num_tokens + num_new_tokens


def prepare_attention_mask(seqlen, start_pos, device, type_as: torch.Tensor):
    mask = None
    if seqlen > 1:
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(type_as)

    return mask


def prepare_correct_devices(next_layer, hidden_states, mask):
    device = next(next_layer.parameters()).device
    hidden_states = hidden_states.to(device)

    if mask is not None:
        mask = mask.to(device)

    return hidden_states, mask
