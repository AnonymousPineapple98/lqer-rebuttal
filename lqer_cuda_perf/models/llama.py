import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_attention_shapes
from ..module import get_linear, FastRMSNorm

from transformers.models.llama.modeling_llama import ACT2FN


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, device, rope_theta):
        super(RoPE, self).__init__()

        self.freqs_cis = nn.Parameter(
            self.precompute_freqs_cis(head_dim, max_seq_len * 2, rope_theta).to(device),
            requires_grad=False,
        )

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int, seqlen: int):
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        )
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        )
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_).to(xq_.device)

        xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-2, -1).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-2, -1).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaQAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        device,
        max_seq_len=2048,
        attention_shapes=None,
        rope_theta=10000,
        head_dim=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_groups = n_heads // n_kv_heads if n_kv_heads != 0 else 0
        self.head_dim = head_dim

        self.cache_batch_size = 0  # *: no cache support for now

        if head_dim is None:
            self.head_dim = hidden_size // n_heads
        self.qkv_proj = qkv_layer
        self.o_proj = o_proj
        self.start_pos = 0

        if kwargs.get("max_new_tokens") is not None:
            max_seq_len = kwargs["max_new_tokens"]

        self.max_seq_len = max_seq_len
        self.is_hf_transformers = False
        self.rope_theta = rope_theta

        self.attention_shapes = get_attention_shapes(
            attention_shapes,
            max_seq_len,
            1,
            n_heads,
            n_kv_heads,
            self.head_dim,
        )

        self.alibi = None
        self.rope = RoPE(self.head_dim, max_seq_len, device, rope_theta)
        self.rotary_dim = self.head_dim
        self.is_neox = True

    def forward(
        self, hidden_states: torch.Tensor, attention_mask=None, *args, **kwargs
    ):
        bsz, seqlen, _ = hidden_states.shape
        if bsz != self.cache_batch_size:
            self.start_pos = 0

        xqkv = self.qkv_proj(hidden_states)
        xqkv = xqkv.view((bsz, seqlen) + self.attention_shapes["xqkv_view"])

        xq = self.attention_shapes["xq_slice"](xqkv)
        xk = self.attention_shapes["xk_slice"](xqkv)
        xv = self.attention_shapes["xv_slice"](xqkv)

        xq = xq.view((bsz, seqlen) + self.attention_shapes["xq_view"])
        xk = xk.view((bsz, seqlen) + self.attention_shapes["xk_view"])
        xv = xv.view((bsz, seqlen) + self.attention_shapes["xv_view"])
        xq, xk = self.rope.forward(xq, xk, self.start_pos, seqlen)

        keys = xk
        values = xv

        if self.n_kv_groups != 0:
            keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_kv_groups)
            values = torch.repeat_interleave(values, dim=2, repeats=self.n_kv_groups)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # When seqlen is 1, there is nothing else to attend to
        if attention_mask is not None and seqlen > 1:
            scores = (
                scores + attention_mask
            )  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        attention_weight = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        attn_output = self.o_proj(attention_weight)
        self.start_pos += seqlen

        # past_key_value is replaced with cache_v, cache_k, returning empty data
        # we pass a dummy past kv cache for transformers to be able to retrieve the correct info
        # about past key length
        past_key_value = [torch.zeros(1, 1, self.start_pos, 1)]

        return attn_output, attention_weight, past_key_value


class LlamaQMLP(nn.Module):
    def __init__(self, config, q_recipe, rank: int, device):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = get_linear(
            q_recipe["name"],
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=device,
            rank=rank,
        )
        self.up_proj = get_linear(
            q_recipe["name"],
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=device,
            rank=rank,
        )
        self.down_proj = get_linear(
            q_recipe["name"],
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            device=device,
            rank=rank,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaQDecoderLayer(torch.nn.Module):
    def __init__(self, config, layer_idx: int, q_recipe, rank: int, device) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        qkv_layer = get_linear(
            q_recipe["name"],
            config.hidden_size,
            self.num_heads * self.head_dim
            + self.num_key_value_heads * 2 * self.head_dim,
            device=device,
            bias=config.attention_bias,
            rank=rank,
        )

        o_proj = get_linear(
            q_recipe["name"],
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
            device=device,
            rank=rank,
        )
        self.attn = LlamaQAttention(
            hidden_size=self.hidden_size,
            n_heads=self.num_heads,
            n_kv_heads=self.num_key_value_heads,
            qkv_layer=qkv_layer,
            o_proj=o_proj,
            device=device,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            head_dim=self.head_dim,
        )
        self.mlp = LlamaQMLP(config, q_recipe=q_recipe, rank=rank, device=device)

        self.norm_1 = FastRMSNorm(
            self.hidden_size, config.rms_norm_eps, dtype=torch.float16, device=device
        )
        self.norm_2 = FastRMSNorm(
            self.hidden_size, config.rms_norm_eps, dtype=torch.float16, device=device
        )

    def forward(
        self, hidden_states, past_key_value=None, attention_mask=None, **kwargs
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.mlp.forward(self.norm_2(h))

        return out, None, past_key_value
