import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from .utils import prepare_input_ids, prepare_attention_mask, prepare_correct_devices
from ..module import FastRMSNorm


class QDecoder(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = layers

    def forward(
        self, hidden_states, past_key_value=None, attention_mask=None, **kwargs
    ):
        past_key_values = []
        for l in self.layers:
            output_hidden_states, _, past_key_value = l(
                hidden_states, past_key_value, attention_mask
            )
            hidden_states = output_hidden_states
            past_key_values.append(past_key_value)

        return hidden_states


class QLlamaModel(nn.Module):
    def __init__(
        self, config, decoder: QDecoder, dtype=torch.float16, device="cuda"
    ) -> None:
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=dtype,
            device=device,
        )

        self.decoder = decoder
        self.norm = FastRMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device
        )
        self.last_forward_num_tokens = 0

    def forward(
        self,
        input_ids,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        *args,
        **kwargs
    ):

        input_ids, self.last_forward_num_tokens = prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        h = self.embedding(input_ids)

        mask = prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.decoder.layers[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )

        for layer in self.decoder.layers:
            h, mask = prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, _ = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h, past_key_values=None, hidden_states=(), attentions=()
        )


class LlamaQModelForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, decoder, dtype=torch.float16, device="cuda") -> None:
        super().__init__()
        self.model = QLlamaModel(config, decoder, dtype, device)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        *args,
        **kwargs
    ):
        outputs = self.model(input_ids, attention_mask=attention_mask)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        return logits
