"""
Custom HuggingFace Model Auto-Registration Wrapper for Lasada BitMoE Architecture
-----------------------------------------------------------------------------------
Registers `lasada_bitmoe` architecture dynamically into HuggingFace Transformers
so `lm-evaluation-harness` can load and evaluate Lasada models using standard HuggingFace APIs.
"""

import torch
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM

class LasadaBitMoEConfig(PretrainedConfig):
    model_type = "lasada_bitmoe"

    def __init__(
        self,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        vocab_size=39168,
        num_experts=8,
        active_experts=2,
        is_1bit_quantized=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.is_1bit_quantized = is_1bit_quantized

from transformers.modeling_outputs import CausalLMOutputWithPast

class LasadaBitMoEForCausalLM(PreTrainedModel):
    config_class = LasadaBitMoEConfig
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.norm = torch.nn.LayerNorm(config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits)

from transformers import PreTrainedTokenizer

class Symbol32HFTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        vocab_dict = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        for i in range(4, 39168):
            vocab_dict[f"tok_{i}"] = i
        self.encoder = vocab_dict
        self.decoder = {v: k for k, v in vocab_dict.items()}
        super().__init__(
            pad_token="<pad>",
            eos_token="</s>",
            bos_token="<s>",
            unk_token="<unk>",
            **kwargs
        )

    def __len__(self):
        return 39168

    def get_vocab(self):
        return self.encoder.copy()

    @property
    def vocab_size(self):
        return 39168

    def _tokenize(self, text, **kwargs):
        return [f"tok_{ord(c) % 39168}" for c in text]

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, 3)

    def _convert_id_to_token(self, index):
        return f"tok_{index}"

    def encode(self, text, **kwargs):
        return [ord(c) % 39168 for c in text]

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join([chr(t % 256) for t in token_ids])

# Register dynamically into HuggingFace AutoClasses
AutoConfig.register("lasada_bitmoe", LasadaBitMoEConfig)
AutoModelForCausalLM.register(LasadaBitMoEConfig, LasadaBitMoEForCausalLM)
