from transformers.models.openai.modeling_openai import OpenAIGPTPreTrainedModel,OpenAIGPTModel,Block,CausalLMOutput,OpenAIGPTLMHeadModel, Attention
from transformers import GPT2LMHeadModel
from transformers import BertTokenizer
from torch import nn
import torch
# from transformers.models.gpt2.modeling_gpt2 import *
import math


# 这是为了加载flash attention优化修改的
class CustomAttention(Attention):
    
    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # self._attn -> model_training/models/patching.py._patched_gpt_neox_attn
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = (a,) + attn_outputs[1:]
        return outputs  # a, (attentions)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k.transpose(-1, -2))
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        try:
            b = self.bias[:, :, : w.size(-2), : w.size(-1)]
            w = w * b + -1e4 * (1 - b)
        except:
            import pdb;pdb.set_trace()
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask
        outputs = torch.matmul(w, v)
        return outputs, w

class CustomBlock(Block):
    
    def __init__(self, n_positions, config, scale=False):
        super().__init__(n_positions, config)
        self.attn = CustomAttention(config.n_embd, n_positions, config, scale)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        attn_outputs = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        a = attn_outputs[0]

        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = (h,) + attn_outputs[1:]
        return outputs

# NOTE For _cdial_gpt LM model
class CustomModel(OpenAIGPTModel):
    # origin used in transformers==2.2.1
    def __init__(self, config):
        # NOTE: not init OpenAIGPTModel
        super(OpenAIGPTModel,self).__init__(config)
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # NOTE n_position -> n_ctx
        self.h = nn.ModuleList([CustomBlock(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        # self.position_ids=torch.arange(config.n_positions).to('cuda') # WHY: 必须手动
        self.register_buffer("position_ids", torch.arange(config.n_positions))
        # module._buffers => 不会被优化 但是会被保存到state_dict，貌似老版的并不会保存到state_dict，所以这里需要去掉
        self.post_init()

class OpenAIGPTLMHeadModel_oldversion(OpenAIGPTLMHeadModel):
    def __init__(self, config):
        # NOTE: not init OpenAIGPTLMHeadModel
        super(OpenAIGPTLMHeadModel, self).__init__(config)
        self.transformer = CustomModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()