import torch
import switchhead
import math
from typing import Tuple, Optional

# SwitchHead does not have a layernorm and a residual connection built in, and requires separate
# inputs for the Q, K, V projections. This is done for greater flexibilty (See our MoEUT paper for 
# example). To use it as a self-attention in a standard Transfromer, see this example below:

class SwitchHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, *args, attention_core=switchhead.SwitchHeadRope, **kwargs):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.attention = attention_core(d_model,  *args, **kwargs)

    def forward(self, x: torch.Tensor, mask: Optional[switchhead.AttentionMask] = None, kv_cache: switchhead.KVCache = None) -> Tuple[torch.Tensor, switchhead.KVCache]:
        xn = self.norm(x)
        res, kv_cache = self.attention(xn, xn, xn, mask=mask)
        return x + res, kv_cache


# 243M param model from the paper.
batch_size = 8
context_window = 1024
d_model = 1024
n_layers = 18

x = torch.randn(batch_size, context_window, d_model).cuda()

# RoPE example (default)
attention = SwitchHeadSelfAttention(d_model, n_heads=4, n_experts=4, d_head=100, init_scale=1/math.sqrt(n_layers)).cuda()
out, _ = attention(x)

print("RoPE: Output shape: ", out.shape)


# Transformer-XL example
attention = SwitchHeadSelfAttention(d_model, n_heads=4, n_experts=4, d_head=100, init_scale=1/math.sqrt(n_layers),
                                    attention_core=switchhead.SwitchHeadXL).cuda()
out, _ = attention(x)
print("XL: Output shape: ", out.shape)