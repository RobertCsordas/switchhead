# SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention

![switchhead](https://robertcsordas.github.io/images/moeatt_simple.svg)

Official implementation of the [SwitchHead](https://arxiv.org/abs/2312.07987) attention from our NeurIPS 2024 paper.

This repository is an user-friendly implementation of SwitchHead. For the training code, please refer to [https://github.com/robertcsordas/moe_attention](https://github.com/robertcsordas/moe_attention).

This implementation relies on the [CVMM Triton kernel](https://github.com/RobertCsordas/moe_layer/blob/master/triton_src/moe_layer/cvmm.py) from $\sigma$-MoE.

## Example

```python

import torch
import switchhead
import math
from typing import Tuple, Optional

class SwitchHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, *args, **kwargs):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.attention = switchhead.SwitchHeadRope(d_model,  *args, **kwargs)

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

print(out.shape)
```

A simple example can be found in `example.py`.

## Usage

We provide two versions:
- `SwitchHeadRope` is a standard RoPE attention implementation. It uses `F.scaled_dot_product_attention` for fast and efficient attention.
- `SwitchHeadXL` is a Transformer XL-style relative positional encoding-based attention. The attention itself is implemented in PyTorch.

SwitchHead does *not* have an internal residual connection or layernorm. This is to provide greater flexibiltity for customization. It also requires passign individual tensors for q, k, v projections. See `example.py` or the example above to see how to use it as a simple self attention.

The signature of the init function of the RoPE version is as follows:
```python
def __init__(self, d_model: int, n_heads: int, n_experts: int, dropout: float = 0.0,
             d_head: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
             init_scale: float = 1.0, rotate_fraction: float = 0.5, rope_base: float = 10000):
```

The meaning of the arguments:
- `d_model` - the number of channels in the residual stream.
- `n_heads` - number of attention heads
- `n_experts` - the number of attention experts. The `att_n_experts=1` case is handled specially for efficiency reasons. It behaves like a standard attention with a learned output gate.
- `d_head` - the size of the K, Q, V projections in the attention
- `att_expert_dropout` - the probability of dropping out an expert.
- `moe_k` - the number of simultaneously active experts in the attention layer
- `dropout` - dropout used on the queries
- `init_scale` - scaling for the std for the initial random weights. Should be set to `1/math.sqrt(n_layers)` for the best performance.
- `rotate_fraction` (RoPE only) - what proporiton of the channels to rotate
- `rope_base` (RoPE only) - controls the gaps between the frequencies of the different channels.


The signature of the forward function:
```python
def forward(self, q_src: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor, mask: Optional[AttentionMask],
                kv_cache: KVCache = None) -> Tuple[torch.Tensor, KVCache]:
```

The meaning of the arguments:
- `q_src` - Input for the query projections. Shape: [batch size, context length, d_model]
- `k_src` - Input for the key projections. Shape: [batch size, context length, d_model]
- `v_src` - Input for the value projections. Shape: [batch size, context length, d_model]
- `mask` - optional attention mask. If None, causal mask is automatically used. Pass AttentionMask() to disable masking.
- `kv_cache` - optional KV cache. Pass an empty dict ({}) to start caching. Otherwise, no KV cache is returned to save memory.

The forward pass returns a tuple of (output, update kv cache). The updated KV cache is None if the argument `kv_cache` was None to save memory. Otherwise it can be fed as the `kv_cache` in the next forward pass.

The AttentionMask has two optional boolean fields. True if to be removed. If None, they are ignored.
- `position_mask` - position mask, e.g. the causal attention mask. Shape: [context length, context length]
- `src_length_mask` - for masking padding tokens in sequences. Useful if no autoregressive mask is applied. Shape: [batch size, context length]

## torch.compile() support

torch.compile() is supported with PyTorch >= 2.3.

## Project structure
```
├───switchhead - the SwitchHead attention implementation. Copy this to your project.
│    ├─  cvmm.py - the CVMM Triton kernel.
│    └─  switchhead.py - the implementation of SwitchHead
│
├───example.py - an example forward using both variants pass.
├───LICENSE - MIT License.
└───README.md - this documentation.
```

## Installation Instruction

The only external dependencies are PyTorch (>= 2.1) and Triton (>= 2.1). Copy the `switchhead` directory to your project and import it as shown in the examples above.

```bash
pip3 install -r requirements.txt
```

## Known issues

Triton seems to be broken on Volta GPUs when using float16 starting from PyTorch 2.2 to 2.3 (see [github issue](https://github.com/pytorch/pytorch/issues/127157)). Until the PyTorch team does not fix the issue, please downgrade to PyTorch 2.1 or disable AMP if you have Volta GPUs.
