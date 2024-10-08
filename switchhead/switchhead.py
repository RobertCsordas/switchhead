import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
from .cvmm import cvmm, cvmm_prepare_sel2, CVMMSel
from dataclasses import dataclass


@dataclass
class AttentionMask:
    def __init__(self, position_mask: Optional[torch.Tensor] = None, src_length_mask: Optional[torch.Tensor] = None):
        self.position_mask = position_mask
        self.src_length_mask = src_length_mask
    
    position_mask: Optional[torch.Tensor]
    src_length_mask: Optional[torch.Tensor]


KVCache = Optional[Dict[str, torch.Tensor]]


class SwitchHeadCore(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 d_head: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2):

        super().__init__()

        self.input_size = d_model
        self.output_size = d_model
        self.pe_size = self.input_size
        self.expert_dropout = expert_dropout
        self.moe_k = moe_k
        self.attention_to_visualize = []
        self.selections_to_visualize = {}
        self.n_experts = n_experts

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = d_head or (d_model // n_heads)

        self.q = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)

        if self.n_experts > 1:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size, self.projection_size))
            self.o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.projection_size, self.output_size))
            self.sel_v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))
        else:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.projection_size, self.input_size))
            self.o = torch.nn.Parameter(torch.empty(self.output_size, self.n_heads * self.projection_size))

        self.sel_o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))

        self.register_buffer("scale", torch.full([1], 1.0 / math.sqrt(self.projection_size)), persistent=False)

    def generate_causal_attention_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=self.q.weight.device), diagonal=1)

    @torch.no_grad
    def reset_parameters(self, init_scale: float):
        if self.n_experts > 1:
            torch.nn.init.normal_(self.sel_v, 0, init_scale / math.sqrt(self.input_size))
            self.renorm_rows(self.sel_v)

        torch.nn.init.normal_(self.sel_o, 0, init_scale / math.sqrt(self.input_size))
        self.renorm_rows(self.sel_o)

        torch.nn.init.normal_(self.k.weight, 0, init_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.q.weight, 0, init_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.v, 0, init_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.o, 0, init_scale / math.sqrt(self.n_heads * self.projection_size))

    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def get_mask_tensor(self, src_len: int, mask: Optional[AttentionMask]) -> Optional[torch.Tensor]:
        if mask is None or (mask.position_mask is None and mask.src_length_mask is None):
            return None

        # mask.position_mask: [..., N_out, N_in]
        # mask.src_length_mask: [B, ...., N_in]
        # True where it has to be masked

        if mask.position_mask is not None:
            n_pad = src_len - mask.position_mask.shape[-1]
            if n_pad > 0:
                pm = F.pad(mask.position_mask, (n_pad, 0), 'constant', value=False)
            else:
                pm = mask.position_mask

        if mask.position_mask is None:
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2)
        elif mask.src_length_mask is None:
            m = pm
        else:
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2) | pm

        return m

    def get_sel(self, t: torch.Tensor, w: torch.Tensor) -> Tuple[CVMMSel, torch.Tensor]:
        sel = F.linear(t, w).float()
        sel = sel_raw = sel.view(*sel.shape[:-1], self.n_heads, -1)
        sel = sel.sigmoid()

        with torch.no_grad():
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, float('-inf'))
            else:
                sel2 = sel
            _, sel_index = sel2.topk(self.moe_k, dim=-1, sorted=False)
        sel_val = torch.gather(sel, -1, sel_index)

        sel_index_shifted = (torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype) * self.n_experts).unsqueeze(-1) + sel_index
        return cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1), sel_val), sel_raw

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def forward(self, q_src: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor, mask: Optional[AttentionMask],
                kv_cache: KVCache = None) -> Tuple[torch.Tensor, KVCache]:
        # *src: [batch_size, out_len, c]

        pos_offset = q_src.shape[1] - k_src.shape[1]
        assert pos_offset >= 0

        if mask is None:
            mask = AttentionMask(self.generate_causal_attention_mask(q_src.shape[1]))

        scale = self.scale.sqrt()

        q = self.q(q_src)
        k = self.k(k_src)
        q = q * scale.type_as(q)
        k = k * scale.type_as(k)

        if self.n_experts > 1:
            v_sel, v_sel_r = self.get_sel(k_src, self.sel_v)
            o_sel, o_sel_r = self.get_sel(q_src, self.sel_o)

            v = cvmm(v_src, v_sel, self.v).transpose(-2, -3)
        else:
            o_gate = F.sigmoid(F.linear(q_src, self.sel_o))
            v = self.project_to_torch_order(F.linear(v_src, self.v))

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)

        if kv_cache is not None:
            v = torch.cat([kv_cache["v"], v], dim=-2) if "v" in kv_cache else v
            k = torch.cat([kv_cache["k"], k], dim=-2) if "k" in kv_cache else k
            kv_cache = {
                "v": v,
                "k": k
            }

        q = self.dropout(q)
        res = self.attend(pos_offset, v, k, q, self.get_mask_tensor(v.shape[-2], mask))
        res = res.transpose(-2, -3)

        if self.n_experts > 1:
            # The output selection indices are calculated from the current state and are also used for projecting "q".
            # But that projection needs to create multiple copies for the different heads. Here we already have the
            # heads, but we have to create copies for the top-k elements. We can calculate that from the reduction
            # weight. We also want to compute not only the weighted average between the top-k elements, but also
            # of the different heads. So reshape the reduction weight accordingly.
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out = cvmm(res, o_sel, self.o)
        else:
            res = res * o_gate[..., None]
            out = F.linear(res.flatten(-2), self.o)

        return out, kv_cache


class RotaryPosEncoding(torch.nn.Module):
    # RoPE based on: https://www.kaggle.com/code/aeryss/rotary-postional-encoding-rope-pytorch
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = seq_dim

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rot(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, seq_dim: int, offset: int) -> torch.Tensor:
        sin = sin.narrow(seq_dim, offset, x.shape[seq_dim])
        cos = cos.narrow(seq_dim, offset, x.shape[seq_dim])
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                             seq_dim: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rot(q, sin, cos, seq_dim, offset), self.apply_rot(k, sin, cos, seq_dim, 0)

    def get(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[self.seq_dim]
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]

            self.cos_cached = emb.cos().view(*tgt_shape)
            self.sin_cached = emb.sin().view(*tgt_shape)

        return self.sin_cached, self.cos_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.get(k)
        return self.apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)


class SwitchHeadRope(SwitchHeadCore):
    def __init__(self, d_model: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 d_head: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 init_scale: float = 1.0, rotate_fraction: float = 0.5, rope_base: float = 10000):

        super().__init__(
            d_model, n_heads, n_experts, dropout, d_head, expert_dropout, moe_k)

        self.n_rotate = int(rotate_fraction * self.projection_size)
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

        super().reset_parameters(init_scale)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_rotate < self.projection_size:
            r_k = k[..., :self.n_rotate]
            nr_k = k[..., self.n_rotate:]
            r_q = q[..., :self.n_rotate]
            nr_q = q[..., self.n_rotate:]

            r_q, r_k = self.pe(r_q, r_k, offset)
            return torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            return self.pe(q, k, offset)

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.n_rotate > 0:
            q, k = self.rotate(q, k, pos_offset or 0)

        return F.scaled_dot_product_attention(q, k, v, ~mask, scale=1.0)



class SwitchHeadXL(SwitchHeadCore):
    def __init__(self, d_model: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 d_head: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 init_scale: float = 1.0):

        super().__init__(
            d_model, n_heads, n_experts, dropout, d_head, expert_dropout, moe_k)

        self.pe_size = d_model
        self.pos_to_pk = torch.nn.Parameter(torch.empty(n_heads * self.projection_size, self.pe_size))

        self.register_buffer("pos_encoding", self.create_pos_buffer(1000), persistent=False)
        super().reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float):
        super().reset_parameters(init_scale)
        torch.nn.init.normal_(self.pos_to_pk, 0, init_scale / math.sqrt(self.pe_size))

    def shift(self, posmat: torch.Tensor) -> torch.Tensor:
        # shape: [..., n_out, n_in * 2 - 1]
        # return: [..., n_out, n_in]

        n_in = (posmat.shape[-1] + 1) // 2
        n_neg = n_in - 1
        n_out = posmat.shape[-2]

        assert posmat.shape[-1] == n_in+n_neg

        # example:
        #p0[-3], p0[-2], p0[-1], | p0[0], p0[1], p0[2], p0[3] |
        #p1[-3], p1[-2], | p1[-1], p1[0], p1[1], p1[2],| p1[3]
        #p2[-3], |p2[-2], p2[-1], p2[0], p2[1],| p2[2], p2[3]
        #|p3[-3], p3[-2], p3[-1], p3[0],| p3[1], p3[2], p3[3]

        posmat = posmat.flatten(-2)
        posmat = posmat.narrow(-1, 1, n_out * (n_in + n_neg - 1))

        # example:
        #p0[-2], p0[-1], | p0[0], p0[1], p0[2], p0[3] |,
        #p1[-3], p1[-2]  | p1[-1], p1[0], p1[1], p1[2] |,
        #p1[3], p2[-3],  | p2[-2], p2[-1], p2[0], p2[1]|,
        #p2[2], p2[3] ,  |p3[-3], p3[-2], p3[-1], p3[0],|

        posmat = posmat.view(*posmat.shape[:-1], n_out, n_in + n_neg - 1)
        return posmat[..., n_neg-1 : ]

    def sinusoidal_pos_embedding(self, d_model: int, max_len: int = 5000, pos_offset: int = 0, omega: float = 1) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model, device=self.pos_to_pk.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.pos_to_pk.device).unsqueeze(1) + pos_offset
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=self.pos_to_pk.device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(omega * position * div_term)
        pe[:, 1::2] = torch.cos(omega * position * div_term)
        return pe

    def create_pos_buffer(self, max_len: int):
        res = self.sinusoidal_pos_embedding(self.pe_size, 2 * max_len - 1, -max_len + 1)
        assert res.shape[0] == 2 * max_len - 1
        return res

    def get_pos_subset(self, length: int, offset: int) -> torch.Tensor:
        total_len = length + offset
        if (2 * total_len - 1) > self.pos_encoding.shape[0]:
            self.pos_encoding = self.create_pos_buffer(total_len).to(self.pos_encoding.device).type_as(self.pos_encoding)

        return self.pos_encoding.narrow(0, self.pos_encoding.shape[0] // 2 - length + 1 - offset, 2 * length - 1)

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        scale = self.scale.sqrt()

        k_pos = self.get_pos_subset(k.shape[-2], pos_offset) * scale
        k_pos = F.linear(k_pos, self.pos_to_pk)
        k_pos = self.project_to_torch_order(k_pos)

        qc = qp = q

        att = self.shift(qp @ k_pos.transpose(-2, -1)) + qc @ k.transpose(-2, -1)
        if mask is not None:
            att.masked_fill_(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        return att @ v
    