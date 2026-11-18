"""The Reflexx Ear: a small causal CTC recogniser that hears each 80 ms once (KARST hybrid: gated delta + windowed attention).

Layers vendored from our KARST research code (`karst/karst/layers.py`, `karst/karst/delta.py`), trimmed to what the ear
needs, with every layer carrying its state so `forward(chunk, state)` continues where the last chunk stopped:
- front end: two causal stride-2 convs, 20 ms feature rows -> 80 ms frames (TurboCTC's CTC rate); frame j hears rows
  up to 4j+3, the end of its own 80 ms;
- gated delta layers (Gated DeltaNet): a K x V memory per head, decayed, then written with the error between the value
  and what the memory already returns for the key;
- sliding-window attention over the last `window` frames, no positional encoding (the delta layers carry order);
- CTC head over TurboCTC's 16,384 word-pieces, blank 0.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # flash-linear-attention: chunkwise kernel for training (and full-clip inference) on CUDA
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAS_FLA = True
except Exception:  # pragma: no cover - depends on the install
    chunk_gated_delta_rule = None
    HAS_FLA = False

FEAT = 320
BLANK = 0


@dataclass
class EarConfig:
    d: int = 256
    pattern: str = "DDDADD"   # D = gated delta, A = sliding-window attention
    heads: int = 4
    head_dim: int = 64
    ffn: int = 640
    window: int = 32          # attention frames (80 ms each): 2.56 s
    conv: int = 4             # short causal conv in the delta layers
    vocab: int = 16384


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.down._residual_out = True

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


def delta_ref(q, k, v, g, beta, scale, s=None):
    """Reference recurrence, fp32. q, k: [B,T,H,K]; v: [B,T,H,V]; g (log decay), beta: [B,T,H]; s: [B,H,K,V]."""
    b, t, h, dk = q.shape
    out_dtype = v.dtype
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    s = q.new_zeros(b, h, dk, v.shape[-1]) if s is None else s.float()
    outs = []
    for i in range(t):
        s = s * g[:, i].exp()[..., None, None]
        ki = k[:, i]
        u = (v[:, i] - torch.einsum("bhk,bhkv->bhv", ki, s)) * beta[:, i, :, None]
        s = s + ki[..., :, None] * u[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * scale, s))
    return torch.stack(outs, dim=1).to(out_dtype), s


def delta_rule(q, k, v, g, beta, scale, s=None, recurrent=False):
    if HAS_FLA and q.is_cuda and not recurrent:
        return chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=s,
                                      output_final_state=True)
    return delta_ref(q, k, v, g, beta, scale, s)


class DeltaMixer(nn.Module):
    """State: (last conv-1 projected inputs [B, conv-1, 3HK], memory [B, H, K, V] fp32)."""

    def __init__(self, d, heads, head_dim, conv):
        super().__init__()
        self.h, self.hd, self.k = heads, head_dim, conv
        hk = heads * head_dim
        self.qkv = nn.Linear(d, 3 * hk, bias=False)
        self.gate = nn.Linear(d, hk, bias=False)
        self.o = nn.Linear(hk, d, bias=False)
        self.o._residual_out = True
        self.a = nn.Linear(d, heads, bias=False)   # decay input
        self.b = nn.Linear(d, heads, bias=False)   # write strength
        self.conv = nn.Conv1d(3 * hk, 3 * hk, conv, groups=3 * hk, bias=False)
        self.o_norm = RMSNorm(head_dim)
        # Mamba-style init: per-head decay rate A in [1, 16], dt in [1e-3, 1e-1]
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(1, 16)))
        dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

    def forward(self, x, state=None, recurrent=False):
        b, t, _ = x.shape
        u = self.qkv(x)
        buf = u.new_zeros(b, self.k - 1, u.shape[-1]) if state is None else state[0].to(u.dtype)
        u = torch.cat([buf, u], dim=1)
        c = self.conv(u.transpose(1, 2)).transpose(1, 2)
        q, k, v = F.silu(c).view(b, t, 3, self.h, self.hd).unbind(2)
        # autocast runs the norm in fp32, which would silently promote q/k; fla needs one dtype
        q, k = F.normalize(q, dim=-1).to(v.dtype), F.normalize(k, dim=-1).to(v.dtype)
        g = -self.A_log.float().exp() * F.softplus(self.a(x).float() + self.dt_bias)
        beta = torch.sigmoid(self.b(x).float())
        o, s = delta_rule(q, k, v.contiguous(), g, beta.to(q.dtype), self.hd ** -0.5,
                          None if state is None else state[1], recurrent)
        o = self.o_norm(o) * F.silu(self.gate(x).view(b, t, self.h, self.hd))
        return self.o(o.reshape(b, t, -1)), (u[:, -(self.k - 1):], s)


class WindowAttention(nn.Module):
    """Causal attention over the last `window` frames, QK-norm with gains, no positions. State: (k, v) of up to
    window-1 past frames, [B, H, n, hd]."""

    def __init__(self, d, heads, window):
        super().__init__()
        self.h, self.hd, self.window = heads, d // heads, window
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.o._residual_out = True
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x, state=None, recurrent=False):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.hd).unbind(2)
        v = v.transpose(1, 2)
        q, k = self.q_norm(q).to(v.dtype).transpose(1, 2), self.k_norm(k).to(v.dtype).transpose(1, 2)
        if state is not None:
            k, v = torch.cat([state[0].to(k.dtype), k], dim=2), torch.cat([state[1].to(v.dtype), v], dim=2)
        past = k.shape[2] - t
        i = torch.arange(t, device=x.device)[:, None] + past
        j = torch.arange(past + t, device=x.device)[None, :]
        mask = (j <= i) & (i - j < self.window)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        keep = self.window - 1
        return self.o(o.transpose(1, 2).reshape(b, t, d)), (k[:, :, -keep:], v[:, :, -keep:])


class Block(nn.Module):
    def __init__(self, kind, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.mix = (DeltaMixer(cfg.d, cfg.heads, cfg.head_dim, cfg.conv) if kind == "D"
                    else WindowAttention(cfg.d, cfg.heads, cfg.window))
        self.ffn = SwiGLU(cfg.d, cfg.ffn)

    def forward(self, x, state=None, recurrent=False):
        h, state = self.mix(self.n1(x), state, recurrent)
        x = x + h
        return x + self.ffn(self.n2(x)), state


class Subsample(nn.Module):
    """Two causal stride-2 convs: [B, rows, 320] (rows a multiple of 4) -> [B, rows/4, d]. State: the last input row
    and the last first-conv output."""

    def __init__(self, d):
        super().__init__()
        self.c1 = nn.Conv1d(FEAT, d, 3, stride=2)
        self.c2 = nn.Conv1d(d, d, 3, stride=2)

    def forward(self, x, state=None):
        b = x.shape[0]
        s1, s2 = (x.new_zeros(b, 1, FEAT), None) if state is None else state
        a = torch.cat([s1.to(x.dtype), x], dim=1)
        h = F.silu(self.c1(a.transpose(1, 2))).transpose(1, 2)
        s2 = h.new_zeros(b, 1, h.shape[-1]) if s2 is None else s2.to(h.dtype)
        h2 = torch.cat([s2, h], dim=1)
        y = F.silu(self.c2(h2.transpose(1, 2))).transpose(1, 2)
        return y, (a[:, -1:], h2[:, -1:])


class Ear(nn.Module):
    def __init__(self, cfg: EarConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.zeros(FEAT))
        self.register_buffer("std", torch.ones(FEAT))
        self.sub = Subsample(cfg.d)
        self.blocks = nn.ModuleList(Block(kind, cfg) for kind in cfg.pattern)
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab)
        self.apply(self._init)
        for m in self.modules():
            if getattr(m, "_residual_out", False):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * len(cfg.pattern)))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feats, state=None, recurrent=False):
        """feats [B, rows, 320] -> logits [B, rows/4, vocab], state. Pass the returned state with the next chunk to
        continue a stream; `recurrent` runs the delta layers step by step (the streaming reference)."""
        x = (feats - self.mean) / self.std
        sub_state, states = (None, [None] * len(self.blocks)) if state is None else state
        x, sub_state = self.sub(x, sub_state)
        new = []
        for block, s in zip(self.blocks, states):
            x, s = block(x, s, recurrent)
            new.append(s)
        return self.head(self.norm(x)), (sub_state, new)

    def config(self):
        return asdict(self.cfg)
