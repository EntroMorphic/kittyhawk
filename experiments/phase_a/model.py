"""Phase A model: TinyGPT with ternary weights (b1.58 QAT) and two attention variants.

Variant A: dense scaled dot-product attention.
Variant B: substrate-routed attention with top-k=4 selection via signature distance.

Both variants share the same weight quantization (b1.58 sign-STE) and the same
architecture; only the attention selection differs.

Phase A scope (per journal/td27_7_phase_a_2026-05-11.md):
  Ternary weights, float activations, float gradients.
  Phase B re-introduces substrate mtfp19 activations.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from task import VOCAB, SEQ_LEN


# ── BitLinear (b1.58 ternary weight QAT) ──────────────────────────────────────


def ternary_quantize(W: torch.Tensor) -> torch.Tensor:
    """b1.58 quantize: per-tensor scale α = mean(|W|); w_q = round(W/α) clipped to {-1,0,+1}.

    Forward returns w_q * α; backward uses STE through round+clip (identity).
    """
    alpha = W.abs().mean().clamp(min=1e-8)
    w_normalized = W / alpha
    w_q = torch.clamp(torch.round(w_normalized), -1.0, 1.0)
    # STE: forward = w_q * α; backward = identity through round/clip (gradient w.r.t. W)
    return (W + (w_q * alpha - W).detach())


class BitLinear(nn.Module):
    """Linear layer with ternary weight QAT (b1.58 style). Bias optional, float."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize like nn.Linear default
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = ternary_quantize(self.weight)
        return F.linear(x, w_q, self.bias)


# ── Substrate routing primitive ───────────────────────────────────────────────


def substrate_route_topk(Q: torch.Tensor, K: torch.Tensor, top_k: int) -> torch.Tensor:
    """Pick top-k K positions per Q position by SIGNATURE DISTANCE.

    Q: (..., n_q, d), K: (..., n_k, d) — head split already applied if multi-head.
    Returns: (..., n_q, top_k) indices into K's n_k axis.

    Implementation:
      - Sign-based signature: sgn(x) ∈ {-1, 0, +1}.
      - Distance: number of dimensions where Q's sign differs from K's sign.
      - top-k = positions with SMALLEST distance.

    Discrete selection — no gradient flows back through the indices.
    Gradients to Q, K flow through downstream `gather` (only to selected positions),
    which is the STE behavior we want.
    """
    # Signatures
    q_sig = torch.sign(Q)  # (..., n_q, d)
    k_sig = torch.sign(K)  # (..., n_k, d)

    # Pairwise signature distance: count where signs differ
    # Broadcast q_sig: (..., n_q, 1, d), k_sig: (..., 1, n_k, d)
    diff = (q_sig.unsqueeze(-2) != k_sig.unsqueeze(-3)).sum(dim=-1)  # (..., n_q, n_k)

    # Top-k smallest (negate to use topk's "largest")
    # detach to ensure no spurious gradient flows
    _, idx = torch.topk(-diff.detach(), k=top_k, dim=-1)  # (..., n_q, top_k)
    return idx


def gather_selected(K: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather selected K positions along the n_k axis.

    K: (..., n_k, d)
    idx: (..., n_q, top_k) — indices into n_k axis
    Returns: (..., n_q, top_k, d)
    """
    # Expand K to (..., n_q, n_k, d) so we can gather along dim=-2 per q-row
    # Actually easier: use advanced indexing
    # K_expanded shape: (..., 1, n_k, d) -> (..., n_q, n_k, d) via expand
    n_q = idx.shape[-2]
    top_k = idx.shape[-1]
    d = K.shape[-1]
    # idx shape (..., n_q, top_k); we need (..., n_q, top_k, d)
    idx_expanded = idx.unsqueeze(-1).expand(*idx.shape, d)
    # K reshape: (..., 1, n_k, d) -> broadcast to (..., n_q, n_k, d) via expand
    K_per_q = K.unsqueeze(-3).expand(*K.shape[:-2], n_q, K.shape[-2], d)
    return torch.gather(K_per_q, dim=-2, index=idx_expanded)


# ── Attention variants ────────────────────────────────────────────────────────


class DenseAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_qkv = BitLinear(model_dim, 3 * num_heads * head_dim)
        self.W_o = BitLinear(num_heads * head_dim, model_dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.W_qkv(x)  # (B, T, 3*H*d)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # (B, T, H, d) -> (B, H, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scores
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # Causal mask
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.W_o(out)


class SubstrateRoutedAttention(nn.Module):
    """Top-k=4 attention selection via substrate signature distance."""

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, top_k: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.W_qkv = BitLinear(model_dim, 3 * num_heads * head_dim)
        self.W_o = BitLinear(num_heads * head_dim, model_dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.W_qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # (B, T, H, d) -> (B, H, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply causal mask to K's pool: substrate route only over valid past positions.
        # We make K positions that should not be attended invalid by setting their sign
        # to a value that never matches (we'll mask in the score step instead — simpler).
        # For Phase A: route picks top-k including future positions; we mask in scores.

        # Substrate route: per query, pick top_k K positions
        # Shapes: q (B, H, T, d), k (B, H, T, d)
        idx = substrate_route_topk(q, k, min(self.top_k, T))  # (B, H, T, top_k)

        # Gather selected K and V
        k_sel = gather_selected(k, idx)  # (B, H, T, top_k, d)
        v_sel = gather_selected(v, idx)

        # Sparse scores: q (B, H, T, d) vs k_sel (B, H, T, top_k, d)
        # We want score[b, h, t, i] = q[b, h, t] · k_sel[b, h, t, i]
        # = sum over d of q[..., t, d] * k_sel[..., t, i, d]
        scores = (q.unsqueeze(-2) * k_sel).sum(dim=-1) / math.sqrt(self.head_dim)  # (B, H, T, top_k)

        # Causal mask on the SELECTED indices:
        # For query position t, valid K positions are j ≤ t. We mark invalid as -inf.
        # idx (B, H, T, top_k): need mask[b, h, t, i] = (idx[b, h, t, i] > t)
        t_pos = torch.arange(T, device=x.device).view(1, 1, T, 1)
        invalid = idx > t_pos
        scores = scores.masked_fill(invalid, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # If all top_k are invalid (early positions), softmax produces NaN; replace with 0.
        attn = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum: attn (B, H, T, top_k) × v_sel (B, H, T, top_k, d)
        out = (attn.unsqueeze(-1) * v_sel).sum(dim=-2)  # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.W_o(out)


# ── Transformer block + tiny GPT ──────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=self.eps).sqrt()
        return x / rms * self.g


class FFN(nn.Module):
    """BitNet-style gated FFN: down(relu²(gate(x)) * up(x))."""

    def __init__(self, model_dim: int, inner_dim: int):
        super().__init__()
        self.gate = BitLinear(model_dim, inner_dim)
        self.up = BitLinear(model_dim, inner_dim)
        self.down = BitLinear(inner_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.relu(self.gate(x)).pow(2)
        return self.down(g * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, ffn_dim, variant: str):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        if variant == "dense":
            self.attn = DenseAttention(model_dim, num_heads, head_dim)
        elif variant == "substrate":
            self.attn = SubstrateRoutedAttention(model_dim, num_heads, head_dim, top_k=4)
        else:
            raise ValueError(f"variant must be 'dense' or 'substrate', got {variant}")
        self.norm2 = RMSNorm(model_dim)
        self.ffn = FFN(model_dim, ffn_dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    """1-layer GPT with ternary weights. Phase A spec."""

    def __init__(self, variant: str, model_dim=64, num_heads=4, head_dim=16, ffn_dim=128):
        super().__init__()
        assert num_heads * head_dim == model_dim
        self.variant = variant
        self.tok_emb = nn.Embedding(VOCAB, model_dim)
        self.pos_emb = nn.Embedding(SEQ_LEN, model_dim)
        self.block = TransformerBlock(model_dim, num_heads, head_dim, ffn_dim, variant)
        self.norm_f = RMSNorm(model_dim)
        # LM head: tied weights would save params but keep simple (BitLinear)
        self.lm_head = BitLinear(model_dim, VOCAB)

        # Pre-compute causal mask (for dense)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        x = self.block(x, self.causal_mask[:T, :T])
        x = self.norm_f(x)
        return self.lm_head(x)  # (B, T, VOCAB)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for v in ("dense", "substrate"):
        m = TinyGPT(v)
        x = torch.randint(0, VOCAB, (2, SEQ_LEN))
        out = m(x)
        print(f"{v}: params={count_params(m)} output={tuple(out.shape)}")
