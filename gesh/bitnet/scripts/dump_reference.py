#!/usr/bin/env python3
"""
dump_reference.py — run HF's BitNet reference and capture per-layer activations.

The harness's per-layer comparison gate (Phase 1 fidelity gate, per
journal/bitnet_phase1_synthesize.md D2) needs ground-truth activations
to compare against. This script produces them.

For the activations to be a meaningful comparator, the HF model is run on
the exact same input the C harness will run on. The output is a structured
.npz file with one array per (layer, sublayer) capture site.

Capture sites per block (matches the structure in bitnet_phase1_o1_findings.md):
  layer.<i>.input_layernorm.input
  layer.<i>.input_layernorm.output
  layer.<i>.attn.q
  layer.<i>.attn.k
  layer.<i>.attn.v
  layer.<i>.attn.scores                   (Q@K^T scaled, pre-softmax)
  layer.<i>.attn.weights                  (post-softmax)
  layer.<i>.attn.output                   (scores @ V, pre-sub-norm)
  layer.<i>.attn_sub_norm.output
  layer.<i>.attn.o_proj.output
  layer.<i>.post_attention_layernorm.output
  layer.<i>.ffn.gate_proj
  layer.<i>.ffn.up_proj
  layer.<i>.ffn.gate_act                  (relu²(gate) * up)
  layer.<i>.ffn_sub_norm.output
  layer.<i>.ffn.down_proj.output
  layer.<i>.block_output                  (post final residual add)

The HF reference runs at bf16. Captures convert to fp32 numpy for storage.
"""

import argparse
import sys
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"[error] missing dep: {e}\n"
          f"Install with: pip install torch transformers", file=sys.stderr)
    sys.exit(1)

MODEL_REPO = "microsoft/bitnet-b1.58-2B-4T"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="The capital of France is",
        help="Input prompt.")
    parser.add_argument("--max-layers", type=int, default=1,
        help="Capture activations for layers [0, max_layers). Default: 1 "
             "(work-unit 1 single-block scope).")
    parser.add_argument("--output", default="bitnet_reference_activations.npz",
        help="Output .npz path.")
    parser.add_argument("--device", default="cpu",
        help="Torch device (cpu / cuda / mps). Default: cpu.")
    args = parser.parse_args()

    print(f"[info] loading {MODEL_REPO} on {args.device}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    # Tokenize input.
    inputs = tok(args.prompt, return_tensors="pt").to(args.device)
    print(f"[info] prompt: {args.prompt!r}", file=sys.stderr)
    print(f"[info] tokens: {inputs['input_ids'].tolist()[0]}", file=sys.stderr)

    # Hook registry: name → tensor (fp32 numpy).
    captures: dict[str, np.ndarray] = {}

    def make_hook(name: str):
        def _hook(module, inputs_, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[name] = t.detach().to(torch.float32).cpu().numpy()
        return _hook

    def make_pre_hook(name: str):
        def _hook(module, inputs_):
            t = inputs_[0]
            captures[name] = t.detach().to(torch.float32).cpu().numpy()
        return _hook

    # Register hooks on the requested layer count.
    handles = []
    for i in range(args.max_layers):
        layer = model.model.layers[i]
        handles.append(layer.input_layernorm.register_forward_pre_hook(
            make_pre_hook(f"layer.{i}.input_layernorm.input")))
        handles.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"layer.{i}.input_layernorm.output")))

        handles.append(layer.self_attn.q_proj.register_forward_hook(
            make_hook(f"layer.{i}.attn.q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            make_hook(f"layer.{i}.attn.k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_hook(f"layer.{i}.attn.v")))
        handles.append(layer.self_attn.attn_sub_norm.register_forward_hook(
            make_hook(f"layer.{i}.attn_sub_norm.output")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(f"layer.{i}.attn.o_proj.output")))

        handles.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"layer.{i}.post_attention_layernorm.output")))

        handles.append(layer.mlp.gate_proj.register_forward_hook(
            make_hook(f"layer.{i}.ffn.gate_proj")))
        handles.append(layer.mlp.up_proj.register_forward_hook(
            make_hook(f"layer.{i}.ffn.up_proj")))
        handles.append(layer.mlp.ffn_sub_norm.register_forward_hook(
            make_hook(f"layer.{i}.ffn_sub_norm.output")))
        handles.append(layer.mlp.down_proj.register_forward_hook(
            make_hook(f"layer.{i}.ffn.down_proj.output")))

        # Block-level output: post-residual.
        handles.append(layer.register_forward_hook(
            make_hook(f"layer.{i}.block_output")))

    # Run forward pass.
    print(f"[info] forward pass on {args.max_layers} layers...", file=sys.stderr)
    with torch.no_grad():
        # Use only the first token for now; multi-token prefill is
        # work-unit 7+ scope (KV cache).
        first_tok = inputs["input_ids"][:, :1]
        _ = model(input_ids=first_tok, use_cache=False)

    # Detach hooks.
    for h in handles:
        h.remove()

    # Save.
    np.savez(args.output, **captures)
    print(f"[done] captured {len(captures)} tensors → {args.output}",
          file=sys.stderr)
    print(f"[info] sample tensor shapes:", file=sys.stderr)
    for k, v in list(captures.items())[:5]:
        print(f"  {k}: {v.shape} {v.dtype}", file=sys.stderr)


if __name__ == "__main__":
    main()
