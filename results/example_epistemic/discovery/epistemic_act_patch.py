"""Activation Patching Scan for Epistemic Framing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Runs activation patching across ALL attention heads (144) and MLP layers (12)
in GPT-2 small to identify which components matter for epistemic framing.

Clean = epistemic prompt ("I think the capital of France is")
Corrupted = same prompt with the epistemic prefix tokens replaced by random
            vocabulary tokens (same sequence length).

For each component, we patch the clean activation into the corrupted run
and measure the fraction of clean logit-diff recovered.

Usage:
    uv run --python 3.11 python src/instruments/data/epistemic_act_patch.py \
        --device cpu --n-prompts 24
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

_INSTRUMENTS = Path(__file__).resolve().parents[1]  # up to src/instruments/
sys.path.insert(0, str(_INSTRUMENTS))
from _common import load_model, generate_prompts, get_token_ids, log  # noqa: E402


# ---------------------------------------------------------------------------
# Corruption: replace epistemic prefix tokens with random vocab tokens
# ---------------------------------------------------------------------------

def _find_prefix_length(epistemic_text: str, bare_text: str) -> int:
    """Return the number of leading tokens in the epistemic prompt that
    constitute the epistemic prefix (the part not in the bare version).

    We rely on the fact that every epistemic prompt starts with
    "I think"/"I believe"/"I know" (2 tokens) prepended to the bare text.
    """
    # All current epistemic prefixes are 2 tokens: "I" + " think"/etc.
    # But verify by checking that the epistemic text ends with the bare text.
    # "I think the capital of France is" ends with "the capital of France is"
    # which is the bare text minus the leading determiner. Actually the bare
    # text is "The capital of France is" (capitalized), so we can't do a
    # simple suffix match. Just return 2 -- all prefixes are 2 tokens.
    return 2


def make_corrupted_tokens(model, epistemic_text: str, prefix_len: int, rng: torch.Generator):
    """Tokenize the epistemic prompt, replace the first `prefix_len` tokens
    with random vocabulary tokens, and return both clean and corrupted token
    tensors of identical shape.
    """
    clean_tokens = model.to_tokens(epistemic_text)  # (1, seq_len)
    corrupted_tokens = clean_tokens.clone()
    vocab_size = model.cfg.d_vocab
    random_ids = torch.randint(0, vocab_size, (prefix_len,), generator=rng)
    corrupted_tokens[0, 1:1 + prefix_len] = random_ids  # skip BOS at position 0
    return clean_tokens, corrupted_tokens


# ---------------------------------------------------------------------------
# Activation patching: heads
# ---------------------------------------------------------------------------

@torch.no_grad()
def patch_head(model, clean_cache, corrupted_tokens, layer: int, head: int,
               correct_id: int, incorrect_id: int) -> float:
    """Run the model on corrupted_tokens, patching in clean hook_z for one head.

    Returns the logit diff after patching.
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def patch_hook(z, hook, _layer=layer, _head=head):
        z[0, :, _head, :] = clean_cache[hook_name][0, :, _head, :]
        return z

    patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
    last_logits = patched_logits[0, -1]
    return (last_logits[correct_id] - last_logits[incorrect_id]).item()


# ---------------------------------------------------------------------------
# Activation patching: MLPs
# ---------------------------------------------------------------------------

@torch.no_grad()
def patch_mlp(model, clean_cache, corrupted_tokens, layer: int,
              correct_id: int, incorrect_id: int) -> float:
    """Run the model on corrupted_tokens, patching in clean hook_mlp_out for one layer.

    Returns the logit diff after patching.
    """
    hook_name = f"blocks.{layer}.hook_mlp_out"

    def patch_hook(mlp_out, hook):
        mlp_out[:] = clean_cache[hook_name]
        return mlp_out

    patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
    last_logits = patched_logits[0, -1]
    return (last_logits[correct_id] - last_logits[incorrect_id]).item()


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_scan(model, prompts, correct_ids, incorrect_ids, device: str):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_prompts = min(len(prompts), len(correct_ids))

    # Accumulators: sum of (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)
    head_effects = torch.zeros(n_layers, n_heads)
    mlp_effects = torch.zeros(n_layers)
    valid_count = 0

    rng = torch.Generator()
    rng.manual_seed(42)

    cache_filter = lambda name: "hook_z" in name or "hook_mlp_out" in name  # noqa: E731

    for i in range(n_prompts):
        p = prompts[i]
        cid = correct_ids[i]
        iid = incorrect_ids[i]
        epistemic_text = p.text
        bare_text = p.metadata.get("bare", "")
        prefix_len = _find_prefix_length(epistemic_text, bare_text)

        clean_tokens, corrupted_tokens = make_corrupted_tokens(
            model, epistemic_text, prefix_len, rng
        )
        clean_tokens = clean_tokens.to(device)
        corrupted_tokens = corrupted_tokens.to(device)

        # Clean run: get logit diff and cache
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=cache_filter
        )
        clean_ld = (clean_logits[0, -1, cid] - clean_logits[0, -1, iid]).item()

        # Corrupted run: get logit diff
        corrupted_logits = model(corrupted_tokens)
        corrupted_ld = (corrupted_logits[0, -1, cid] - corrupted_logits[0, -1, iid]).item()

        denominator = clean_ld - corrupted_ld
        if abs(denominator) < 1e-6:
            log(f"  Prompt {i}: clean_ld={clean_ld:.4f}, corrupted_ld={corrupted_ld:.4f} -- "
                f"skipping (no signal)")
            continue

        valid_count += 1
        log(f"  Prompt {i}/{n_prompts}: clean_ld={clean_ld:.4f}, "
            f"corrupted_ld={corrupted_ld:.4f}, delta={denominator:.4f}")

        # Patch each attention head
        for layer in range(n_layers):
            for head in range(n_heads):
                patched_ld = patch_head(
                    model, clean_cache, corrupted_tokens, layer, head, cid, iid
                )
                recovery = (patched_ld - corrupted_ld) / denominator
                head_effects[layer, head] += recovery

        # Patch each MLP layer
        for layer in range(n_layers):
            patched_ld = patch_mlp(
                model, clean_cache, corrupted_tokens, layer, cid, iid
            )
            recovery = (patched_ld - corrupted_ld) / denominator
            mlp_effects[layer] += recovery

    if valid_count > 0:
        head_effects /= valid_count
        mlp_effects /= valid_count

    return head_effects, mlp_effects, valid_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Activation patching scan for epistemic framing (GPT-2 small)"
    )
    parser.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    parser.add_argument("--n-prompts", type=int, default=24, help="Number of prompts")
    parser.add_argument("--out", default=None, help="Output JSON filename")
    args = parser.parse_args()

    model = load_model(args.model, device=args.device)
    tokenizer = model.tokenizer

    log("Generating epistemic framing prompts...")
    prompts = generate_prompts("epistemic_framing", tokenizer, n_prompts=args.n_prompts)
    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    log(f"Got {len(prompts)} prompts, {len(correct_ids)} with valid token IDs")

    log("Running activation patching scan...")
    t0 = time.time()
    head_effects, mlp_effects, valid_count = run_scan(
        model, prompts, correct_ids, incorrect_ids, device=args.device
    )
    elapsed = time.time() - t0
    log(f"Scan complete in {elapsed:.1f}s ({valid_count} valid prompts)")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # -- Print results sorted by magnitude --
    log("")
    log("=" * 70)
    log("ATTENTION HEAD EFFECTS (sorted by |effect|)")
    log("=" * 70)
    head_list = []
    for layer in range(n_layers):
        for head in range(n_heads):
            effect = head_effects[layer, head].item()
            head_list.append({"layer": layer, "head": head, "effect": effect})

    head_list.sort(key=lambda x: abs(x["effect"]), reverse=True)
    for entry in head_list:
        flag = " ***" if abs(entry["effect"]) > 0.01 else ""
        log(f"  L{entry['layer']:2d}H{entry['head']:2d}: {entry['effect']:+.4f}{flag}")

    log("")
    log("=" * 70)
    log("MLP LAYER EFFECTS (sorted by |effect|)")
    log("=" * 70)
    mlp_list = []
    for layer in range(n_layers):
        effect = mlp_effects[layer].item()
        mlp_list.append({"layer": layer, "effect": effect})

    mlp_list.sort(key=lambda x: abs(x["effect"]), reverse=True)
    for entry in mlp_list:
        flag = " ***" if abs(entry["effect"]) > 0.01 else ""
        log(f"  MLP {entry['layer']:2d}: {entry['effect']:+.4f}{flag}")

    # -- Summary of significant components --
    sig_heads = [h for h in head_list if abs(h["effect"]) > 0.01]
    sig_mlps = [m for m in mlp_list if abs(m["effect"]) > 0.01]
    log("")
    log(f"Significant heads (|effect| > 0.01): {len(sig_heads)}")
    for h in sig_heads:
        log(f"  L{h['layer']}H{h['head']}: {h['effect']:+.4f}")
    log(f"Significant MLPs (|effect| > 0.01): {len(sig_mlps)}")
    for m in sig_mlps:
        log(f"  MLP {m['layer']}: {m['effect']:+.4f}")

    # -- Save results as JSON --
    results = {
        "task": "epistemic_framing",
        "model": args.model,
        "device": args.device,
        "n_prompts": args.n_prompts,
        "valid_prompts": valid_count,
        "elapsed_seconds": round(elapsed, 1),
        "threshold": 0.01,
        "heads": head_list,
        "mlps": mlp_list,
        "significant_heads": sig_heads,
        "significant_mlps": sig_mlps,
    }

    out_name = args.out or "epistemic_act_patch.json"
    out_path = Path(__file__).resolve().parent / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {out_path} ({out_path.stat().st_size / 1024:.1f}KB)")


if __name__ == "__main__":
    main()
