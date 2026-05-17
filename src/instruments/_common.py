"""Shared infrastructure for the circuit evaluation pipeline.

Provides model loading, circuit/prompt access, logit-diff computation,
mean-activation calibration, and result serialization. All evaluator
scripts import from here.
"""
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

TAXONOMY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TAXONOMY_ROOT))

from lib.tasks.ioi import circuit as _ioi_circuit  # noqa: E402
from lib.tasks.greater_than import circuit as _gt_circuit  # noqa: E402
from lib.tasks.induction import circuit as _ind_circuit  # noqa: E402
from lib.tasks.sva import circuit as _sva_circuit  # noqa: E402
from lib.tasks.gendered_pronoun import circuit as _gp_circuit  # noqa: E402
from lib.tasks.rti import circuit as _rti_circuit  # noqa: E402
from lib.tasks.acronym import circuit as _acr_circuit  # noqa: E402
from lib.tasks.copy_suppression import circuit as _cs_circuit  # noqa: E402
from lib.tasks.prompts import TASK_REGISTRY  # noqa: E402

_TASK_TO_MODULE = {
    "ioi": _ioi_circuit, "greater_than": _gt_circuit,
    "induction": _ind_circuit, "sva": _sva_circuit,
    "gendered_pronoun": _gp_circuit, "rti": _rti_circuit,
    "acronym": _acr_circuit, "copy_suppression": _cs_circuit,
    "rti_pattern": _rti_circuit, "sequence_internal": _ind_circuit,
    "alternating_pair": _ind_circuit, "novel_song": _ind_circuit,
    "centering_theory": _ioi_circuit, "resumptive": _ioi_circuit,
    "self_allo": _ioi_circuit, "token_flood": _rti_circuit,
    "buffalo": _rti_circuit,
}


def get_circuit(task: str) -> dict:
    if task not in _TASK_TO_MODULE:
        raise ValueError(f"Unknown task: {task}. Available: {list(_TASK_TO_MODULE.keys())}")
    mod = _TASK_TO_MODULE[task]
    return {"roles": mod.ROLES, "bands": mod.BANDS, "pathways": mod.PATHWAYS}


def get_all_heads(circuit: dict) -> set[tuple[int, int]]:
    heads = set()
    for role_heads in circuit["roles"].values():
        heads.update(role_heads)
    return heads


def get_all_edges(circuit: dict) -> set[tuple[int, int, int, int]]:
    edges = set()
    roles = circuit["roles"]
    for sender_role, receiver_role in circuit["pathways"]:
        for s in roles[sender_role]:
            for r in roles[receiver_role]:
                if s[0] < r[0]:
                    edges.add((s[0], s[1], r[0], r[1]))
    return edges

CIRCUIT_TASKS = [
    "ioi", "greater_than", "induction", "sva", "gendered_pronoun",
    "rti", "acronym", "copy_suppression",
]
ALIAS_TASKS = [
    "rti_pattern", "sequence_internal", "alternating_pair", "novel_song",
    "centering_theory", "resumptive", "self_allo", "token_flood", "buffalo",
]
ALL_TASKS = sorted(CIRCUIT_TASKS + ALIAS_TASKS)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    metric_id: str
    value: float
    baseline_random: float | None = None
    baseline_untrained: float | None = None
    baseline_literature: float | None = None
    n_samples: int = 0
    ci_low: float | None = None
    ci_high: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = {k: _jsonable(v) for k, v in d["metadata"].items()}
        return d


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    return v


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model loading (generic)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}


def load_model(model_name: str = "gpt2", device: str = "cpu"):
    key = f"{model_name}:{device}"
    if key not in _MODEL_CACHE:
        from transformer_lens import HookedTransformer
        log(f"Loading {model_name} on {device}...")
        model = HookedTransformer.from_pretrained(model_name, device=device)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Circuit access
# ---------------------------------------------------------------------------

def get_circuit_heads(task: str) -> set[tuple[int, int]]:
    try:
        circuit = get_circuit(task)
        return get_all_heads(circuit)
    except Exception:
        return set()


def get_circuit_info(task: str):
    try:
        circuit = get_circuit(task)
        return circuit, get_all_heads(circuit), get_all_edges(circuit)
    except Exception:
        return None, set(), set()


def get_all_circuit_info():
    per_task_heads = {}
    per_task_circuits = {}
    for task in ALL_TASKS:
        circuit, heads, _ = get_circuit_info(task)
        per_task_heads[task] = heads
        per_task_circuits[task] = circuit
    return per_task_heads, per_task_circuits


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

class _RtiPromptAdapter:
    def __init__(self, d, tokenizer):
        self.text = d["text"]
        self.target_correct = tokenizer.decode([d["correct_id"]])
        self.target_incorrect = tokenizer.decode([d["wrong_id"]])
        self.metadata = {}
        self._correct_id = d["correct_id"]
        self._wrong_id = d["wrong_id"]


def generate_prompts(task_name: str, tokenizer, n_prompts: int = 40):
    if task_name == "rti":
        from lib.tasks.rti.prompts import make_rti_prompts
        raw = make_rti_prompts(tokenizer, n=n_prompts, seed=42)
        return [_RtiPromptAdapter(d, tokenizer) for d in raw]
    if task_name not in TASK_REGISTRY:
        return []
    builder = TASK_REGISTRY[task_name]
    if task_name == "buffalo":
        return builder(tokenizer, seed=42)[:n_prompts]
    return builder(tokenizer, n_prompts=n_prompts, seed=42)


def get_token_ids(prompts, tokenizer) -> tuple[list[int], list[int]]:
    correct_ids, incorrect_ids = [], []
    for p in prompts:
        ct = tokenizer.encode(p.target_correct)
        it = tokenizer.encode(p.target_incorrect)
        if len(ct) >= 1 and len(it) >= 1:
            correct_ids.append(ct[0])
            incorrect_ids.append(it[0])
    return correct_ids, incorrect_ids


# ---------------------------------------------------------------------------
# Logit-diff helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def logit_diff_from_logits(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    last_logits = logits[0, -1]
    return (last_logits[correct_id] - last_logits[incorrect_id]).item()


@torch.no_grad()
def compute_logit_diffs(model, prompts, correct_ids, incorrect_ids) -> list[float]:
    diffs = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model(tokens)
        diffs.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))
    return diffs


# ---------------------------------------------------------------------------
# Mean activation calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_mean_z(model, prompts, n_calibration: int = 100) -> torch.Tensor:
    """Compute mean hook_z per (layer, head) for ablation baselines."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    mean_z = torch.zeros(n_layers, n_heads, d_head)
    count = 0
    for p in prompts[:n_calibration]:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]
            mean_z[L] += z[0, -1].cpu()
        count += 1
    if count > 0:
        mean_z /= count
    return mean_z


# ---------------------------------------------------------------------------
# Ablation hooks
# ---------------------------------------------------------------------------

def make_ablation_hook(heads_by_layer: dict[int, list[int]], mean_z: torch.Tensor,
                       ablation_type: str = "mean"):
    """Return list of (hook_name, hook_fn) for the given ablation type."""
    hooks = []
    for layer, head_list in heads_by_layer.items():
        def _hook(z, hook, _layer=layer, _heads=head_list):
            for H in _heads:
                if ablation_type == "zero":
                    z[0, :, H, :] = 0.0
                elif ablation_type == "mean":
                    z[0, :, H, :] = mean_z[_layer, H].to(z.device)
                elif ablation_type == "noise":
                    std = z[0, :, H, :].std()
                    z[0, :, H, :] += torch.randn_like(z[0, :, H, :]) * std
                elif ablation_type == "soft":
                    z[0, :, H, :] *= 0.1
                elif ablation_type == "mean_last":
                    z[0, -1, H, :] = mean_z[_layer, H].to(z.device)
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))
    return hooks


def heads_to_layer_dict(heads: set[tuple[int, int]]) -> dict[int, list[int]]:
    d: dict[int, list[int]] = {}
    for L, H in heads:
        d.setdefault(L, []).append(H)
    return d


# ---------------------------------------------------------------------------
# Faithfulness / completeness helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_faithfulness(model, prompts, correct_ids, incorrect_ids,
                         circuit_heads: set[tuple[int, int]], mean_z: torch.Tensor) -> float:
    """Faithfulness = logit_diff(circuit_only) / logit_diff(full).

    Ablate all NON-circuit heads, keep circuit heads intact.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)
    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, "mean")

    faith_num, faith_den = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        faith_num += ablated_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


@torch.no_grad()
def compute_completeness(model, prompts, correct_ids, incorrect_ids,
                         circuit_heads: set[tuple[int, int]], mean_z: torch.Tensor) -> float:
    """Completeness = 1 - logit_diff(ablate_circuit) / logit_diff(full).

    Ablate circuit heads, keep non-circuit heads intact.
    """
    circuit_by_layer = heads_to_layer_dict(circuit_heads)
    hooks = make_ablation_hook(circuit_by_layer, mean_z, "mean")

    comp_num, comp_den = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        comp_num += ablated_ld
        comp_den += clean_ld

    if abs(comp_den) < 1e-8:
        return 0.0
    return 1.0 - comp_num / comp_den


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(results: list[EvalResult] | dict, filename: str) -> Path:
    path = DATA_DIR / filename
    if isinstance(results, list):
        data = [r.to_dict() for r in results]
    else:
        data = results
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_jsonable)
    log(f"Saved {path.name} ({path.stat().st_size / 1024:.1f}KB)")
    return path


def load_results(filename: str) -> Any:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_common_args(description: str = "Circuit evaluator"):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to evaluate (default: all circuit tasks)")
    parser.add_argument("--n-prompts", type=int, default=40, help="Prompts per task")
    parser.add_argument("--n-random-baselines", type=int, default=100,
                        help="Random baseline iterations")
    parser.add_argument("--out", default=None, help="Output filename override")
    return parser


# ---------------------------------------------------------------------------
# Literature baselines (from task_reference_baselines.py + MIB)
# ---------------------------------------------------------------------------

LITERATURE_BASELINES = {
    "ioi": {"faithfulness": 0.87, "das_iia": 0.95, "eap_auroc": 0.90},
    "greater_than": {"faithfulness": 0.80, "eap_auroc": 0.85},
    "induction": {"faithfulness": 0.75, "eap_auroc": 0.80},
    "sva": {"faithfulness": 0.70, "das_iia": 0.60, "eap_auroc": 0.40},
    "gendered_pronoun": {"faithfulness": 0.85, "eap_auroc": 0.72},
    "copy_suppression": {"faithfulness": 0.65, "eap_auroc": 0.74},
    "acronym": {"eap_auroc": 0.81},
    "rti": {"eap_auroc": 0.39},
}
