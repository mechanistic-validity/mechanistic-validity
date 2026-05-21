"""Shared infrastructure for the circuit evaluation pipeline.

Provides model loading, circuit/prompt access, logit-diff computation,
mean-activation calibration, and result serialization. All evaluator
scripts import from here.
"""
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "MV_OUTPUT_DIR",
    os.environ.get("INSTRUMENT_DATA_DIR", str(SCRIPT_DIR / "data")),
))
DATA_DIR.mkdir(exist_ok=True)

from mechval.registry import list_tasks, load_task


def get_circuit(task: str) -> dict:
    return load_task(task).get_circuit().to_dict()


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

CIRCUIT_TASKS = list_tasks(source="published")
EXPERIMENTAL_TASKS = list_tasks(source="experimental")
ALIAS_TASKS = list_tasks(source="ours")
ALL_TASKS = sorted(list_tasks())
EVALUABLE_TASKS = sorted(list_tasks(has_circuit=True))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InstrumentInfo:
    """Method citation and description for an instrument."""
    name: str
    paper_ref: str | None = None
    paper_cite: str | None = None
    description: str | None = None
    category: str | None = None
    tier: str | None = None
    origin: str | None = None
    subcategory: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


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
    faithfulness_curve: dict[float, float] | None = None
    cpr: float | None = None
    cmd: float | None = None
    instrument_info: InstrumentInfo | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = _sanitize_for_json(d["metadata"])
        if d["instrument_info"] is None:
            del d["instrument_info"]
        return d


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, range):
        return list(v)
    return v


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types to plain Python types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_sanitize_for_json(v) for v in obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, range):
        return list(obj)
    return obj


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

def generate_prompts(task_name: str, tokenizer, n_prompts: int = 40):
    try:
        task = load_task(task_name)
    except ValueError:
        return []
    return task.get_prompts(tokenizer, n_prompts=n_prompts, seed=42)


def get_token_ids(prompts, tokenizer) -> tuple[list[int], list[int]]:
    correct_ids, incorrect_ids = [], []
    for p in prompts:
        scoring = p.metadata.get("scoring", "text") if hasattr(p, "metadata") and p.metadata else "text"
        if scoring == "by_id":
            try:
                correct_ids.append(int(p.target_correct))
                incorrect_ids.append(int(p.target_incorrect))
            except (ValueError, TypeError):
                continue
        else:
            ct = tokenizer.encode(p.target_correct, add_special_tokens=False)
            it = tokenizer.encode(p.target_incorrect, add_special_tokens=False)
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
# Provenance — every output JSON is self-documenting
# ---------------------------------------------------------------------------

def _git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _pkg_version() -> str:
    try:
        from importlib.metadata import version
        return version("mechanistic-validity")
    except Exception:
        return "dev"


def get_provenance() -> dict[str, Any]:
    """Snapshot of when/where/what-version this run happened."""
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_version": _pkg_version(),
        "git": _git_info(),
    }


def args_to_config(args: Any) -> dict[str, Any]:
    """Convert an argparse Namespace (or any object with __dict__) to a config dict."""
    if args is None:
        return {}
    d = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {k: _sanitize_for_json(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(results: list[EvalResult] | dict, filename: str,
                 config: dict | None = None, args: Any = None) -> Path:
    """Save results as a self-documenting JSON with provenance envelope.

    The output format:
        {
          "provenance": { timestamp, git, version, ... },
          "config": { model, device, tasks, n_prompts, ... },
          "results": [ ... ]
        }

    Instruments call: save_results(results, "01_foo.json", args=args)
    """
    path = DATA_DIR / filename
    if isinstance(results, list):
        data = [r.to_dict() for r in results]
    else:
        data = results

    cfg = config or args_to_config(args)

    envelope = {
        "provenance": get_provenance(),
        "config": cfg,
        "results": data,
    }
    with open(path, "w") as f:
        json.dump(envelope, f, indent=2, default=_jsonable)
    log(f"Saved {path.name} ({path.stat().st_size / 1024:.1f}KB)")
    return path


def save_incremental(result: EvalResult, filename: str) -> Path:
    """Append one result as a JSONL line — survives crashes between tasks."""
    path = DATA_DIR / filename
    line = json.dumps(result.to_dict(), default=_jsonable)
    with open(path, "a") as f:
        f.write(line + "\n")
    log(f"Incremental save -> {path.name}")
    return path


def load_completed_tasks(filename: str) -> tuple[set[str], list[EvalResult]]:
    """Load previously completed tasks from a JSONL file for resumption.

    Returns (set of completed task names, list of EvalResult dicts).
    """
    path = DATA_DIR / filename
    completed = set()
    results = []
    if not path.exists():
        return completed, results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                task = d.get("metadata", {}).get("task", "")
                if task:
                    completed.add(task)
                results.append(d)
            except json.JSONDecodeError:
                continue
    if completed:
        log(f"Resuming: {len(completed)} tasks already done in {filename}")
    return completed, results


def load_results(filename: str) -> Any:
    """Load results. Handles both envelope format and legacy flat lists."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def load_results_full(filename: str) -> dict | None:
    """Load the full envelope (provenance + config + results)."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def set_data_dir(path: str | Path) -> Path:
    global DATA_DIR
    DATA_DIR = Path(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def parse_common_args(description: str = "Circuit evaluator"):
    import argparse

    class _Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            ns = super().parse_args(args, namespace)
            if getattr(ns, "data_dir", None):
                set_data_dir(ns.data_dir)
            return ns

    parser = _Parser(description=description)
    parser.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to evaluate (default: all circuit tasks)")
    parser.add_argument("--n-prompts", type=int, default=40, help="Prompts per task")
    parser.add_argument("--n-random-baselines", type=int, default=100,
                        help="Random baseline iterations")
    parser.add_argument("--out", default=None, help="Output filename override")
    parser.add_argument("--data-dir", default=None,
                        help="Output directory for results (default: src/metrics/data/)")
    return parser


# ---------------------------------------------------------------------------
# Literature baselines (loaded from task registry)
# ---------------------------------------------------------------------------

def _build_literature_baselines() -> dict[str, dict]:
    baselines: dict[str, dict] = {}
    for task_id in list_tasks():
        try:
            task = load_task(task_id)
            b = task.get_baselines()
            if b:
                baselines[task_id] = b
        except Exception:
            pass
    return baselines


LITERATURE_BASELINES = _build_literature_baselines()


def get_task_baselines(task_id: str) -> dict:
    """Get baselines for a specific task from the registry."""
    return LITERATURE_BASELINES.get(task_id, {})
