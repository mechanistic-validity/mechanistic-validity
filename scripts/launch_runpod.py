#!/usr/bin/env python3
"""
Simple RunPod launcher for mechanistic-validity G instruments.

Launches one pod per instrument, each runs on GPU with incremental JSONL saves.
Results are uploaded to W&B as artifacts before self-stop.

Commands:
    launch           Launch pods for G instruments (dry run by default, add --go)
    status           List running pods
    terminate <id>   Terminate one pod

Usage:
    # Dry run
    uv run python scripts/launch_pod.py launch --instruments g1 g4 g5

    # Actually launch
    uv run python scripts/launch_pod.py launch --instruments g1 g4 g5 --go

    # All G instruments
    uv run python scripts/launch_pod.py launch --go

    # Check status
    uv run python scripts/launch_pod.py status
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/mechanistic-validity/mechanistic-validity.git"
BRANCH = "main"
DEFAULT_IMAGE = "runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04"

GPU_PREFERENCE = [
    "NVIDIA RTX A4000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
]

INSTRUMENTS = {
    "g1": {"script": "82_path_identification.py", "name": "G1 Path Identification"},
    "g2": {"script": "83_edge_necessity.py", "name": "G2 Edge Necessity"},
    "g3": {"script": "84_path_specificity.py", "name": "G3 Path Specificity"},
    "g4": {"script": "85_compositional_sufficiency.py", "name": "G4 Compositional Sufficiency"},
    "g5": {"script": "86_graph_minimality.py", "name": "G5 Graph Minimality"},
}

TASKS = "epistemic_framing epistemic_tight epistemic_eap epistemic_expanded"


def make_startup_script(instrument_key: str, branch: str) -> str:
    info = INSTRUMENTS[instrument_key]
    script_name = info["script"]

    return f"""#!/bin/bash
set -euo pipefail

exec > >(tee /tmp/pod_full.log) 2>&1

cat > /tmp/_upload_and_stop.sh << 'STOPEOF'
#!/bin/bash
echo "[pod] Uploading results to W&B..."
cd /workspace/mechanistic-validity
pip install wandb -q 2>/dev/null
python3 -c "
import wandb, os, glob, datetime
data_dir = 'src/instruments/data'
files = glob.glob(f'{{data_dir}}/8*.json') + glob.glob(f'{{data_dir}}/8*.jsonl')
log_files = ['/tmp/pod_full.log']
all_files = [f for f in files + log_files if os.path.exists(f)]
if not all_files:
    print('[pod] No files to upload')
    exit(0)
ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
pod_id = os.environ.get('RUNPOD_POD_ID', 'unknown')
run = wandb.init(
    project='mechanistic-validity',
    entity='factorized-circuits',
    name=f'g-instruments-{{pod_id}}-{{ts}}',
    job_type='g_instrument',
)
artifact = wandb.Artifact(f'g-results-{{pod_id}}-{{ts}}', type='g_instrument_results')
for f in all_files:
    artifact.add_file(f)
run.log_artifact(artifact)
run.finish()
print(f'[pod] Uploaded {{len(all_files)}} files to W&B')
" 2>&1 || echo "[pod] WARNING: W&B upload failed"

echo "[pod] Self-stopping..."
if [[ -n "${{RUNPOD_POD_ID:-}}" ]] && [[ -n "${{RUNPOD_API_KEY:-}}" ]]; then
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \\
        -H "Content-Type: application/json" \\
        -d '{{"query": "mutation {{ podStop(input: {{podId: \\\\"'$RUNPOD_POD_ID'\\\\"}} ) {{ id desiredStatus }} }}"}}' || true
fi
sleep infinity
STOPEOF
chmod +x /tmp/_upload_and_stop.sh

trap 'echo "[pod] ERROR at line $LINENO"; bash /tmp/_upload_and_stop.sh' ERR

# --- Setup ---
GH_TOKEN="${{GITHUB_TOKEN:-${{RUNPOD_SECRET_GH_PAT:-}}}}"
WANDB_KEY="${{WANDB_API_KEY:-${{RUNPOD_SECRET_WANDB_API_KEY:-}}}}"
export WANDB_API_KEY="$WANDB_KEY"

echo "[pod] {info['name']} — starting"
echo "[pod] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Clone repo
cd /workspace
if [[ -d mechanistic-validity/.git ]]; then
    cd mechanistic-validity
    git fetch origin {branch}
    git reset --hard FETCH_HEAD
else
    git clone --branch {branch} "https://x-access-token:${{GH_TOKEN}}@github.com/mechanistic-validity/mechanistic-validity.git"
    cd mechanistic-validity
fi

# Install deps
pip install -e ".[dev]" 2>&1 | tail -5
python3 -c "import torch; print(f'torch {{torch.__version__}}, CUDA {{torch.cuda.is_available()}}')"

# --- Run instrument ---
cd src/instruments/structural/edge_analysis
echo "[pod] Running {script_name} on epistemic tasks..."
python3 {script_name} \\
    --tasks {TASKS} \\
    --device cuda \\
    --n-prompts 40 \\
    2>&1 | tee /tmp/experiment.log

echo "[pod] {info['name']} complete!"

# Upload and stop
bash /tmp/_upload_and_stop.sh
"""


class RunPodGraphQL:
    API_URL = "https://api.runpod.io/graphql"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _query(self, query: str) -> dict:
        import requests
        resp = requests.post(
            f"{self.API_URL}?api_key={self.api_key}",
            json={"query": query},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(data["errors"][0].get("message", str(data["errors"])))
        return data.get("data", {})

    def create_pod(self, *, name: str, image_name: str, gpu_type_id: str,
                   cloud_type: str = "ALL", container_disk_in_gb: int = 30,
                   volume_in_gb: int = 0, env: dict | None = None,
                   docker_args: str = "", ports: str = "22/tcp") -> dict:
        env_list = []
        for k, v in (env or {}).items():
            safe_v = v.replace("\\", "\\\\").replace('"', '\\"')
            env_list.append(f'{{ key: "{k}", value: "{safe_v}" }}')
        env_str = "[" + ", ".join(env_list) + "]"
        safe_docker = docker_args.replace("\\", "\\\\").replace('"', '\\"')

        query = f"""mutation {{
          podFindAndDeployOnDemand(input: {{
            cloudType: {cloud_type}
            gpuCount: 1
            volumeInGb: {volume_in_gb}
            containerDiskInGb: {container_disk_in_gb}
            gpuTypeId: "{gpu_type_id}"
            name: "{name}"
            imageName: "{image_name}"
            dockerArgs: "{safe_docker}"
            env: {env_str}
            ports: "{ports}"
            volumeMountPath: "/workspace"
          }}) {{
            id name desiredStatus costPerHr
            machine {{ podHostId gpuDisplayName }}
          }}
        }}"""
        data = self._query(query)
        return data["podFindAndDeployOnDemand"]

    def get_pods(self) -> list[dict]:
        query = """query {
          myself {
            pods {
              id name desiredStatus costPerHr
              runtime { uptimeInSeconds }
              machine { gpuDisplayName }
            }
          }
        }"""
        data = self._query(query)
        return data.get("myself", {}).get("pods", [])

    def terminate_pod(self, pod_id: str) -> dict:
        query = f"""mutation {{
          podTerminate(input: {{ podId: "{pod_id}" }})
        }}"""
        return self._query(query)


def load_env() -> dict:
    env: dict = {}
    for search_dir in [Path("."), Path(__file__).resolve().parent.parent,
                       Path.home() / "Documents" / "GitHub" / "factorization-circuits"]:
        for fname in (".env.local", ".env"):
            p = search_dir / fname
            if p.exists():
                for line in p.read_text().splitlines():
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        env.setdefault(k.strip(), v.strip())
    return env


def get_runpod(env: dict) -> RunPodGraphQL:
    key = env.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("ERROR: RUNPOD_API_KEY not found in .env or environment")
    return RunPodGraphQL(api_key=key)


def cmd_launch(args, env):
    rp = get_runpod(env)
    github_token = (env.get("GH_REPO_TOKEN")
                    or env.get("GITHUB_PERSONAL_ACCESS_TOKEN")
                    or env.get("GITHUB_TOKEN", ""))
    wandb_key = env.get("WANDB_API_KEY", "")

    instruments = args.instruments or ["g1", "g4", "g5"]

    print(f"Instruments   : {', '.join(instruments)}")
    print(f"Branch        : {args.branch}")
    print(f"GPU           : {args.gpu or 'auto (cheapest available)'}")
    print(f"GitHub PAT    : {'yes' if github_token else 'MISSING'}")
    print(f"W&B key       : {'yes' if wandb_key else 'MISSING'}")
    print()

    if not args.go:
        print("Dry run — pass --go to launch.")
        for inst in instruments:
            info = INSTRUMENTS[inst]
            print(f"  Would launch: {info['name']} ({info['script']})")
        return

    pod_env = {
        "GITHUB_TOKEN": github_token,
        "WANDB_API_KEY": wandb_key,
    }

    for inst in instruments:
        info = INSTRUMENTS[inst]
        startup = make_startup_script(inst, args.branch)
        startup_b64 = base64.b64encode(startup.encode()).decode()
        docker_cmd = f"bash -c 'echo {startup_b64} | base64 -d | bash'"
        ts = datetime.now(tz=timezone.utc).strftime("%H%M")
        pod_name = f"mv-{inst}-{ts}"

        gpu_types = [args.gpu] if args.gpu else GPU_PREFERENCE
        pod = None
        for gpu in gpu_types:
            for cloud in ("COMMUNITY", "SECURE"):
                try:
                    pod = rp.create_pod(
                        name=pod_name,
                        image_name=DEFAULT_IMAGE,
                        gpu_type_id=gpu,
                        cloud_type=cloud,
                        container_disk_in_gb=30,
                        env=pod_env,
                        docker_args=docker_cmd,
                    )
                    print(f"  Launched {pod['id']} — {info['name']} ({gpu}, {cloud}, "
                          f"${pod.get('costPerHr', '?')}/hr)")
                    break
                except Exception as e:
                    if args.verbose:
                        print(f"    {gpu} / {cloud}: {str(e)[:60]}")
            if pod:
                break
        if not pod:
            print(f"  ERROR: No GPU available for {info['name']}")


def cmd_status(args, env):
    rp = get_runpod(env)
    pods = rp.get_pods()
    mv_pods = [p for p in pods if p["name"].startswith("mv-")]
    if not mv_pods:
        print("No mechanistic-validity pods running.")
        return
    for p in mv_pods:
        gpu = p.get("machine", {}).get("gpuDisplayName", "?")
        uptime = p.get("runtime", {}).get("uptimeInSeconds", 0)
        mins = int(uptime / 60) if uptime else 0
        cost = p.get("costPerHr", "?")
        print(f"  {p['id']}  {p['name']:20s}  {p['desiredStatus']:10s}  "
              f"{gpu:30s}  {mins}min  ${cost}/hr")


def cmd_terminate(args, env):
    rp = get_runpod(env)
    rp.terminate_pod(args.pod_id)
    print(f"Terminated {args.pod_id}")


def main():
    parser = argparse.ArgumentParser(description="RunPod launcher for mechanistic-validity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    launch = sub.add_parser("launch", help="Launch G instrument pods")
    launch.add_argument("--instruments", nargs="+", choices=list(INSTRUMENTS.keys()),
                        help="Which instruments to run (default: g1 g4 g5)")
    launch.add_argument("--branch", default=BRANCH)
    launch.add_argument("--gpu", default=None, help="Force specific GPU type")
    launch.add_argument("--go", action="store_true", help="Actually launch (default: dry run)")
    launch.add_argument("--verbose", action="store_true")

    sub.add_parser("status", help="List running pods")

    term = sub.add_parser("terminate", help="Terminate a pod")
    term.add_argument("pod_id")

    args = parser.parse_args()
    env = load_env()

    if args.cmd == "launch":
        cmd_launch(args, env)
    elif args.cmd == "status":
        cmd_status(args, env)
    elif args.cmd == "terminate":
        cmd_terminate(args, env)


if __name__ == "__main__":
    main()
