#!/usr/bin/env python3
"""
Nebius Cloud H100 orchestration for the MLOps assignment.

Provisions an H100 VM, runs the requested phases, copies results back,
and tears the VM (and its boot disk) down.

Usage:
    uv run python scripts/nebius_run.py                         # Phase 1 + 5
    uv run python scripts/nebius_run.py --phases 1,5,6          # + load test
    uv run python scripts/nebius_run.py --keep-vm               # leave VM for manual work

Required in .env:
    NEBIUS_PARENT_ID      Project ID (console.nebius.ai URL: /projects/<id>)
    NEBIUS_SUBNET_ID      Subnet ID  (console → VPC → Subnets)
    SSH_PRIVATE_KEY_PATH  Local SSH private key (~/.ssh/id_ed25519)
    HF_TOKEN              HuggingFace token for model download (optional if model is public)

The nebius CLI must already be authenticated (nebius iam whoami must succeed).
Run `~/.nebius/bin/nebius iam auth` once to set that up.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEBIUS = str(Path.home() / ".nebius" / "bin" / "nebius")

# ---------------------------------------------------------------------------
# VM configuration (matches the working gpu_and_inference_hw setup)
# ---------------------------------------------------------------------------
SSH_USER           = "karke"
VM_GPU_PLATFORM    = "gpu-h100-sxm"
VM_INSTANCE_PRESET = "1gpu-16vcpu-200gb"
VM_IMAGE_FAMILY    = "ubuntu24.04-cuda13.0"
VM_DISK_SIZE_BYTES = 214748364800   # 200 GiB
VLLM_READY_TIMEOUT = 2400           # 40 min: covers model download + GPU load
SSH_READY_TIMEOUT  = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        sys.exit(".env not found — copy .env.example and fill in values")
    result: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip().strip('"').strip("'")
    # Expand ~ in path values so subprocess calls work correctly
    for k in ("SSH_PRIVATE_KEY_PATH",):
        if k in result:
            result[k] = str(Path(result[k]).expanduser())
    return result


def require(env: dict, *keys: str) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        sys.exit(f"Missing required .env vars: {', '.join(missing)}")


def sh(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def sh_out(cmd: list[str]) -> str:
    return sh(cmd, capture=True).stdout.strip()


def neb(*args: str) -> str:
    """Run nebius CLI, return stdout (always JSON via --format json)."""
    result = subprocess.run(
        [NEBIUS, *args, "--format", "json"],
        check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


def _ssh_opts(key: str) -> list[str]:
    return ["-i", key, "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]


def remote(ip: str, key: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return sh(["ssh", *_ssh_opts(key), f"{SSH_USER}@{ip}", cmd], check=check)


def scp_get(ip: str, key: str, remote_path: str, local_path: str) -> None:
    sh(["scp", "-i", key, "-o", "StrictHostKeyChecking=no",
        f"{SSH_USER}@{ip}:{remote_path}", local_path])


# ---------------------------------------------------------------------------
# VM lifecycle
# ---------------------------------------------------------------------------

def create_disk(env: dict, suffix: str) -> str:
    """Create a boot disk; return its ID."""
    print("\n[1a/7] Creating boot disk …")
    raw = neb(
        "compute", "disk", "create",
        "--name",               f"mlops-eval-disk-{suffix}",
        "--parent-id",          env["NEBIUS_PARENT_ID"],
        "--type",               "network_ssd",
        "--block-size-bytes",   "4096",
        "--size-bytes",         str(VM_DISK_SIZE_BYTES),
        "--source-image-family-image-family", VM_IMAGE_FAMILY,
        "--disk-encryption-type", "disk_encryption_unspecified",
    )
    disk_id = json.loads(raw)["metadata"]["id"]
    print(f"  Disk ID: {disk_id}")
    return disk_id


def create_instance(env: dict, disk_id: str, suffix: str) -> str:
    """Create the GPU instance; return its ID."""
    print("\n[1b/7] Creating GPU instance …")
    # Prefer the .pub file (works even with passphrase-protected keys)
    pub_path = Path(env["SSH_PRIVATE_KEY_PATH"] + ".pub")
    if pub_path.exists():
        pub_key = pub_path.read_text().strip()
    else:
        pub_key = sh_out(["ssh-keygen", "-y", "-f", env["SSH_PRIVATE_KEY_PATH"]])

    network_iface = json.dumps([{
        "name": "eth0",
        "ip_address": {"allocationId": ""},
        "subnet_id": env["NEBIUS_SUBNET_ID"],
        "public_ip_address": {},
    }])

    cloud_init = f"""users:
 - name: {SSH_USER}
   sudo: ALL=(ALL) NOPASSWD:ALL
   shell: /bin/bash
   ssh_authorized_keys:
    - {pub_key}
"""

    raw = neb(
        "compute", "instance", "create",
        "--name",                     f"mlops-eval-{suffix}",
        "--parent-id",                env["NEBIUS_PARENT_ID"],
        "--stopped",                  "false",
        "--resources-platform",       VM_GPU_PLATFORM,
        "--resources-preset",         VM_INSTANCE_PRESET,
        "--boot-disk-existing-disk-id", disk_id,
        "--boot-disk-attach-mode",    "read_write",
        "--boot-disk-device-id",      "boot-disk",
        "--network-interfaces",       network_iface,
        "--cloud-init-user-data",     cloud_init,
        "--reservation-policy-policy", "auto",
    )
    instance_id = json.loads(raw)["metadata"]["id"]
    print(f"  Instance ID: {instance_id}")

    print("  Sending start command …")
    neb("compute", "instance", "start", "--id", instance_id)
    return instance_id


def wait_for_running(instance_id: str, timeout: int = 600) -> None:
    print(f"\n[2/7] Waiting for RUNNING state (up to {timeout}s) …")
    deadline = time.monotonic() + timeout
    for i in range(1, 1000):
        data = json.loads(neb("compute", "instance", "get", "--id", instance_id))
        state = (data.get("status", {}).get("state")
                 or data.get("metadata", {}).get("state")
                 or "UNKNOWN")
        print(f"  ({i}) state: {state}")
        if state == "RUNNING":
            return
        if time.monotonic() > deadline:
            sys.exit(f"Instance did not reach RUNNING within {timeout}s")
        time.sleep(10)


def get_public_ip(instance_id: str, timeout: int = 120) -> str:
    print("\n[3a/7] Waiting for public IP …")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = json.loads(neb("compute", "instance", "get", "--id", instance_id))
        for iface in (data.get("status", {}).get("network_interfaces", [])
                      or data.get("spec", {}).get("network_interfaces", [])):
            addr = (iface.get("public_ip_address", {}) or {}).get("address", "")
            if addr and addr != "null":
                ip = addr.split("/")[0]   # strip CIDR suffix if present
                print(f"  Public IP: {ip}")
                return ip
        time.sleep(10)
    sys.exit("Could not retrieve public IP within timeout")


def wait_for_ssh(ip: str, key: str, timeout: int = SSH_READY_TIMEOUT) -> None:
    print(f"\n[3b/7] Waiting for SSH on {ip} (up to {timeout}s) …")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["ssh", *_ssh_opts(key), f"{SSH_USER}@{ip}", "echo ok"],
            capture_output=True,
        )
        if r.returncode == 0:
            print("  SSH ready.")
            return
        time.sleep(10)
    sys.exit(f"SSH not available on {ip} within {timeout}s")


def destroy_resources(instance_id: str, disk_id: str) -> None:
    print(f"\n[7/7] Destroying instance {instance_id} and disk {disk_id} …")
    try:
        neb("compute", "instance", "delete", "--id", instance_id)
        print("  Instance deleted. Waiting 20s before deleting disk …")
        time.sleep(20)
    except Exception as e:
        print(f"  Warning: instance delete returned: {e}", file=sys.stderr)
    try:
        neb("compute", "disk", "delete", "--id", disk_id)
        print("  Disk deleted.")
    except Exception as e:
        print(f"  Warning: disk delete returned: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Remote setup
# ---------------------------------------------------------------------------

def setup_vm(ip: str, env: dict) -> None:
    key = env["SSH_PRIVATE_KEY_PATH"]
    print(f"\n[4/7] Syncing repo to {ip} …")
    sh([
        "rsync", "-az", "--progress",
        "--exclude=.venv/", "--exclude=data/", "--exclude=results/*.json",
        "--exclude=screenshots/", "--exclude=.git/",
        "-e", f"ssh -i {key} -o StrictHostKeyChecking=no",
        f"{ROOT}/",
        f"{SSH_USER}@{ip}:~/mlops-assignment/",
    ])

    print("  Installing system deps and uv …")
    remote(ip, key, (
        "sudo apt-get update -qq && "
        "sudo apt-get install -y -qq python3-dev build-essential docker.io docker-compose-plugin git curl && "
        "sudo systemctl enable --now docker && "
        f"sudo usermod -aG docker {SSH_USER}"
    ))
    remote(ip, key, "curl -LsSf https://astral.sh/uv/install.sh | sh")

    print("  Writing production .env on VM …")
    hf = env.get("HF_TOKEN", "")
    prod_env = "\n".join([
        "VLLM_BASE_URL=http://localhost:8000/v1",
        "VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507",
        "LLM_API_KEY=not-needed",
        f"HF_TOKEN={hf}",
        f"LANGFUSE_PUBLIC_KEY={env.get('LANGFUSE_PUBLIC_KEY', '')}",
        f"LANGFUSE_SECRET_KEY={env.get('LANGFUSE_SECRET_KEY', '')}",
        "LANGFUSE_HOST=http://localhost:3001",
    ])
    remote(ip, key, f"printf '%s' '{prod_env}' > ~/mlops-assignment/.env")

    print("  Installing Python deps (uv sync) …")
    remote(ip, key, "cd ~/mlops-assignment && ~/.local/bin/uv sync")

    print("  Starting o11y stack …")
    # newgrp doesn't persist across SSH calls; use sg to run docker with the new group
    remote(ip, key, "cd ~/mlops-assignment && sg docker -c 'docker compose up -d'")

    print("  Downloading BIRD data …")
    remote(ip, key, "cd ~/mlops-assignment && ~/.local/bin/uv run python scripts/load_data.py")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase1_start_vllm(ip: str, env: dict) -> None:
    key = env["SSH_PRIVATE_KEY_PATH"]
    print("\n[5/7] Starting vLLM (model download + load — up to 40 min) …")
    remote(ip, key,
           "cd ~/mlops-assignment && "
           "nohup bash scripts/start_vllm.sh > /tmp/vllm.log 2>&1 &")

    deadline = time.monotonic() + VLLM_READY_TIMEOUT
    while time.monotonic() < deadline:
        r = remote(ip, key, "curl -sf http://localhost:8000/health", check=False)
        if r.returncode == 0:
            print("  vLLM is healthy.")
            return
        elapsed = int(time.monotonic() - (deadline - VLLM_READY_TIMEOUT))
        print(f"  … {elapsed}s elapsed", flush=True)
        time.sleep(30)
    remote(ip, key, "tail -20 /tmp/vllm.log", check=False)
    sys.exit(f"vLLM did not become healthy within {VLLM_READY_TIMEOUT}s")


def _ensure_agent(ip: str, key: str) -> None:
    if remote(ip, key, "curl -sf http://localhost:8001/health", check=False).returncode != 0:
        print("  Starting agent server …")
        remote(ip, key,
               "cd ~/mlops-assignment && "
               "nohup ~/.local/bin/uv run uvicorn agent.server:app "
               "--host 0.0.0.0 --port 8001 > /tmp/agent.log 2>&1 &")
        time.sleep(6)


def phase5_eval(ip: str, env: dict) -> None:
    key = env["SSH_PRIVATE_KEY_PATH"]
    print("\n[6/7] Running baseline eval (Phase 5) …")
    _ensure_agent(ip, key)
    remote(ip, key,
           "cd ~/mlops-assignment && "
           "~/.local/bin/uv run python evals/run_eval.py --out results/eval_baseline.json")
    (ROOT / "results").mkdir(exist_ok=True)
    scp_get(ip, key, "~/mlops-assignment/results/eval_baseline.json",
            str(ROOT / "results" / "eval_baseline.json"))
    print("  → results/eval_baseline.json")


def phase6_load_test(ip: str, env: dict, rps: float, duration: int) -> None:
    key = env["SSH_PRIVATE_KEY_PATH"]
    print(f"\n[6b/7] Load test at {rps} RPS for {duration}s (Phase 6) …")
    _ensure_agent(ip, key)
    remote(ip, key,
           f"cd ~/mlops-assignment && "
           f"~/.local/bin/uv run python load_test/driver.py "
           f"--rps {rps} --duration {duration} --out results/load_test.json")
    scp_get(ip, key, "~/mlops-assignment/results/load_test.json",
            str(ROOT / "results" / "load_test.json"))
    print("  → results/load_test.json")


# ---------------------------------------------------------------------------
# SSH agent bootstrap
# ---------------------------------------------------------------------------

def _setup_ssh_agent(env: dict) -> None:
    """Start an ssh-agent for this process and add the key (using passphrase if set)."""
    key = env["SSH_PRIVATE_KEY_PATH"]
    passphrase = env.get("SSH_KEY_PASSPHRASE", "")

    # Start a fresh agent scoped to this process.
    # ssh-agent -s emits lines like: SSH_AUTH_SOCK=/tmp/...; export SSH_AUTH_SOCK;
    result = subprocess.run(["ssh-agent", "-s"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if "=" in line and not line.startswith("echo"):
            k, _, rest = line.partition("=")
            v = rest.split(";")[0]   # drop "; export VAR;" suffix
            os.environ[k.strip()] = v.strip()

    if passphrase:
        # Use SSH_ASKPASS so ssh-add can read the passphrase non-interactively
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(f"#!/bin/sh\necho {shlex.quote(passphrase)}\n")
            askpass_path = f.name
        os.chmod(askpass_path, 0o700)
        add_env = {**os.environ, "SSH_ASKPASS": askpass_path,
                   "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":"}
        subprocess.run(["ssh-add", key], env=add_env, check=True, capture_output=True)
        Path(askpass_path).unlink(missing_ok=True)
    else:
        subprocess.run(["ssh-add", key], check=True)

    print("  SSH agent ready.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nebius Cloud H100 orchestration for the MLOps assignment"
    )
    parser.add_argument("--phases", default="1,5",
        help="Phases to run: 1=vLLM, 5=eval, 6=load-test  (default: 1,5)")
    parser.add_argument("--keep-vm", action="store_true",
        help="Do not destroy the VM after the run")
    parser.add_argument("--rps", type=float, default=10.0)
    parser.add_argument("--load-duration", type=int, default=300)
    args = parser.parse_args()
    phases = {int(p.strip()) for p in args.phases.split(",")}

    env = load_env()
    require(env, "NEBIUS_PARENT_ID", "NEBIUS_SUBNET_ID", "SSH_PRIVATE_KEY_PATH")

    _setup_ssh_agent(env)

    # Verify nebius CLI is authenticated before doing any work
    print("Verifying nebius CLI auth …")
    try:
        subprocess.run([NEBIUS, "iam", "whoami", "--format", "json"],
                       check=True, capture_output=True)
        print("  OK")
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(
            f"nebius CLI not found or not authenticated.\n"
            f"Install: curl -sSL https://storage.eu-north1.nebius.cloud/nebius-cli-releases/install.sh | bash\n"
            f"Auth:    {NEBIUS} iam auth"
        )

    suffix = time.strftime("%Y%m%d%H%M%S")
    disk_id: str | None = None
    instance_id: str | None = None
    ip: str | None = None

    try:
        disk_id = create_disk(env, suffix)
        instance_id = create_instance(env, disk_id, suffix)
        wait_for_running(instance_id)
        ip = get_public_ip(instance_id)
        wait_for_ssh(ip, env["SSH_PRIVATE_KEY_PATH"])
        setup_vm(ip, env)

        if 1 in phases:
            phase1_start_vllm(ip, env)
        if 5 in phases:
            phase5_eval(ip, env)
        if 6 in phases:
            phase6_load_test(ip, env, rps=args.rps, duration=args.load_duration)

        print("\nAll phases complete. Results are in results/")

    except (KeyboardInterrupt, Exception) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\nInterrupted.")
        else:
            print(f"\nERROR: {exc}", file=sys.stderr)
        if not args.keep_vm and instance_id:
            print("Cleaning up resources …")
            destroy_resources(instance_id, disk_id or "")
        elif args.keep_vm and instance_id:
            print(f"\n--keep-vm: VM is still running at {ip}")
            print(f"  SSH:         ssh -i {env['SSH_PRIVATE_KEY_PATH']} {SSH_USER}@{ip}")
            print(f"  Destroy:     {NEBIUS} compute instance delete --id {instance_id}")
            print(f"               {NEBIUS} compute disk delete --id {disk_id}")
        sys.exit(1 if not isinstance(exc, KeyboardInterrupt) else 0)

    if not args.keep_vm:
        destroy_resources(instance_id, disk_id)
    else:
        print(f"\n--keep-vm: VM left running at {ip}")
        print(f"  SSH:     ssh -i {env['SSH_PRIVATE_KEY_PATH']} {SSH_USER}@{ip}")
        print(f"  Destroy: {NEBIUS} compute instance delete --id {instance_id}")
        print(f"           {NEBIUS} compute disk delete --id {disk_id}")


if __name__ == "__main__":
    main()
