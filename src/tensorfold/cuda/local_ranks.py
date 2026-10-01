"""``tensorfold serve MODEL --tp N`` without --master: every rank on this host's GPUs, one process a GPU.

This process becomes rank 0 (it serves HTTP) and starts ranks 1 .. N-1 as children with the same arguments plus
``--rank r --master 127.0.0.1``. Each rank sees its own GPU as device 0 and the others after it
(``CUDA_VISIBLE_DEVICES=own,others``), so code that means "this rank's GPU" by device 0 keeps working while NCCL can
still reach peers. ``TF_LOCAL_RANKS=1`` tells the engines the ranks share host memory (``fastcomm``). The children
die with this process, and a child that dies takes the server down with it.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading


def gpus() -> list[str]:
    """This host's GPU ids, in CUDA_VISIBLE_DEVICES's order when it is set, without starting CUDA here."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return [v.strip() for v in visible.split(",") if v.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"--tp without --master needs nvidia-smi to count this host's GPUs ({exc}); or set "
                         "CUDA_VISIBLE_DEVICES") from None
    return [line.strip() for line in out.splitlines() if line.strip()]


def order(ids: list[str], rank: int) -> str:
    """Rank ``rank``'s CUDA_VISIBLE_DEVICES: its GPU first (device 0), the others after (TF_ISOLATE_GPUS=1: alone)."""

    own = ids[rank]
    if os.environ.get("TF_ISOLATE_GPUS") == "1":
        return own
    return ",".join([own] + [i for i in ids if i != own])


def _die_with_parent() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)          # PR_SET_PDEATHSIG: SIGTERM when rank 0 goes
    except OSError:
        pass


def spawn(tp: int, argv: list[str] | None = None) -> list[subprocess.Popen]:
    """Start ranks 1 .. tp-1 (this process stays rank 0); call before anything starts CUDA here."""

    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise RuntimeError("local ranks must start before CUDA does in rank 0")
    ids = gpus()
    if len(ids) < tp:
        raise ValueError(f"--tp {tp} needs {tp} GPUs on this host; {len(ids)} are visible ({','.join(ids)})")
    argv = list(sys.argv[1:] if argv is None else argv)
    env = {**os.environ, "TF_LOCAL_RANKS": "1", "TENSORFOLD_NO_LIVE": "1"}
    children = []
    for rank in range(1, tp):
        child_env = {**env, "CUDA_VISIBLE_DEVICES": order(ids, rank)}
        cmd = [sys.executable, "-m", "tensorfold", *argv, "--rank", str(rank), "--master", "127.0.0.1",
               "--no-update-check"]
        children.append(subprocess.Popen(cmd, env=child_env, preexec_fn=_die_with_parent))
    os.environ["TF_LOCAL_RANKS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = order(ids, 0)
    _watch(children)
    return children


def _watch(children: list[subprocess.Popen]) -> None:
    """A rank that exits takes rank 0 down (its peers would wait on it forever); rank 0's exit stops the rest."""

    import atexit

    def stop_all() -> None:
        for p in children:
            if p.poll() is None:
                p.terminate()

    atexit.register(stop_all)

    def watch(p: subprocess.Popen, rank: int) -> None:
        code = p.wait()
        if code != 0:
            print(f"[tensorfold] rank {rank} exited with code {code}: stopping every rank", flush=True)
            stop_all()
            os.kill(os.getpid(), signal.SIGTERM)

    for rank, p in enumerate(children, start=1):
        threading.Thread(target=watch, args=(p, rank), daemon=True).start()
