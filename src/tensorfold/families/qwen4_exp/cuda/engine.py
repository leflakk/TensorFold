"""The Flash Next CUDA engine: MTP chains verified exactly on one GPU or two ranks in lockstep."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from tensorfold.cuda import prompt_precision
from . import CONFIDENCE, DEPTH

MAX_DEPTH = 15           # a verify window of at most 16 rows
KEEP_SERIAL = 4          # prompt states the serial engine keeps (they share its attention rows)
KEEP = 8                 # prompt states (one token before each end) a concurrent decoder keeps to resume from
# Reserve bounded tower workspace separately from its weights; override for measured deployments.
VISION_WORKSPACE = 4 * 2**30


def vision_workspace() -> int:
    import os

    value = os.environ.get("TENSORFOLD_VISION_WORKSPACE_MIB")
    if value is None or value == "":
        return VISION_WORKSPACE
    if not value.isdecimal() or int(value) > 16384:
        raise ValueError(f"TENSORFOLD_VISION_WORKSPACE_MIB: 0 to 16,384 MiB, not {value!r}")
    return int(value) * 2**20


class FlashNextEngine:
    """``eos``, ``generate`` (rank 0 or one GPU) and ``follow`` (rank 1), as ``tensorfold.cuda.server`` expects."""

    def __init__(self, model_dir: Path, *, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 draft_vocab: str | int | None = "default", max_len: int | None = None,
                 context_explicit: bool | None = None, tp: int = 1, rank: int = 0, master: str = "", port: int = 29551,
                 prefetch: bool = True, graphs: bool = True, streams: int = 1, ple_on_ssd: bool = False,
                 kv_dtype: str = "bf16", share: float = 0.0, vision: bool = False, vision_urls: bool = False) -> None:
        import torch

        from .exl3_pack import admission, extra_files, is_exl3

        from tensorfold.families import quant_method, read_config

        exl3 = is_exl3(model_dir)
        if (exl3 or quant_method(read_config(model_dir)) == "modelopt") and tp != 1:
            raise ValueError(f"{'EXL3 packs' if exl3 else 'NVFP4 checkpoints'} of Flash Next run on one GPU: drop --tp, "
                             "or serve the MLX checkpoint (TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP) on 2, 4 or 8")
        if vision and (streams < 2 or tp != 1):
            raise ValueError("image input on Flash Next runs on one GPU with --parallel 2 or more")
        if exl3 and ple_on_ssd:
            raise ValueError("--ple-on-ssd reads the MLX checkpoint's n-gram tables; an EXL3 pack maps its own table "
                             "from its file, so drop --ple-on-ssd")
        from .decode import Engine
        from .prompt_plan import choose as prompt_plan
        from .kvcache import BITS_OF, check as check_kv
        from .weights import draft_token_ids, load
        from tensorfold.cuda.capacity import admit, config, gather_ints
        from tensorfold.cuda.geometry import PREFILL_ROWS, gdn_geometry, indexed_stream_geometry, indexed_weights

        from .tp import WORLDS

        if tp not in WORLDS or rank not in range(tp):
            raise ValueError(f"rank {rank} of {tp}: Flash Next runs on {', '.join(map(str, WORLDS))} GPUs")
        import os

        if streams > 1 and tp > 1 and os.environ.get("TF_LOCAL_RANKS") != "1":
            raise ValueError(f"--parallel with --tp {tp} runs every rank on this host (rank 0 sends each round's "
                             f"plan through shared memory): start `tensorfold serve MODEL --tp {tp} --parallel "
                             f"{streams}` without --master")
        if not 0 <= int(depth) <= MAX_DEPTH:
            raise ValueError(f"MTP drafts a round: 0 to {MAX_DEPTH}, not {depth}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"MTP draft confidence: a probability from 0 to 1, not {confidence}")
        torch.cuda.set_device(0)
        self.tp, self.rank, self.depth, self.confidence = tp, rank, int(depth), float(confidence)
        self.streams = int(streams)
        self.kv_dtype = check_kv(kv_dtype)
        self.comm = None
        self.vision = None                   # the image tower (``QwenCudaVision``) with --vision
        ids = draft_token_ids(draft_vocab) if self.depth > 0 else None
        if tp > 1:
            from tensorfold.cuda.comm import NCCL

            if not master:
                raise ValueError(f"{tp} ranks need rank 0's address (master)")
            # ``tensorfold serve --tp N`` without --master starts every rank on this host (TF_LOCAL_RANKS=1)
            local = os.environ.get("TF_LOCAL_RANKS") == "1"
            self.comm = NCCL(rank, tp, master, port, **({"local": True} if local else {}))
            self.comm.barrier()
        more = {} if tp == 2 else {"world": tp}                 # gather_ints defaults to two ranks
        gather = (lambda values: gather_ints(torch, self.comm.all_gather, values, **more)) if tp > 1 else None
        each, mtp, bits = self.depth + 1, self.depth > 0, BITS_OF[self.kv_dtype]
        # paired prompt chunks (ranks on this host) hold a second half-chunk of prompt buffers
        import os as _os

        from tensorfold.cuda.geometry import fast_partials

        paired = fast_partials(tp) and _os.environ.get("TF_PREFILL_OVERLAP", "1") != "0"
        prompt_rows = PREFILL_ROWS + (PREFILL_ROWS // 2 if paired else 0)
        # one admission for one stream or many (every slot, the shared rows and kept snapshots), before any load
        geometry = ((lambda text: indexed_stream_geometry(text, streams, each, KEEP, mtp=mtp, kv_bits=bits,
                                                          world=tp))
                    if streams > 1 else
                    (lambda text: gdn_geometry(text, tp, each, indexed=True, mtp=mtp, kv_bits=bits,
                                               kept=KEEP_SERIAL + 1, prefill_rows=prompt_rows)))
        if exl3:
            geometry = admission(geometry)
        from tensorfold.vision.qwen_cuda import capacity_geometry, weight_transform as vision_weights

        workspace = vision_workspace() if vision else 0
        self.capacity_plan = admit(model_dir, max_len, context_explicit, torch,
                                   capacity_geometry(geometry, model_dir, vision, rank, workspace),
                                   vision_weights(indexed_weights(tp, mtp, mapped_tables=not ple_on_ssd,
                                                                 kv_heads=int(config(model_dir)["num_key_value_heads"])),
                                                 vision, rank),
                                   rank=rank, world=tp,
                                   gather=gather, extra_files=extra_files(model_dir) if exl3 else ())
        self.prefill_rows, prompt_workspace = (PREFILL_ROWS, 0) if exl3 else prompt_plan(
            self.capacity_plan, config(model_dir), torch.cuda.get_device_capability(), world=tp, vision=vision,
            fp8=prompt_precision.fp8())
        if prompt_workspace:
            peak = self.capacity_plan["total_bytes_estimate"] / 2**30
            print(f"[tensorfold] {self.prefill_rows}-row idle prompt workspace {prompt_workspace / 2**30:.2f} GiB; "
                  f"planned peak {peak:.2f} GiB at the admitted window", flush=True)
        self.max_len = self.capacity_plan["cache_slots"]
        if tp > 1:
            self._same_settings(torch, ids)
            # every partial sum and gather through shared host memory when the ranks share this host: a prompt
            # chunk's partials (bf16 or fp32), a window's fp32 ones, or gathered sampling candidates fit
            from tensorfold.cuda.geometry import PREFILL_ROWS as _ROWS
            from .tp import prefill_partials

            hidden = int(config(model_dir)["hidden_size"])
            cap = max(max(self.prefill_rows, _ROWS) * hidden * (2 if prefill_partials() == "bf16" else 4),
                      64 * hidden * 4, 8 << 20)
            self.comm.barrier()
            if getattr(self.comm, "use_fast", lambda cap: False)(cap):
                print(f"[tensorfold] rank {rank} of {tp}: partial sums through shared host memory "
                      f"({self.comm.fast.bytes / 2**20:.0f} MiB region; TF_COMM=nccl uses NCCL)", flush=True)
        from concurrent.futures import wait

        from tensorfold.cuda.direct_read import wait_all

        reads: list = []                              # the n-gram tables' pages, read while the weights load
        try:
            # one rank a host reads the n-gram tables ahead: the others map the same page cache
            self.table_host = prefetch and not ple_on_ssd and (rank == 0 or not getattr(self.comm, "local", False))
            # a discrete GPU with the host's RAM to spare pins the tables while the weights load (pages read later
            # could have been evicted by then and be read from disk again)
            early = False
            if self.table_host:
                from tensorfold.cuda.capacity import _meminfo, unified

                from .. import ple_bytes

                memory = _meminfo()
                early = (not unified(torch) and memory is not None
                         and memory["MemAvailable"] - 8 * 2**30 >= ple_bytes(model_dir))
            w = load(model_dir, mtp=self.depth > 0, tp=(rank, tp) if tp > 1 else None,
                     draft_vocab=draft_vocab if self.depth > 0 else None, ple_on_ssd=ple_on_ssd,
                     table_reads=reads if self.table_host else None, table_lock=early)
        except BaseException:
            wait(reads)                               # a failed load leaves no table read behind it
            raise
        tables_read = bool(reads)
        waited = time.perf_counter()
        wait_all(reads)                               # raises a table read's error
        waited = time.perf_counter() - waited
        w.comm = self.comm
        if self.comm is not None:
            self.comm.ready("loading")               # a peer stuck loading is named, not waited on in NCCL
        if self.depth > 0 and w.mtp is None:
            raise ValueError("this checkpoint has no MTP head, which Flash Next's CUDA engine drafts with: use one "
                             "that has it, or --no-drafts for the serial reference (one token a round)")
        self.w = w
        from tensorfold.cuda.markers import resume_points

        self.points = resume_points(model_dir)          # a prompt's message starts to keep states at, or None
        if vision:
            from tensorfold.vision.qwen_cuda import QwenCudaVision

            self.vision = QwenCudaVision(model_dir, torch.device("cuda", 0),
                                         allow_urls=vision_urls)
            torch.cuda.empty_cache()
            print(f"[tensorfold] vision: image input, a "
                  f"{self.vision.weight_bytes / 2**30:.2f} GiB tower with {vision_workspace() / 2**30:.2f} GiB of "
                  f"workspace reserved{'; https URLs allowed' if vision_urls else ''}", flush=True)
        # ``streams`` > 1: up to that many requests decoded together, every stream's chain in one forward
        self.concurrent = streams > 1
        self.multi = self.scheduler = self.leader = self.ring = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.e = None
            self.multi = MultiDecoder(w, slots=streams, capacity=self.max_len, depth=self.depth,
                                      confidence=self.confidence, keep=KEEP, points=self.points,
                                      kv_dtype=self.kv_dtype, share=share, vision=self.vision,
                                      prefill_rows=self.prefill_rows, workspace_bytes=prompt_workspace)
            decoder = self.multi
            if tp > 1:                       # rank 0 decides each call; every rank runs it (``multi_tp``)
                from .multi_tp import Leader, Ring

                self.ring = Ring(self.comm.store, rank, tp)
                decoder = self.leader = Leader(self.multi, self.ring) if rank == 0 else None
            if decoder is not None:
                self.scheduler = Scheduler(decoder, max_streams=streams)
        else:
            # TF_GRAPHS=0: eager windows (how much CUDA graphs save a round, next to concurrent rounds, which run eager)
            self.e = Engine(w, capacity=self.max_len, max_rows=max(8, self.depth + 1),
                            graphs=graphs and os.environ.get("TF_GRAPHS", "1") != "0",
                            kv_dtype=self.kv_dtype, prefill_rows=self.prefill_rows)
        started = time.perf_counter()
        locked = False
        if self.table_host:                           # the n-gram tables' pages, read now rather than by requests
            tables = {id(layer.ple.table): layer.ple.table for layer in w.layers if layer.ple is not None}
            size = sum(t.nbytes for t in tables.values())
            # pinned pages are no longer reclaimable: lock only what the startup budget leaves room for (a
            # discrete GPU's budget is its own memory: the host's free memory, less 8 GiB, decides there)
            room = self.capacity_plan["budget_bytes"] - self.capacity_plan["total_bytes_estimate"]
            from tensorfold.cuda.capacity import _meminfo, unified

            memory = _meminfo()
            if not unified(torch) and memory is not None:
                room = memory["MemAvailable"] - 8 * 2**30
            for table in tables.values():
                if not tables_read:
                    table.prefetch()                  # eight readers first: mlock alone faults the pages in one by one
                locked = getattr(table, "early_locked", False) or (room >= size and table.lock())
        read_s = time.perf_counter() - started
        captured = self.e.graphs.warm(self.depth + 1) if self.e is not None and self.e.graphs is not None else 0
        started = time.perf_counter()
        if self.concurrent:
            self.multi.warm()                       # with round graphs: each stream count's rounds captured
            rounds = self.multi.rounds
            captured = rounds.captures if rounds is not None and rounds.capture else 0
        else:
            from .decode import warm

            warm(self.e)
        if self.vision is not None:
            self.vision.warm()
            torch.cuda.empty_cache()
        warm_s = time.perf_counter() - started
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.served = 0
        self.cache: list[tuple[list[int], dict]] = []    # (committed ids, what resuming from them needs)
        self.serial = None                                # the serial requests' engine, made on first use
        rule = (f"1 to {self.depth} MTP drafts a round, a chain stops before a later draft under "
                f"{self.confidence:.0%}" if self.depth else "no drafts: the serial reference, one token a round")
        where = (f"up to {streams} streams, each growing to {self.context_window} prompt/reply tokens while memory "
                 f"lasts ({self.multi.memory_gate.room / 2**30:.1f} GiB free for their caches, "
                 f"{self.multi.window_bytes / 2**30:.2f} GiB for one at the full window), "
                 f"{'rounds in CUDA graphs' if captured else 'eager'}" if self.concurrent else
                 f"{self.context_window}-token prompt/reply window; {self.max_len}-token cache")
        if ple_on_ssd:
            how = "read from SSD at each lookup"
        elif tables_read:                             # read during the load: the wait after it, then any lock
            how = f"read alongside the weights ({waited:.1f}s after them)" + (
                f", locked in memory in {read_s:.1f}s" if locked else "")
        else:
            how = f"{'locked in memory' if locked else 'read'} in {read_s:.1f}s"
        kv = "" if self.kv_dtype == "bf16" else f"; {self.kv_dtype} KV cache (fp16 scale per 32 values)"
        print(f"[tensorfold] Flash Next on CUDA: {rule}; {where}{kv}; n-gram tables {how}; {captured} "
              f"decode graphs captured; idle prompt pieces {self.prefill_rows} rows; "
              f"prompt kernels warmed in {warm_s:.1f}s", flush=True)

    def _same_settings(self, torch, ids) -> None:
        """Both ranks must decode with the same rule, context, draft vocabulary and KV cache, or they would fall out of step: refuse to start otherwise."""

        from .kvcache import BITS_OF

        total = int(ids.sum()) if ids is not None else -1
        from .tp import decode_partials, prefill_partials

        partials = 2 * (decode_partials() == "bf16") + (prefill_partials() == "bf16")     # every rank sends alike
        mine = torch.tensor([self.depth, round(self.confidence * 1e6), self.max_len,
                             len(ids) if ids is not None else -1, total, BITS_OF[self.kv_dtype], partials,
                             int(getattr(self, "streams", 1)), int(prompt_precision.fp8())], dtype=torch.int64,
                            device="cuda")
        world = getattr(self.comm, "world", None) or getattr(self, "tp", 2)
        every = torch.empty((world * mine.numel(),), dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, every)
        every = every.view(world, -1).cpu()
        for r in range(1, world):
            prompt_precision.same_on_ranks(int(every[0, -1]), int(every[r, -1]))
            if not torch.equal(every[0], every[r]):
                raise RuntimeError(f"the ranks were started with different settings (drafts, confidence, context, "
                                   f"draft vocabulary, KV cache, TF_*_PARTIALS, --parallel): rank 0 "
                                   f"{every[0].tolist()}, rank {r} {every[r].tolist()}")

    def _key(self, n: int) -> str:
        return f"tensorfold/flashnext/request/{n}"

    def shutdown(self) -> None:
        """Rank 0: tell the other ranks to leave ``follow``."""

        if self.tp > 1 and self.rank == 0:
            if getattr(self, "leader", None) is not None:
                self.leader.stop()
                return
            for r in range(1, self.tp):
                self.comm.store.set(f"{self._key(self.served)}/{r}", json.dumps({"stop": True}))

    def _share(self, prompt: list[int], max_tokens: int, sampling, draft: bool, cached: int, constraint=None,
               stop_eos: bool = True, mtp=None) -> tuple:
        from tensorfold.engine.grammar import pack

        points = getattr(self, "points", None)
        body = {"prompt": prompt, "max_tokens": max_tokens, "draft": bool(draft), "cached": int(cached),
                "points": list(points(prompt)) if draft and points is not None else [],
                "stop_eos": bool(stop_eos),
                "sampling": None if sampling is None else [int(sampling.seed), float(sampling.temperature),
                                                           int(sampling.top_k), float(sampling.top_p),
                                                           float(sampling.min_p)],
                "grammar": pack(constraint),                 # rank 1 walks and masks the same rows
                "mtp": None if mtp is None else [int(mtp[0]), float(mtp[1])]}
        text = json.dumps(body)
        for r in range(1, max(2, int(getattr(self, "tp", 2) or 2))):    # one key a follower: each deletes its own
            self.comm.store.set(f"{self._key(self.served)}/{r}", text)
        return self._unpack(text)

    def _receive(self) -> tuple | None:
        from torch.distributed import DistNetworkError

        key = f"{self._key(self.served)}/{self.rank}"
        while True:
            try:
                self.comm.store.wait([key], timedelta(hours=1))
                break
            except DistNetworkError:                        # rank 0 is gone: leave ``follow``
                print(f"[tensorfold] rank 0 closed the connection; rank {self.rank} stops", flush=True)
                return None
            except Exception:                               # noqa: BLE001  (no request within the hour: wait on)
                continue
        text = self.comm.store.get(key).decode()
        self.comm.store.delete_key(key)
        return self._unpack(text)

    @staticmethod
    def _unpack(text: str) -> tuple | None:
        from tensorfold.engine.exact_sampling import Sampling

        body = json.loads(text)
        if body.get("stop"):
            return None
        s = body["sampling"]
        return (body["prompt"], body["max_tokens"], None if s is None else Sampling(*s),
                body["draft"], body["cached"], body.get("grammar") or [], body.get("stop_eos", True),
                body.get("points", []), body.get("mtp"))

    @property
    def context_window(self) -> int:
        """Prompt and reply capacity after reserving speculative scratch positions."""

        return max(0, self.max_len - self.depth - 1)

    def _limit(self, prompt: list[int], max_tokens: int) -> int:
        room = self.max_len - len(prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.max_len}-token context")
        return max(1, min(max_tokens, room))

    def _resume(self, prompt: list[int]):
        """The longest kept state the prompt extends (with at least one new token), or None."""

        best = None
        for ids, snap in self.cache:
            if len(ids) < len(prompt) and prompt[:len(ids)] == ids and (best is None or len(ids) > len(best[0])):
                best = (ids, snap)
        return best

    def _start_from(self, hit) -> None:
        """Before a prefill: resuming overwrites the cache rows past the kept prefix, so the states that extend it go; a fresh prompt overwrites them all."""

        if hit is None:
            self.cache = []
        else:
            n = len(hit[0])
            self.cache = [c for c in self.cache if len(c[0]) < n and hit[0][:len(c[0])] == c[0]] + [hit]

    def _remember(self, ids: list[int], snap: dict) -> None:
        self.cache = [c for c in self.cache if c[0] != ids][-(KEEP_SERIAL - 1):] + [(ids, snap)]

    @property
    def supports_logprobs(self) -> bool:
        return self.tp == 1

    def _serial(self, prompt: list[int], max_tokens: int, sampling, on_tokens, constraint=None,
                stop_eos: bool = True, probabilities=None) -> dict[str, Any]:
        """One token a round from a fresh prefill in the serial engine's own state (no drafts, no kept states)."""

        import torch

        from .decode import prefill, serial_decode

        if self.serial is None:
            self.serial = self.e.twin()
        t0 = time.perf_counter()
        first = prefill(self.serial, prompt, sampling, mtp=False, constraint=constraint, probabilities=probabilities)
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "drafts": False}
        if (on_tokens is not None and on_tokens([first])) or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        res = serial_decode(self.serial, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                            constraint=constraint, probabilities=probabilities)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def _decode(self, prompt: list[int], max_tokens: int, sampling, on_tokens, hit, constraint=None,
                stop_eos: bool = True, probabilities=None, points=None, mtp=None) -> dict[str, Any]:
        import torch

        from .decode import entry_end, mtp_decode, prefill, serial_decode

        t0 = time.perf_counter()
        self._start_from(hit)
        from tensorfold.cuda.markers import MIN_GAP

        end, cached = entry_end(prompt), len(hit[0]) if hit else 0
        markers = getattr(self, "points", None)
        if points is None:
            points = markers(prompt) if markers is not None else []
        stops = [p for p in points if cached + MIN_GAP <= p < end]
        def keep(p, snap, tail):
            self._remember(list(prompt[:p]), {"state": snap, "tail": tail})
        first = prefill(self.e, prompt, sampling, resume=hit[1] if hit else None, constraint=constraint,
                        probabilities=probabilities, keep_at=end, stops=stops, keep=keep)
        # the state one token before the prompt's end, so the same prompt or a next turn resumes from it
        self._remember(list(prompt[:end]), self.e.kept)
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": len(hit[0]) if hit else 0,
                                 "drafts": True}
        if (on_tokens is not None and on_tokens([first])) or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        if self.depth > 0:
            depth, confidence = (self.depth, self.confidence) if mtp is None else (int(mtp[0]), float(mtp[1]))
            res = mtp_decode(self.e, first, max_tokens, sampling, depth=depth, confidence=confidence,
                             stop_eos=stop_eos, on_tokens=on_tokens, constraint=constraint, probabilities=probabilities)
            stats.update(drafted=res.drafted, accepted=res.accepted, min_rows=min(res.widths, default=0))
        else:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                                constraint=constraint, probabilities=probabilities)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def close(self) -> None:
        """Stop the concurrent scheduler's worker, so the engine's GPU memory can go (tests start several engines)."""

        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        if getattr(self, "leader", None) is not None:      # tensor parallel: the followers leave ``follow``
            self.leader.stop()

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True, constraint=None,
                 stop_eos: bool = True, background: bool = False, probabilities=None, *, vision=None,
                 mtp=None) -> dict[str, Any]:
        """``draft=False``: one token a round, no MTP drafts; ``background``: last, yielding lanes to waiting ones;
        ``mtp``: this request's (most drafts a round, confidence), the drafts at most the server's --mtp-drafts."""

        max_tokens = self._limit(prompt, max_tokens)
        if mtp is not None:
            drafts, confidence = int(mtp[0]), float(mtp[1])
            if not 0 <= drafts <= self.depth or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"mtp_drafts: 0 to the server's --mtp-drafts ({self.depth}); mtp_confidence: 0 to 1")
            if self.scheduler is not None:
                raise ValueError("mtp_drafts and mtp_confidence are per-request settings of the one-stream engine")
            mtp = (drafts, confidence)
        if probabilities is not None and not self.supports_logprobs:
            raise ValueError("logprobs are supported on one GPU only")
        if probabilities is not None and constraint is not None:
            raise ValueError("logprobs do not support structured output")
        if vision is not None and self.vision is None:
            raise ValueError("image inputs require starting this server with --vision")
        if vision is not None and background:
            raise ValueError("image requests cannot yield a background lane")
        if self.scheduler is not None:
            grammar = {} if constraint is None else {"constraint": constraint}
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                         **grammar, **({"background": True} if background else {}),
                                         probabilities=probabilities,
                                         **({"vision": vision} if vision is not None else {}))
        hit = self._resume(prompt) if draft else None
        points = None
        if self.tp > 1:                      # rank 0 decodes exactly what it hands the other ranks
            prompt, max_tokens, sampling, draft, _, _, stop_eos, points, mtp = self._share(
                prompt, max_tokens, sampling, draft, len(hit[0]) if hit else 0, constraint, stop_eos, mtp)
            self.served += 1
            emit = on_tokens
            on_tokens = lambda new: (emit(new), False)[1]       # noqa: E731  both ranks decode to the end
        if not draft:
            return self._serial(prompt, max_tokens, sampling, on_tokens, constraint, stop_eos, probabilities=probabilities)
        return self._decode(prompt, max_tokens, sampling, on_tokens, hit, constraint, stop_eos,
                            probabilities=probabilities, points=points, mtp=mtp)

    def follow(self) -> None:
        """Rank 1: decode every request rank 0 serves, until rank 0 stops."""

        if getattr(self, "multi", None) is not None:       # --parallel: replay rank 0's calls (``multi_tp``)
            from tensorfold.engine import grammar

            from .multi_tp import follow

            follow(self.multi, self.ring,
                   lambda packed: grammar.compiler(self, self.model_dir, self.eos).follow(packed))
            return
        while True:
            request = self._receive()
            if request is None:
                return
            prompt, max_tokens, sampling, draft, cached, packed, stop_eos, points, mtp = request
            constraint = None
            if packed:                                      # the request's grammar, compiled here as on rank 0
                from tensorfold.engine import grammar

                constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
            self.served += 1
            hit = None
            if draft and cached:
                hit = next(((ids, snap) for ids, snap in self.cache if len(ids) == cached and prompt[:cached] == ids),
                           None)
                if hit is None:
                    raise RuntimeError(f"rank {self.rank} has no kept state for the {cached} tokens rank 0 resumes from")
            try:
                if draft:
                    self._decode(prompt, max_tokens, sampling, None, hit, constraint, stop_eos, points=points, mtp=mtp)
                else:
                    self._serial(prompt, max_tokens, sampling, None, constraint, stop_eos)
            except ValueError as exc:                       # rank 0 raised at the same point on the same input
                print(f"[tensorfold] request {self.served} failed on both ranks: {exc}", flush=True)
