"""Decode speed of an OpenAI-compatible server, measured from the stream: one request at a time, or N at once.

Decode tok/s = (completion tokens - 1) / (last token time - first token time), the same span the
CUDA engine's bench times (the first sampled token excluded). Standard library only, so it runs on
a bare host.

  python3 tools/bench_openai.py http://127.0.0.1:8080 MODEL --tokens 64 --reps 5 --output out.json
  python3 tools/bench_openai.py http://127.0.0.1:8080 MODEL --tokens 256 --concurrency 4   # a --parallel server

``--concurrency N`` sends N requests at once (seeds 1234 ..), after the same N one at a time: each stream's tok/s,
everything's tok/s over the wall time, and whether each reply equals its request's reply alone (exact decoding).
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPTS = [
    {"name": "fibonacci-raw", "kind": "completion",
     "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."},
    {"name": "gpu-chat-no-think", "kind": "chat",
     "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."},
]


def stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None,
           draft: bool = True, extra: dict | None = None) -> dict:
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True, **(extra or {})}
    if not draft:
        body["draft"] = False                     # the serial reference: one token a forward
    if seed is not None:
        body["seed"] = seed
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if item["kind"] == "chat":
        url = base + "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": item["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = base + "/v1/completions"
        body["prompt"] = item["prompt"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage = stats = None
    text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("tensorfold"):
                stats = chunk["tensorfold"]
            for choice in chunk.get("choices", []):
                piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    text.append(piece)
    n = int(usage["completion_tokens"]) if usage else None
    rounds = int((stats or {}).get("rounds") or 0)
    engine_s = float((stats or {}).get("decode_s") or 0.0)
    return {"ttft_s": first - start, "decode_s": last - first, "tokens": n, "start": start, "last": last,
            "decode_tps": (n - 1) / (last - first) if n and last > first else None, "text": "".join(text),
            # the engine's own clock: a round's time (kernel speed) and tokens a round (drafting), apart
            "ms_round": engine_s / rounds * 1e3 if rounds else None,
            "tokens_round": (n - 1) / rounds if rounds and n else None}


def together(base: str, model: str, item: dict, tokens: int, temperature: float, seeds: list, draft: bool,
             extra: dict) -> tuple[list[dict], float]:
    """One request a seed, all at once: their results and the wall time from the first sent to the last token."""

    runs: list = [None] * len(seeds)

    def one(i: int) -> None:
        runs[i] = stream(base, model, item, tokens, temperature, seeds[i], draft, extra)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(seeds))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return runs, max(r["last"] for r in runs) - min(r["start"] for r in runs)


def concurrent(args, temp: float, item: dict, draft: bool, extra: dict) -> dict:
    n = args.concurrency
    seeds = [None] * n if args.seed_from_prompt else [1234 + i for i in range(n)]
    alone = [stream(args.base, args.model, item, args.tokens, temp, seed, draft, extra) for seed in seeds]
    together(args.base, args.model, item, args.tokens, temp, seeds, draft, extra)               # warm-up
    reps = [together(args.base, args.model, item, args.tokens, temp, seeds, draft, extra) for _ in range(args.reps)]
    tps = [r["decode_tps"] for runs, _ in reps for r in runs if r["decode_tps"]]
    total = [sum(r["tokens"] or 0 for r in runs) / wall for runs, wall in reps]
    same = all(r["text"] == a["text"] for runs, _ in reps for r, a in zip(runs, alone))
    return {"label": args.label, "prompt": item["name"], "temperature": temp, "tokens": args.tokens,
            "concurrency": n, "per_stream_tps_median": statistics.median(tps), "per_stream_tps_all":
            [round(x, 2) for x in tps], "alone_tps_median": statistics.median(a["decode_tps"] for a in alone),
            "aggregate_tps_median": statistics.median(total),
            "ms_round_median": statistics.median([r["ms_round"] for runs, _ in reps for r in runs if r["ms_round"]]
                                                 or [0.0]),
            "tokens_round_median": statistics.median([r["tokens_round"] for runs, _ in reps for r in runs
                                                      if r["tokens_round"]] or [0.0]),
            "same_as_alone": same}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--temperatures", default="1.0,0")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    p.add_argument("--no-drafts", action="store_true",
                   help="request the serial reference (\"draft\": false): one token a forward, so 1 / tok/s is a "
                        "forward's latency")
    p.add_argument("--extra", default="{}",
                   help='JSON fields merged into every request, e.g. \'{"mtp_drafts": 8, "mtp_confidence": 0.3}\'')
    p.add_argument("--seed-from-prompt", action="store_true",
                   help="send no seed: cuda_server then seeds from the prompt, as the engine benches do, "
                        "so the reply equals the bench's and the timings compare directly")
    p.add_argument("--concurrency", type=int, default=1,
                   help="N requests at once (a server started with --parallel N or more), against the same alone")
    args = p.parse_args()
    results = []
    for temp in [float(t) for t in args.temperatures.split(",")]:
        for item in PROMPTS:
            seeds = [None] * args.reps if args.seed_from_prompt else [1234 + i for i in range(args.reps)]
            draft, extra = not args.no_drafts, json.loads(args.extra)
            if args.concurrency > 1:
                row = concurrent(args, temp, item, draft, extra)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}),
                      flush=True)
                results.append(row)
                continue
            stream(args.base, args.model, item, args.tokens, temp, seeds[0], draft, extra)          # warm-up
            runs = [stream(args.base, args.model, item, args.tokens, temp, seed, draft, extra) for seed in seeds]
            tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
            row = {"label": args.label, "prompt": item["name"], "temperature": temp, "tokens": args.tokens,
                   "decode_tps_median": statistics.median(tps), "decode_tps_all": [round(x, 2) for x in tps],
                   "ttft_s_median": statistics.median(r["ttft_s"] for r in runs),
                   "ms_round_median": statistics.median([r["ms_round"] for r in runs if r["ms_round"]] or [0.0]),
                   "tokens_round_median": statistics.median([r["tokens_round"] for r in runs if r["tokens_round"]]
                                                            or [0.0]),
                   "sample": runs[0]["text"][:160]}
            print(json.dumps({k: (round(row[k], 3) if isinstance(row[k], float) else row[k])
                              for k in ("label", "prompt", "temperature", "decode_tps_median", "decode_tps_all",
                                        "ttft_s_median", "ms_round_median", "tokens_round_median")}), flush=True)
            results.append(row)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
