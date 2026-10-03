"""Long prompts against an OpenAI-compatible server: prompt time, decode speed and exact decoding at a context size.

  python3 tools/long_context.py http://127.0.0.1:8080 MODEL --contexts 65536,131072,200000,258000 --serial
  python3 tools/long_context.py http://127.0.0.1:8080 MODEL --contexts 131072 --concurrency 2   # a --parallel server

The prompts are this repository's own sources and docs, file after file (a second stream starts further in), cut
to about the tokens asked: a short request first measures the characters a token takes, and each line prints the
server's own count. Replies are greedy (temperature 0) and ignore end tokens. ``--serial`` sends the same prompt
with "draft": false and checks both replies are the same tokens (exact decoding at that context); ``--concurrency
N`` sends N different prompts at once and checks each reply equals its prompt's reply alone. One JSON line a test.
Standard library only.
"""

import argparse
import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bench_openai import stream

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUFFIXES = (".py", ".md", ".cu", ".cpp")
QUESTION = "\n\nIn three sentences, what does the code above do?\n"


def corpus(root: pathlib.Path) -> str:
    """Every source and doc file under ``root``'s src, docs, tests and tools, in path order, each under its name."""

    parts = []
    for folder in ("src", "docs", "tests", "tools"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in SUFFIXES:
                parts.append(f"\n\n# file: {path.relative_to(root)}\n" + path.read_text(errors="ignore"))
    return "".join(parts)


def prompt(text: str, chars: int, start: int = 0) -> str:
    """``chars`` characters of ``text`` from ``start`` on (around again past its end), then the question."""

    body = (text[start:] + text[:start]) * (chars // max(1, len(text)) + 1)
    return body[:chars] + QUESTION


def run(args, text: str, draft: bool = True) -> dict:
    item = {"name": "long", "kind": "completion", "prompt": text}
    try:
        r = stream(args.base, args.model, item, args.tokens, 0.0, None, draft)
    except Exception as exc:                     # noqa: BLE001  (a refusal or a failure is a result too)
        body = getattr(exc, "read", lambda: b"")()
        return {"error": f"{type(exc).__name__}: {exc} {body.decode(errors='ignore')[:300]}".strip()}
    keep = ("prompt_tokens", "cached", "prefill_s", "ttft_s", "decode_tps", "ms_round", "tokens_round", "tokens")
    out = {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keep}
    out["text"] = r["text"]
    return out


def at_once(args, texts: list[str]) -> list[dict]:
    """``run`` of every text at the same time, in their order."""

    out: list = [None] * len(texts)

    def one(i: int) -> None:
        out[i] = run(args, texts[i])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(texts))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--contexts", default="65536,131072,200000,258000", help="prompt tokens to aim at, comma-separated")
    p.add_argument("--tokens", type=int, default=256, help="reply tokens")
    p.add_argument("--serial", action="store_true", help='also "draft": false: the same tokens?')
    p.add_argument("--concurrency", type=int, default=1, help="different long prompts at once (a --parallel server)")
    p.add_argument("--source", default=str(ROOT), help="the tree whose src/docs/tests/tools make the prompts")
    args = p.parse_args()

    text = corpus(pathlib.Path(args.source))
    sample = run(argparse.Namespace(**{**vars(args), "tokens": 1}), text[:40000])
    if "error" in sample or not sample.get("prompt_tokens"):
        print(json.dumps({"calibration": sample}), flush=True)
        return
    per = (40000 + len(QUESTION)) / sample["prompt_tokens"]             # characters a token
    print(json.dumps({"corpus_chars": len(text), "chars_per_token": round(per, 3)}), flush=True)
    for target in [int(t) for t in args.contexts.split(",")]:
        chars = int((target - 64) * per)
        if args.concurrency <= 1:
            got = run(args, prompt(text, chars))
            line = {"target": target, **{k: v for k, v in got.items() if k != "text"}}
            if args.serial and "error" not in got:
                ref = run(args, prompt(text, chars), draft=False)
                line["serial"] = {k: v for k, v in ref.items() if k != "text"}
                line["same_as_serial"] = ref.get("text") == got["text"]
            print(json.dumps(line), flush=True)
            continue
        texts = [prompt(text, chars, start=i * len(text) // args.concurrency) for i in range(args.concurrency)]
        alone = [run(args, t) for t in texts]
        together = at_once(args, texts)
        print(json.dumps({"target": target, "concurrency": args.concurrency,
                          "alone": [{k: v for k, v in a.items() if k != "text"} for a in alone],
                          "together": [{k: v for k, v in t.items() if k != "text"} for t in together],
                          "same_as_alone": all("error" not in a and a.get("text") == t.get("text")
                                               for a, t in zip(alone, together))}), flush=True)


if __name__ == "__main__":
    main()
