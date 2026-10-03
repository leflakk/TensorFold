"""Tensor-parallel draws from gathered candidates (any machine): several streams' rows of one sampling rule in one
call, each row the token (and probability) ``choose_host`` gives it alone."""

import numpy as np

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.decode import choose_host, choose_host_rows


def _host(rng, rows: int, width: int = 256):
    values = rng.normal(0, 3.0, (rows, width)).astype(np.float32)
    values[:, 5] = values[:, 40]                                  # ties: the lower id wins, alone or together
    tokens = np.stack([rng.choice(248320, width, replace=False) for _ in range(rows)]).astype(np.int64)
    total = np.log(np.exp(values.astype(np.float64)).sum(axis=1)) + rng.uniform(0, 2, rows)
    return values, tokens, total


def test_rows_of_one_rule_draw_together_as_each_alone():
    rng = np.random.default_rng(3)
    rules = [None, Sampling(seed=0, temperature=0.0), Sampling(seed=0, top_k=20, top_p=0.95),
             Sampling(seed=0, temperature=0.7, top_k=40, top_p=0.9), Sampling(seed=0, top_k=20, top_p=0.95, min_p=0.05)]
    for trial in range(100):
        host = _host(rng, 16)
        n = int(rng.integers(1, 9))
        rows = [int(r) for r in rng.choice(16, n, replace=False)]
        positions = [int(p) for p in rng.integers(0, 50000, n)]
        samplings = []
        for _ in range(n):
            rule = rules[int(rng.integers(len(rules)))]
            samplings.append(None if rule is None else Sampling(seed=int(rng.integers(1 << 62)),
                                                                temperature=rule.temperature, top_k=rule.top_k,
                                                                top_p=rule.top_p, min_p=rule.min_p))
        alone = [choose_host(host, r, r + 1, [p], smp, with_prob=True) for r, p, smp in zip(rows, positions, samplings)]
        toks, probs = choose_host_rows(host, rows, positions, samplings, with_prob=True)
        assert toks == [a[0][0] for a in alone] and probs == [a[1][0] for a in alone], trial
        assert choose_host_rows(host, rows, positions, samplings) == toks
