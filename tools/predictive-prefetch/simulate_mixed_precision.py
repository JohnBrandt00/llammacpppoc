#!/usr/bin/env python3
"""Popularity-tiered precision -- Stage 0 (simulation on routing traces).

Synthesis failed because experts are incompressible in the *weights*. But the
*traffic* is highly skewed (a few experts do most of the work), so we can spend
bits where they matter: keep the hot experts at high precision and store the
cold ones at low precision. This measures the trade for a per-layer split that
keeps the top fraction f of experts (by activation count) at `hi` bpw and the
rest at `lo` bpw:

  footprint : model size vs all-`hi` (smaller = more fits in RAM/VRAM cache)
  exposure  : fraction of token-expert activations that hit a LOW-precision
              expert -- the proxy for quality risk (lower = safer)

If footprint drops meaningfully while exposure stays small, the idea is worth
implementing (requantize cold experts, then confirm with real perplexity).
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# effective bits-per-weight of common llama.cpp quants (incl. K-quant overhead)
BPW = {"Q8_0": 8.5, "Q6_K": 6.56, "Q5_K": 5.5, "Q4_K": 4.5,
       "Q3_K": 3.9, "Q2_K": 2.6, "IQ2": 2.2, "IQ1": 1.6}


def load_counts(paths):
    """(layer,expert) -> count, aggregated across traces."""
    counts = defaultdict(Counter)
    for path in paths:
        for line in open(path, encoding="utf-8", errors="replace"):
            p = line.split()
            if len(p) < 2:
                continue
            try:
                layer = int(p[0]); experts = [int(x) for x in p[1:]]
            except ValueError:
                continue
            counts[layer].update(experts)
    return counts


def evaluate(counts, f_high, hi, lo):
    n_low_experts = n_experts = 0
    act_low = act_total = 0
    bits = 0.0
    for layer, c in counts.items():
        experts = sorted(c, key=lambda e: c[e], reverse=True)
        # include experts never seen? assume the full pool exists; pad with the
        # observed max id+1 so cold (unused) experts count as low-precision
        n = max(c) + 1
        k_high = max(1, round(f_high * n))
        hot = set(experts[:k_high])
        for e in range(n):
            n_experts += 1
            if e in hot:
                bits += BPW[hi]
            else:
                bits += BPW[lo]
                n_low_experts += 1
                act_low += c.get(e, 0)
            act_total_add = c.get(e, 0)
        act_total += sum(c.values())
    footprint = bits / (n_experts * BPW[hi])
    exposure = act_low / act_total if act_total else 0.0
    return footprint, exposure, n_low_experts / n_experts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--hi", default="Q4_K")
    ap.add_argument("--lo", default="Q2_K")
    args = ap.parse_args()

    counts = load_counts(args.traces)
    print(f"hot={args.hi} ({BPW[args.hi]} bpw)  cold={args.lo} ({BPW[args.lo]} bpw)  "
          f"layers={len(counts)}")
    print(f"  {'keep hot':>9} {'cold experts':>13} {'footprint':>10} {'low-prec exposure':>18}")
    for f in (0.10, 0.25, 0.40, 0.50, 0.75):
        fp, exp, frac_low = evaluate(counts, f, args.hi, args.lo)
        print(f"  {f*100:>7.0f}% {frac_low*100:>11.0f}% {fp*100:>9.1f}% {exp*100:>17.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
