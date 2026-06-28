#!/usr/bin/env python3
"""Expert Weight Synthesis -- Stage 0 go/no-go (numpy only, no model run).

Measures whether a layer's MoE expert weight matrices live on a low-dimensional
manifold: if a handful of principal components capture most of the variance and
reconstruct each expert with high fidelity, then storing a small latent code per
expert + one shared basis (decoder) could replace the full weights.

Decision gate (from the plan):
  - <64 dims for 85% variance AND >60% of experts at fidelity >0.92  -> proceed
  - otherwise                                                        -> abandon

Reads a Q4_K_M GGUF, dequantizes one layer's gate/up/down expert tensors to
float32, and runs SVD across the 128 experts.

  python validate_synthesis.py models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf --layer 24
"""
import argparse
import numpy as np
import gguf
import gguf.quants as quants


def load_layer_experts(path, layer):
    reader = gguf.GGUFReader(path)
    want = {f"blk.{layer}.ffn_{p}_exps.weight": p for p in ("gate", "up", "down")}
    parts = {}
    for t in reader.tensors:
        if t.name in want:
            deq = quants.dequantize(t.data, t.tensor_type)   # [n_expert, a, b]
            parts[want[t.name]] = deq.reshape(deq.shape[0], -1).astype(np.float32)
    if set(parts) != {"gate", "up", "down"}:
        raise SystemExit(f"layer {layer}: found only {sorted(parts)}")
    M = np.concatenate([parts["gate"], parts["up"], parts["down"]], axis=1)
    return M  # [n_expert, expert_dim]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("-K", type=int, default=32, help="latent dim for fidelity test")
    args = ap.parse_args()

    M = load_layer_experts(args.model, args.layer)
    n_expert, dim = M.shape
    print(f"layer {args.layer}: {n_expert} experts x {dim} weights "
          f"({M.nbytes/1e9:.2f} GB f32)")

    mean = M.mean(axis=0)
    Mc = M - mean
    # economy SVD: U [n,n], s [n], Vt [n, dim]
    U, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    var = np.cumsum(s**2) / np.sum(s**2)

    print("dims for variance thresholds:")
    for thr in (0.70, 0.80, 0.85, 0.90, 0.95, 0.99):
        d = int(np.searchsorted(var, thr) + 1)
        print(f"  {thr*100:4.0f}%: {d:3d} of {n_expert}")

    K = min(args.K, n_expert)
    basis = Vt[:K]                       # [K, dim]
    codes = Mc @ basis.T                 # [n_expert, K]
    recon = codes @ basis + mean         # [n_expert, dim]
    num = (M * recon).sum(axis=1)
    den = np.linalg.norm(M, axis=1) * np.linalg.norm(recon, axis=1) + 1e-8
    fid = num / den

    print(f"\nreconstruction fidelity at K={K} (latent {K} floats/expert):")
    print(f"  mean {fid.mean():.4f}  min {fid.min():.4f}  median {np.median(fid):.4f}")
    for thr in (0.90, 0.92, 0.95, 0.97):
        print(f"  >{thr}: {(fid > thr).mean()*100:5.1f}% of experts")

    dims85 = int(np.searchsorted(var, 0.85) + 1)
    cover = (fid > 0.92).mean()
    verdict = "PROCEED" if (dims85 < 64 and cover > 0.60) else "ABANDON synthesis"
    print(f"\ngate: 85%-variance dims={dims85} (<64?), fidelity>0.92 coverage={cover*100:.1f}% (>60%?)")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
