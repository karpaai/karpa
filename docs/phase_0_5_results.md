# Phase 0.5 — First H100 Run: Noise Floor, Calibration, and Karpathian-1

**Date:** May 26, 2026
**Hardware:** NVIDIA H100 PCIe 80GB (Shadeform / ShadeCloud)
**Data:** 1B tokens from FineWeb-Edu (sample-10BT), GPT-2 BPE tokenizer
**Code:** [`KarpathianBase/karpathian`](https://github.com/KarpathianBase/karpathian) @ `main`

---

## What was tested

Phase 0.5 is the first run of the Karpathian protocol on real hardware with real data. Three things were measured:

1. **H100 calibration benchmark** — pins the reference timings for the hardware-independent compute unit (whitepaper §5.5)
2. **Noise floor** — 10 baseline runs with different seeds to empirically set the "decisively beats the king" margin (§5.7)
3. **Karpathian-1** — the first demonstration model trained on the canonical recipe (§6.8)

All runs used the unverified tier (α=0.5) since the H100 instance was not CC-capable. Verified-tier (α=1.0) testing with real TDX+nvtrust attestation is Phase 0.5c.

---

## Results

### H100 Calibration Benchmark

The calibration benchmark is a deterministic workload (matmul + attention + collective) that fingerprints the hardware. These timings become the reference for normalizing compute claims across different GPUs.

| Workload | H100 PCIe timing |
|---|---|
| Matmul (2048×2048, 20 reps) | **0.512 ms** |
| Attention (B=4, H=16, T=1024, 20 reps) | **0.464 ms** |
| Collective (sum 4M floats, 20 reps) | **0.012 ms** |
| **Total** | **0.988 ms** |

For reference: the same benchmark takes ~47ms on a CPU workstation (45× slower).

### Noise Floor Calibration

10 runs of the unchanged baseline (125M params, 500 steps each) with different random seeds. This measures how much val_bpb varies from seed alone — the noise floor that a real improvement must exceed.

| Metric | Value |
|---|---|
| Runs | 10 |
| Config | `h100_proxy.json` (125M params, 500 steps) |
| val_bpb mean | **1.9430** |
| val_bpb std (σ) | **0.0064** |
| val_bpb min | 1.9346 |
| val_bpb max | 1.9557 |
| val_bpb range | 0.0211 |
| **Suggested margin (2σ)** | **0.0127 val_bpb** |
| Wall-clock per run | 17.6 min |
| Total wall-clock | 176 min |

**What this means:** A miner's patch needs to improve val_bpb by at least **0.013** to be considered a genuine improvement rather than seed noise. This is a tight noise floor — the protocol can detect small, real gains.

Per-seed breakdown:

| Seed | val_bpb |
|---|---|
| 5000 | 1.9393 |
| 5001 | 1.9513 |
| 5002 | 1.9390 |
| 5003 | 1.9346 |
| 5004 | 1.9441 |
| 5005 | 1.9376 |
| 5006 | 1.9465 |
| 5007 | 1.9447 |
| 5008 | 1.9377 |
| 5009 | 1.9557 |

### Karpathian-1 Training

The first model trained on the canonical recipe — the proof artifact from whitepaper §6.8.

| Metric | Value |
|---|---|
| Model | Karpathian-base (Llama-style: RMSNorm + RoPE + SwiGLU + MHA) |
| Parameters | **253,872,128 (~254M)** |
| Config | `h100_default.json` (dim=1024, 16 layers, 16 heads) |
| Tokens trained | **262,144,000 (~262M)** |
| Data | FineWeb-Edu (sample-10BT subset), 1B tokens tokenized |
| Seed | 1337 |
| **Final loss** | **3.8173** |
| Wall-clock | 258.8 min (~4.3 hours) |
| Throughput | 16,882 tok/s |
| Device | NVIDIA H100 PCIe, CUDA, fp32 |

Loss curve:

```
step  |  loss
------+--------
    0 | 11.005
  100 | 7.076
  200 | 6.084
  300 | 5.474
  500 | 4.914
  700 | 4.497
 1000 | 4.223
 1300 | 3.987
 1500 | 3.911
 1800 | 3.806
 2000 | 3.817
```

**Note on throughput:** This run used fp32 (no mixed precision). Adding bf16 autocast is expected to roughly double throughput to ~34K tok/s, cutting wall-clock to ~2 hours. This is a Phase 0.5b optimization — the architecture proof was the priority for this run.

---

## What was verified

1. **The full data pipeline works at scale.** 1B tokens from FineWeb-Edu streamed, tokenized, and sharded in 7.4 minutes at 2.27M tok/s. Content-addressed manifest verified.

2. **The noise floor is measurably tight.** σ = 0.006 val_bpb means the network can detect genuine improvements as small as 0.013 val_bpb. This is the number the whitepaper §5.7 promised to measure empirically — now measured.

3. **The canonical training loop runs end-to-end on GPU.** Deterministic seeding, AdamW + cosine LR, gradient clipping, structured training log — all working on H100.

4. **The calibration benchmark produces stable H100 reference timings.** These anchor the hardware-independent compute unit for scoring (§5.5).

---

## How to reproduce

```bash
git clone https://github.com/KarpathianBase/karpathian.git
cd karpathian
bash scripts/run_h100.sh
```

Requires an H100 GPU. The script handles everything: venv, dependencies, data prep, calibration, noise floor, and Karpathian-1 training. Expected wall-clock: ~6-7 hours total (fp32). Results land in `runs/`.

---

## What's next

- **Phase 0.5b:** Add bf16 mixed precision for 2× throughput
- **Phase 0.5c:** Rent a CC-capable H100 (Azure NCC / GCP A3-Confidential) for real TDX+nvtrust attestation — replace mock attestation with hardware-rooted signatures
- **Phase 0.5d:** Bittensor testnet integration — replace local JSON chain with real on-chain commitments
- **Phase 1:** Register subnet, open to external miners, first bounty pilot

---

## New infrastructure shipped with this milestone

| Component | What it does |
|---|---|
| `--wandb` flag in `recipe/train.py` | Live training monitoring at wandb.ai — loss, lr, grad norms, throughput, every step |
| `dashboard/app.py` | Streamlit dashboard: king status, submission feed, noise floor, loss curves |
| `miner/hub.py` | HuggingFace Hub integration: upload/download proof bundles to `KarpathianBase/proof-bundles` |
| Two-tier scoring (`validator/scoring.py`) | Verified (α=1.0) vs unverified (α=0.5) credibility model per whitepaper v1.1 §5.4 |
| Stage 5 audit (`validator/audit.py`) | Score-and-trajectory equivalence checking (not hash equality) per §5.2 |

---

🔗 **Repo:** [github.com/KarpathianBase/karpathian](https://github.com/KarpathianBase/karpathian)
📄 **Whitepaper:** v1.1 (available in repo)
