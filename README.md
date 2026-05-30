<img src="docs/assets/karpa-header.jpg" alt="Karpa — decentralized · autonomous · AI research" />

A Bittensor subnet for decentralized, autonomous AI research. An open,
continuously improving training recipe — and the public knowledge corpus
behind it — built by an autonomous research network on Bittensor.

🌐 [karpa.ai](https://karpa.ai) · 📄 [Whitepaper v1.2](docs/Karpa-Whitepaper-v1.2.pdf) · 🏷️ [Releases](https://github.com/karpaai/karpa/releases) · 📊 [Wandb](https://wandb.ai/karpaai-hub/karpa) · 💬 [Discussions](https://github.com/orgs/karpaai/discussions)

## What Karpa produces

1. **A canonical training recipe** — a Git repo containing the best-known open recipe for each track (model class × objective). Anyone can clone it and train a model with state-of-the-art settings.
2. **A public experiment-record corpus** — every submission the network has ever processed, including verified negative results. Searchable, citable, openly licensed.
3. **A demonstration model lineage** — Karpa-1, -2, … — open-weights reference models proving the recipe works and the improvement compounds.

The subnet and its token fund the production of these artifacts. They are not the deliverable.

## Current status

| Phase | Status | Key results |
|---|---|---|
| **0 — MVP** | ✅ Complete | End-to-end protocol on CPU: model, training, eval, proof-test, validator, scoring, king-change cycle |
| **0.5 — H100** | ✅ Complete ([`v0.5.0`](https://github.com/karpaai/karpa/releases/tag/v0.5.0) · [results](https://github.com/orgs/karpaai/discussions/4)) | Real data (1B tokens FineWeb-Edu), noise floor measured (2σ = 0.013 val_bpb), Karpa-1 trained (254M params, loss 3.82) |
| **0.5b — Optimization** | ✅ Complete ([`v0.5.1`](https://github.com/karpaai/karpa/releases/tag/v0.5.1)) | bf16: 3.8× throughput (63K tok/s), same loss. wandb live monitoring, Streamlit dashboard, wandb metrics export in proof bundles |
| **0.5c — Attestation** | ✅ Code-complete | Real TDX + nvtrust attestation module: auto-detects CC hardware, falls back to mock. Untested on real CC (needs Azure NCC / GCP A3-Confidential) |
| **0.5d — Testnet** | ✅ Complete ([`v0.6.0`](https://github.com/karpaai/karpa/releases/tag/v0.6.0)) | Bittensor testnet (netuid 16): two miners competed, validator set weights on-chain, king changed. Chain abstraction layer with rate-limit handling. |
| **1.0 — Launch** | Planned | Register subnet, open to external miners, first bounty pilot |
| **1.1 — SDK** | Planned | `pip install karpa-subnet` on PyPI, CI/CD, changelog, semver |
| **1.2 — Docs** | Planned | Documentation site, miner/validator quickstart guides, corpus query tutorials |

## Architecture (three layers)

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Miner's private search                   │
│  Any agent, any LLM, any GPU, any training code.    │
│  The protocol doesn't see this.                     │
└──────────────────────┬──────────────────────────────┘
                       │ candidate patch
┌──────────────────────▼──────────────────────────────┐
│  Layer 2 — Canonical proof test                     │
│  Official Karpa Docker on miner's GPU.              │
│  Applies patch to canonical recipe, trains under    │
│  fixed (seed, data, config), produces checkpoint +  │
│  training log + calibration + attestation chain.    │
└──────────────────────┬──────────────────────────────┘
                       │ proof bundle
┌──────────────────────▼──────────────────────────────┐
│  Layer 3 — Submission + judgment                    │
│  PR to canonical recipe repo + proof bundle on HF.  │
│  Validator: diff scan → attestation verify →        │
│  log plausibility → hidden eval → score.            │
│  If it decisively beats the king → merge.           │
└─────────────────────────────────────────────────────┘
```

## Repo layout

Karpa lives across **two repos**:

| Repo | What | Patchable by miners? |
|---|---|---|
| **[karpaai/recipe](https://github.com/karpaai/recipe)** | `model/`, `recipe/`, `configs/`, `data/` — the canonical training recipe miners patch and the merged history of accepted improvements | **Yes** |
| **karpaai/karpa** (this repo) | Protocol: validator, proof-test runner, attestation, scoring, submission tooling | **No** (restricted) |

### This repo (protocol)

| Path | What |
|---|---|
| `eval/` | Hidden-eval harness, val_bpb, benchmark mix |
| `calibration/` | Deterministic compute benchmark (matmul + attention) |
| `proof/` | Proof-test runner (future: Docker container) |
| `miner/` | Submission bundle assembly, HuggingFace upload, hotkey signing |
| `validator/` | Four cheap ops + scoring + Stage 5 audit |
| `chain_layer/` | Bittensor + local-JSON chain abstractions |
| `dashboard/` | Karpa Live — Streamlit monitoring dashboard |
| `scripts/` | `miner_run.py`, `run_h100.sh`, `noise_floor.py`, `smoke_test.py`, `gpu.py` |
| `karpa_bootstrap.py` | Adds the sibling recipe repo to `sys.path` for protocol code |

The protocol code locates the recipe via `$KARPA_RECIPE_DIR` (defaults to `../recipe`). Clone both repos side-by-side and everything just works.

## Quick start

### CPU smoke test (no GPU needed)

```bash
# Clone both repos side-by-side
git clone https://github.com/karpaai/karpa.git
git clone https://github.com/karpaai/recipe.git
cd karpa
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy tiktoken cryptography

# Generate synthetic data into the recipe repo
(cd ../recipe && python -m data.prepare --source synthetic --out data/shards \
    --shard-tokens 50000 --total-tokens 200000 --eval-tokens 10000)

# Run end-to-end: two miners submit, validator scores, king changes
python scripts/smoke_test.py
```

### H100 full run (real data)

```bash
git clone https://github.com/karpaai/karpa.git
git clone https://github.com/karpaai/recipe.git
cd karpa
bash scripts/run_h100.sh
```

This bootstraps everything on a fresh H100: FineWeb-Edu data prep (1B tokens),
calibration benchmark, noise floor (10 seeds), and Karpa-1 training
(254M params, ~262M tokens). Wall-clock: ~6-7 hours (fp32).

### Live monitoring

```bash
# wandb (real-time loss curves during training)
python -m recipe.train --config configs/h100_default.json --out-dir runs/my_run --wandb

# Streamlit dashboard (network status, king history, submissions)
pip install 'karpa-subnet[dashboard]'
streamlit run dashboard/app.py
```

## Two-tier credibility model

| Tier | Requirements | Scoring |
|---|---|---|
| **Verified** (α=1.0) | Official Docker in CC-CVM (H100/H200/B200 + TDX/SEV-SNP) | Compute claim at face value |
| **Unverified** (α=0.5) | Official Docker on any GPU | Effective cost = 2× claimed (0.5× credibility discount) |

The 0.5× discount is calibrated against the H100/4090 price ratio (~5-10×) so
lying about hardware is unprofitable. As Confidential Computing commoditizes,
the unverified tier is expected to be deprecated.

## Phase 0.5 measured results

| Metric | Value |
|---|---|
| H100 calibration (matmul) | 0.512 ms |
| Noise floor (10 seeds, 125M model) | σ = 0.006 val_bpb, margin (2σ) = 0.013 |
| Karpa-1 fp32 (254M params, 262M tokens) | Final loss = 3.82, 16.9K tok/s, 259 min |
| Karpa-1 bf16 (same model, same data) | Final loss = 3.82, **63.4K tok/s, 69 min (3.8× faster)** |

Full results: [Phase 0.5 Discussion](https://github.com/orgs/karpaai/discussions/4) ·
Release: [`v0.5.0`](https://github.com/karpaai/karpa/releases/tag/v0.5.0)

## License

Apache-2.0
