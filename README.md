# Piggybank — 0DTE SPX Options Strategy Discovery Pipeline

A research codebase for discovering systematic 0DTE SPX options strategies
via a three-layer AI pipeline: a self-supervised transformer for market
representation learning, strongly-typed vectorial genetic programming for
strategy evolution, and an LLM-agent workflow for QuantConnect deployment.

This is the implementation codebase for a DEng dissertation (AI/ML) on
evolutionary 0DTE strategy discovery. **Branch of record:**
`feature/qc-data-transformer-training`.

## Architecture at a glance

| Layer | Purpose | Status |
|---|---|---|
| **L1 — Representation** | iTransformer SSL encoder on minute-level 0DTE microstructure (IV surface, VIX term structure, order-flow). | Canonical: SSL-031 CNN (val_loss=0.1044, R²=0.469, pct_learned=57.6%). 3-arm ablation complete (linear/patch/CNN). |
| **L2 — Evolution** | Strongly-typed vectorial GP (STVGP) with NSGA-III. 8 defined-risk multi-leg templates (iron condors, butterflies, verticals). Typed per-group embedding terminals () consumed as opaque vectors. | GP v4 walk-forward running (Condition A, 4 credit templates). Vectorized evaluator with Black-Scholes pricing, parametric skew, realistic costs. |
| **L3 — Deployment** | Three-agent LLM pipeline (Spec / Coder / Exec) running against QuantConnect's MCP server to compile and backtest evolved strategies. | Working end-to-end (see `qc_mcp/`). |

## Research questions and hypotheses (RQ1/RQ2/RQ3 + H1/H2/H3, reframed )

**Research questions:**
- **RQ1 (REPRESENTATION LEARNING):** How can the patterns, signal, and temporal dependencies be discovered in multi-dimensional time series data such as 0DTE options to inform trading decisions?
- **RQ2 (TRANSMISSION):** What architectural conditions enable cross-paradigm transmission of learned representations from iTransformer to Genetic Programming module for combinatorial search for 0DTE options trading strategy discovery?
- **RQ3 (COMPOSITION):** How can a multi-paradigm composition (transformer learning + combinatorial search + multi-agent LLM validation) discover exploitable signal in high-dimensional time-series environments such as 0DTE options markets?

**Hypotheses:**
- **H1:** Patch-based tokenization with temporal attention produces more transferable representations than linear or convolutional tokenization in iTransformer architecture.
- **H2:** Frozen encoder embeddings that are pooled into typed GP terminal nodes via dimensionality-matched projection transmit more exploitable signal than equivalent-dimensionality handcrafted statistical features.
- **H3:** The three-paradigm composition (Transformer layer, GP layer, and multi-agent LLM setup) discovers economically viable strategies that no single paradigm or pairwise combination produces from equivalent input data.

Full methodology trail in `(internal doc)` (D-series decisions
////////////////). For
Claude Code / AI agent collaborators, start at `(internal doc)`.

## Repository layout

```
piggybank/
├── (internal doc)                       # project guidance for AI collaborators
├── (internal doc)
├── LICENSE
├── archive/                        # scrapped docs + v1 registry ( scope reset)
│   └── layer1/                     # pre-2026-04-23 experiments + audits
├── docs/
│   ├── (internal doc)                    # documentation entry point
│   ├── (internal doc)  # living results doc (RQ1+RQ2)
│   ├── dissertation/               # chapter drafts + three-ML framing
│   └── methodology/                # evaluation specs + voided pre-reg
├── experiments/
│   ├── registry.jsonl              # canonical experiment tracking
│   ├── log_experiment.py           # registry writer
│   └── registry_v1_before_2026_04_23_scope_reset.jsonl
├── layer1/                         # SSL encoder + data pipeline
│   ├── PRAXIS_DECISIONS.md         # D-series decision log (methodology trail)
│   ├── EXPERIMENTS.md              # canonical + baseline experiments
│   ├── FEATURE_DOCUMENTATION.md    # per-variate documentation (v2 141 + v3 +18 aggs)
│   ├── DATA_PIPELINE_ARCHITECTURE.md
│   ├── data/                       # corpus collection, backfill, audit scripts
│   ├── training/
│   │   ├── pipeline.py             # full SSL training pipeline (self-contained)
│   │   ├── extended_probes.py      # RF/GBT/poly-kernel bracketing probes
│   │   └── deploy_training.py      # QC Research deployer (legacy cloud path)
│   ├── inference/
│   │   └── batch_forecast.py       # L1→Parquet adapter for L2 consumption
│   └── docs/
│       ├── (internal doc)
│       ├── (internal doc)
│       ├── (internal doc)
│       ├── (internal doc)
│       └── future_work/
│           └── (internal doc)   # SSL-009 → future-work proposal
├── layer2/                         # STVGP GP engine
│   ├── grammar.py, evaluator.py, templates.py, gp_engine.py, fitness.py
│   └── PRAXIS_DECISIONS.md         # L2-series decision log
├── scripts/
│   ├── train_local.py              # canonical local SSL trainer (MPS)
│   └── run_extended_probes.py      # post-training probe runner
├── qc_mcp/                         # Layer 3 — QC MCP client + strategy agents
├── raw_data/                       # local v3 corpus bundle + local MLflow store
└── tests/                          # L2 unit tests (26 + 28)
```

## Running things

- **L1 training (canonical, local MPS):** `/opt/anaconda3/bin/python -m scripts.train_local --full`
- **L1 probes:** `/opt/anaconda3/bin/python -m scripts.run_extended_probes`
- **L3 (strategy generation via MCP):** `cd qc_mcp && python main.py` — requires `.env` with `QUANTCONNECT_USER_ID`, `QUANTCONNECT_API_TOKEN`, `ANTHROPIC_API_KEY` and Docker running.
- See `(internal doc)` for full environment and deployment details.

## Legacy note

The `qc_mcp/` directory contains the original MCP client for generating
individual strategies via a three-agent LLM pipeline. It is still live
(Layer 3 of the dissertation architecture) but is no longer the primary
research path — the L1+L2 evolutionary pipeline is.

## License

See `LICENSE`.
