# Dynamic Uncertainty-Aware Attribution for Hallucination Detection
### CS F429 Natural Language Processing — Track A

---

## Overview

This project implements a **training-free, black-box hallucination detection system** for Retrieval-Augmented Generation (RAG) pipelines. It quantifies hallucination risk by comparing token-level probability distributions produced by a language model *with* and *without* retrieved context, deriving four uncertainty signals that are fused into a single composite score.

The system is evaluated across four experiments on two benchmarks: **RAGTruth** (RAG-specific hallucination corpus) and **HaluEval** (cross-domain QA hallucination benchmark).

---

## Repository Structure

```
.
├── notebook.ipynb          # Main Kaggle notebook (all 11 cells)
├── README.md               # This file
└── /kaggle/working/
    ├── ckpt_faithful.pkl   # Inference checkpoint — faithful samples
    ├── ckpt_hallucinated.pkl # Inference checkpoint — hallucinated samples
    └── exp3_temporal.png   # Output plot: Experiment 3 temporal precedence
```

---

## Requirements

```bash
pip install transformers accelerate scipy scikit-learn matplotlib tqdm
```

| Package | Purpose |
|---|---|
| `transformers` | Model and tokenizer loading (Mistral-7B) |
| `accelerate` | Multi-device model dispatch |
| `scipy` | Spearman correlation, Mann-Whitney U test |
| `scikit-learn` | AUROC, F1 computation |
| `matplotlib` | Temporal precedence plot (Exp 3) |
| `tqdm` | Progress bars during inference |

**Hardware:** A GPU is strongly recommended. The model is loaded in `float32` with `device_map='auto'`. On a single T4/P100 (Kaggle), expect ~2–4 GB GPU memory for Mistral-7B with truncated inputs.

---

## Dataset Setup

Place the following files under `/kaggle/input/datasets/nehuhhhh/nlp-dataset/`:

| File | Description |
|---|---|
| `qa_data.txt` | HaluEval QA split — each line is a JSON object with `question`, `knowledge`, `right_answer`, `hallucinated_answer` |
| `response.jsonl` | RAGTruth responses — each line is a JSON object with `source_id`, `response`, `labels`, `split`, `model` |
| `source_info.jsonl` | RAGTruth source contexts — each line maps `source_id` → `source_info` (context) + `prompt` |

If `source_info.jsonl` is absent, the loader falls back to building it inline from `response.jsonl` fields.

---

## Model

**`mistralai/Mistral-7B-v0.1`** is used for all inference. A Hugging Face token (`HF_TOKEN`) is read from Kaggle Secrets if available; otherwise public access is attempted.

To set your token on Kaggle: **Add-ons → Secrets → Add Secret → Name: `HF_TOKEN`**.

---

## How It Works

### Core Idea

For each response, the model runs two forward passes:
- **With context:** `"Context: {retrieved_docs}\n{prompt}\n{response}"`
- **Without context:** `"{prompt}\n{response}"`

The difference between the resulting probability distributions reveals how much the model *relies* on the retrieved context at each token. Low reliance (small distributional shift) at a token that should be grounded → hallucination signal.

### The Four Uncertainty Signals

| Signal | Formula | Interpretation |
|---|---|---|
| **Information Gain** (`delta_H`) | H(without) − H(with) | Entropy drop when context is added; low/negative = model ignores context |
| **KL Divergence** (`kl_div`) | KL(P_with ‖ P_without) | Distribution shift; low = predictions unchanged by context |
| **Confidence Drop** (`conf_drop`) | max(P_with) − max(P_without) | Drop in peak probability; negative = context reduced confidence |
| **Semantic Entropy** (`sem_ent`) | H over top-5 token probs | Residual uncertainty even after context; high = ambiguous prediction |

### Composite Score

Signals are z-normalised and summed sequentially (ablation-style):

```
composite = znorm(IG) + znorm(KL) + znorm(conf_drop) + znorm(sem_ent)
```

A higher composite score indicates higher hallucination risk.

---

## Experiments

### E1 & E2 — Composite AUROC on RAGTruth
Ablation study adding one signal at a time. Reports AUROC, best-threshold F1, Spearman ρ, and ECE against two baselines: entropy-only and SelfCheckGPT (KL proxy).

### E3 — Temporal Precedence
Analyses whether uncertainty signals peak *before* the hallucinated token (at `t-1`, `t-2`, `t-3`) rather than at onset `t`. Uses Mann-Whitney U tests for statistical significance. Output: `exp3_temporal.png`.

### E4 — Cross-domain Transfer (HaluEval)
Runs the same composite scorer zero-shot on HaluEval QA pairs (faithful vs. hallucinated answers). Reports per-metric AUROC on both datasets and the degradation gap.

### E6 — Generator Model Breakdown
Stratifies RAGTruth by the generator model that produced each response and computes per-model composite AUROC. Reveals which generators exhibit more detectable uncertainty before hallucinations.

### E8 — SOTA Gap Analysis
Benchmarks the composite against published supervised (LUMINA) and unsupervised (ReDeEP, Semantic Entropy, SelfCheckGPT) systems. Reports percentage of the SelfCheckGPT→LUMINA gap closed.

---

## Checkpointing

Inference saves progress every 25 items to avoid re-running on interruption:

```
/kaggle/working/ckpt_faithful.pkl
/kaggle/working/ckpt_hallucinated.pkl
```

On re-run, completed items are loaded automatically and only remaining items are processed.

---

## Live Demo (Cell 10)

Cell 10 runs a single passage through the pipeline and prints token-level flags:

```
Context: Marie Curie was a physicist and chemist...
Passage: Marie Curie won the Nobel Prize in Literature in 1911

Token         delta_H     KL_div  conf_drop  sem_ent  Flag
"Literature"   -0.42       0.61     -0.28     1.03    RISK
"1911"         -0.11       0.08     -0.02     0.44
```

A token is flagged `RISK` if `delta_H < 0` and `kl_div > 0.3`. The final verdict applies lexical overlap as a fast-path check before the composite score threshold.

---

## Key Hyperparameters

| Parameter | Value | Location |
|---|---|---|
| `MAX_LEN` | 600 tokens | Cell 4 — truncation for two-pass inference |
| `CTX_LEN` | 1500 chars | Cell 5 — context truncation |
| `RESP_LEN` | 500 chars | Cell 5 — response truncation |
| `MAX_EACH` | 800 | Cell 5 — max samples per class |
| `SAVE_EVERY` | 25 | Cell 5 — checkpoint frequency |
| `n_bins` | 10 | Cell 4 — ECE bin count |
| Bootstrap `n` | 1000 | Cell 4 — CI resamples |

---

## References

This implementation draws on methods from the following works (see `references.bib`):

- **RAGTruth** — Xu et al., ACL 2024
- **HaluEval** — Li et al., EMNLP 2023
- **SelfCheckGPT** — Manakul et al., EMNLP 2023
- **Semantic Uncertainty** — Kuhn et al., ICLR 2023
- **FActScore** — Min et al., EMNLP 2023
- **ReDeEP** — Zhang et al., ICLR 2025
- **RAG** — Lewis et al., NeurIPS 2020
