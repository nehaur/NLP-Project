# Dynamic Uncertainty-Aware Attribution for Hallucination Detection
### CS F429 Natural Language Processing — Track A

---

## Requirements

- **Python:** 3.12
- **GPU:** CUDA-capable GPU recommended (tested on NVIDIA T4/P100)
- **CUDA:** 11.8 or higher

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. (Recommended) Create a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. HuggingFace token (if required)

`mistralai/Mistral-7B-v0.1` is a public model and does not require authentication. If you encounter access issues, generate a token at https://huggingface.co/settings/tokens and pass it via `--hf_token`.

---

## Running demo.py

### Basic usage — inline context

```bash
python demo.py \
  --context "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911." \
  --passage "Marie Curie won the Nobel Prize in Literature in 1911."
```

### Context from a file

```bash
python demo.py --context_file context.txt --passage "Your passage here."
```

### With a HuggingFace token

```bash
python demo.py \
  --context "Your context here." \
  --passage "Your passage here." \
  --hf_token hf_xxxxxxxxxxxxxxxxxxxx
```

---

## Expected Output

```
======================================================================
DYNAMIC UNCERTAINTY-AWARE HALLUCINATION DETECTION
======================================================================
Context : Marie Curie was a physicist and chemist...
Passage : Marie Curie won the Nobel Prize in Literature in 1911.
======================================================================

Running two-pass inference...

TOKEN-LEVEL UNCERTAINTY SIGNALS
--------------------------------------------------------------------------
Token                  delta_H    kl_div  conf_drop   sem_ent  Flag
--------------------------------------------------------------------------
'Marie'                 0.1231    0.0412     0.0821    0.3201
' Curie'                0.0892    0.0317     0.0612    0.2874
' Literature'          -0.4201    0.6123    -0.2812    1.0341  <- RISK
' 1911'                -0.1123    0.0891    -0.0723    0.4891

SEQUENCE-LEVEL SUMMARY
----------------------------------------
  Information Gain (delta_H) :   0.3821
  KL Divergence              :   0.1278
  Confidence Drop            :   0.1234
  Semantic Entropy           :   0.4238
  Composite Score            :   1.0571

======================================================================
  VERDICT: HALLUCINATION DETECTED
======================================================================
```

### Output explained

| Column | Meaning |
|---|---|
| `delta_H` | Entropy drop when context is added. Negative = model ignores context at this token |
| `kl_div` | KL divergence between with-context and without-context distributions |
| `conf_drop` | Drop in peak token probability. Negative = context reduced confidence |
| `sem_ent` | Entropy over top-5 predicted tokens. High = uncertain prediction |
| `<- RISK` | Flagged when `delta_H < 0` AND `kl_div > 0.3` simultaneously |
| `Composite Score` | Sum of all four signals. Positive = hallucination likely |

---

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--context` | Yes (or `--context_file`) | Retrieved context string passed inline |
| `--context_file` | Yes (or `--context`) | Path to a `.txt` file containing the context |
| `--passage` | Yes | The passage/response to evaluate |
| `--hf_token` | No | HuggingFace access token for gated models |

---

## Project Structure

```
.
├── demo.py              # End-to-end inference script (compre evaluation entry point)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── notebook.ipynb       # Full experimental notebook (Experiments 1-8)
└── references.bib       # IEEE-format bibliography
```

---

## How It Works

For each passage, `demo.py` runs two forward passes through Mistral-7B:

1. **With context:** `"Context: {retrieved_docs}\nPassage: {passage}"`
2. **Without context:** `"Passage: {passage}"`

The difference between the resulting token probability distributions reveals how much the model relies on the retrieved context at each token. Four uncertainty signals are extracted and fused into a composite score:

| Signal | Formula | Interpretation |
|---|---|---|
| Information Gain | H(without) − H(with) | Negative = context ignored |
| KL Divergence | KL(P_with ‖ P_without) | Low = predictions unchanged by context |
| Confidence Drop | max(P_with) − max(P_without) | Negative = context reduced confidence |
| Semantic Entropy | H over top-5 token probs | High = ambiguous prediction |

---

## Full Experiments (notebook.ipynb)

The notebook contains the complete experimental pipeline:

| Cell | Content |
|---|---|
| 1 | Dependency installation |
| 2 | Model loading |
| 3 | Dataset loading (RAGTruth + HaluEval) |
| 4 | Metric function definitions |
| 5 | Inference on RAGTruth test set (with checkpointing) |
| 6 | E1 & E2 — Composite AUROC ablation |
| 7 | E3 — Temporal precedence analysis |
| 8 | E4 — Cross-domain transfer (HaluEval) |
| 9 | E6 & E8 — Generator model breakdown + SOTA gap |
| 10 | Live demo pipeline |
| 11 | Final mark summary |

---

## References

- Xu et al., *RAGTruth*, ACL 2024
- Li et al., *HaluEval*, EMNLP 2023
- Manakul et al., *SelfCheckGPT*, EMNLP 2023
- Kuhn et al., *Semantic Uncertainty*, ICLR 2023
- Min et al., *FActScore*, EMNLP 2023
- Zhang et al., *ReDeEP*, ICLR 2025
- Lewis et al., *RAG*, NeurIPS 2020
