# NLP-Project

Dynamic Uncertainty-Aware Attribution for Hallucination Detection in Retrieval-Augmented Generation (RAG) systems.

This repository contains the implementation for CS F429 (Natural Language Processing), BITS Pilani Dubai Campus, Semester II 2025–26.

The project detects hallucinations in RAG systems using token-level uncertainty metrics derived from dual forward passes through a large language model.

---

## Project Overview

The system computes token-level uncertainty signals including:

- Prediction Entropy (with context)
- Prediction Entropy (without context)
- Information Gain
- KL Divergence
- Confidence Drop
- Semantic Entropy

These signals are combined into a composite hallucination score.

Model used:
- `mistralai/Mistral-7B-v0.1`

---

## Repository Structure

```text
NLP-Project/
│
├── demo.py                 # Main demo script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── references.bib          # Bibliography file
├── report.tex              # Main LaTeX report
├── arch.png                # Pipeline architecture figure
├── temporal_plot.png       # Temporal analysis plot
├── bits_logo.png           # BITS logo
├── tagline.jpg             # BITS tagline image
└── outputs/                # Generated outputs/results
```

---

## Python Version

Python 3.10

---

## Hardware

Experiments were conducted using:

- NVIDIA Tesla T4 GPU
- CUDA-enabled environment
- Kaggle Notebook runtime

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nehaur/NLP-Project.git
cd NLP-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Demo

Example usage:

```bash
python demo.py --passage "The Eiffel Tower is in Berlin." --context "The Eiffel Tower is located in Paris, France."
```

---

## Example Output

The script returns token-level uncertainty metrics such as:

```text
Token: Berlin
Entropy_with_context: 1.82
Entropy_without_context: 1.21
Information_Gain: -0.61
KL_Divergence: 0.77
Confidence_Drop: -0.19
Semantic_Entropy: 0.58
```

---

## Experimental Protocol

- All reported tables use held-out test-set numbers only.
- Validation results are not substituted into final result tables.
- Test data remained unseen until final evaluation.

---

## Statistical Settings

Bootstrap confidence intervals:
- 1000 resamples
- Replacement enabled
- Random seed = 42

Mann–Whitney U tests:
- Two-sided
- Significance threshold α = 0.05

---

## Reproducibility

The implementation uses:
- Fixed random seed
- Deterministic evaluation setup
- Explicit hyperparameter documentation in the report

---

## Dependencies

Main libraries used:

- torch
- transformers
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- accelerate
- tqdm
- sentencepiece

---

## Academic Context

Course:
- CS F429 — Natural Language Processing

Track:
- Track A — Dynamic Uncertainty-Aware Attribution

Institution:
- BITS Pilani, Dubai Campus

Semester:
- Second Semester 2025–26

---

## Citation

If referencing this work, please cite the accompanying report and referenced papers included in `references.bib`.

---

## Authors

- Ria Singh
- Neha Nair L

Under the supervision of:
- Prof. Elakkiya Rajasekar
