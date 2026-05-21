# NLP-Project

This repository contains code for hallucination detection in Retrieval-Augmented Generation systems using uncertainty-aware attribution metrics.

## Python Version

Python 3.10

## Environment

CUDA was used for the experiments.  
Model used: mistralai/Mistral-7B-v0.1

## Installation

```bash
git clone https://github.com/nehaur/NLP-Project.git
cd NLP-Project
pip install -r requirements.txt
```

## Run Demo

```bash
python demo.py --passage "The Eiffel Tower is in Berlin." --context "The Eiffel Tower is located in Paris, France."
```

The script returns token-level scores including:

- prediction entropy with context
- prediction entropy without context
- information gain
- KL divergence
- confidence drop
- semantic entropy

## Data Protocol

All reported tables use held-out test-set numbers only.

No validation figures are substituted into final result tables.

## Statistical Settings

Bootstrap confidence intervals use:

- 1000 resamples
- replacement enabled
- seed = 42

Mann–Whitney U tests are:

- two-sided
- alpha = 0.05
