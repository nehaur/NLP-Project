import argparse
import torch
import numpy as np
from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()

# ---------------------------------------------------
# Argument Parser
# ---------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--passage",
    type=str,
    required=True,
    help="Generated response / passage"
)

parser.add_argument(
    "--context",
    type=str,
    required=True,
    help="Retrieved context"
)

args = parser.parse_args()

passage = args.passage
context = args.context

# ---------------------------------------------------
# Build Inputs
# ---------------------------------------------------

with_context = f"Context: {context}\nPassage: {passage}"
without_context = f"Passage: {passage}"

# ---------------------------------------------------
# Tokenize
# ---------------------------------------------------

enc_with = tokenizer(
    with_context,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(model.device)

enc_without = tokenizer(
    without_context,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(model.device)

# ---------------------------------------------------
# Forward Passes
# ---------------------------------------------------

with torch.no_grad():
    out_with = model(**enc_with)
    out_without = model(**enc_without)

logits_with = out_with.logits[0]
logits_without = out_without.logits[0]

probs_with = torch.softmax(logits_with, dim=-1)
probs_without = torch.softmax(logits_without, dim=-1)

# ---------------------------------------------------
# Tokens
# ---------------------------------------------------

tokens = tokenizer.convert_ids_to_tokens(
    enc_with["input_ids"][0]
)

# ---------------------------------------------------
# Compute Metrics
# ---------------------------------------------------

print("\n================ TOKEN LEVEL SCORES ================\n")

print(
    f"{'TOKEN':15} {'H_with':10} {'H_without':12} "
    f"{'IG':10} {'KL':10} {'ConfDrop':10}"
)

print("-" * 75)

max_tokens_to_show = min(len(tokens), 30)

for i in range(max_tokens_to_show):

    pw = probs_with[i].cpu().numpy()
    pwo = probs_without[i].cpu().numpy()

    # Entropy
    h_with = entropy(pw)
    h_without = entropy(pwo)

    # Information Gain
    info_gain = h_without - h_with

    # KL Divergence
    kl = np.sum(
        pw * np.log((pw + 1e-12) / (pwo + 1e-12))
    )

    # Confidence Drop
    conf_with = np.max(pw)
    conf_without = np.max(pwo)

    conf_drop = conf_without - conf_with

    token = tokens[i]

    print(
        f"{token[:14]:15} "
        f"{h_with:<10.4f} "
        f"{h_without:<12.4f} "
        f"{info_gain:<10.4f} "
        f"{kl:<10.4f} "
        f"{conf_drop:<10.4f}"
    )

print("\nDone.\n")
