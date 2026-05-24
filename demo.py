"""
demo.py — Dynamic Uncertainty-Aware Attribution for Hallucination Detection
CS F429 Natural Language Processing — Track A

Usage:
    python demo.py --context "Your retrieved context here" --passage "The passage to evaluate"
    python demo.py --context_file context.txt --passage "The passage to evaluate"
"""

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MAX_LEN    = 600
HF_TOKEN   = None   # Set via --hf_token if the model requires auth

# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Token-level hallucination detection via uncertainty attribution."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--context",      type=str, help="Retrieved context string (inline)")
    group.add_argument("--context_file", type=str, help="Path to a plain-text file containing the context")

    parser.add_argument("--passage",   type=str, required=True,
                        help="The passage/response to evaluate for hallucination")
    parser.add_argument("--hf_token",  type=str, default=None,
                        help="HuggingFace token (only needed for gated models)")
    return parser.parse_args()

# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------
def load_model(hf_token=None):
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"Mem: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    else:
        print("Running on CPU — inference will be slower")

    return tokenizer, model

# -----------------------------------------------------------------------
# Two-pass inference
# -----------------------------------------------------------------------
def two_forward_passes(tokenizer, model, text_with, text_without):
    enc_w  = tokenizer(text_with,    return_tensors="pt",
                       truncation=True, max_length=MAX_LEN).to(model.device)
    enc_wo = tokenizer(text_without, return_tensors="pt",
                       truncation=True, max_length=MAX_LEN).to(model.device)

    with torch.no_grad():
        out_w  = model(**enc_w)
        out_wo = model(**enc_wo)

    seq = min(out_w.logits.shape[1], out_wo.logits.shape[1])
    return out_w.logits[0, :seq].float(), out_wo.logits[0, :seq].float()

# -----------------------------------------------------------------------
# Uncertainty signals
# -----------------------------------------------------------------------
def get_token_signals(tokenizer, model, text_with, text_without):
    logits_w, logits_wo = two_forward_passes(tokenizer, model, text_with, text_without)

    probs_w  = torch.softmax(logits_w,  dim=-1)
    probs_wo = torch.softmax(logits_wo, dim=-1)
    log_w    = torch.log(probs_w  + 1e-10)
    log_wo   = torch.log(probs_wo + 1e-10)

    # Entropy with / without context
    H_with    = (-torch.sum(probs_w  * log_w,  dim=-1)).cpu().numpy()
    H_without = (-torch.sum(probs_wo * log_wo, dim=-1)).cpu().numpy()

    # Information gain: entropy drops when context is useful
    delta_H   = H_without - H_with

    # KL divergence: distribution shift due to context
    kl_div    = torch.sum(probs_w * (log_w - log_wo), dim=-1).cpu().numpy()
    kl_div    = np.clip(kl_div, 0, None)

    # Confidence drop: peak probability shift
    conf_w    = probs_w.max(dim=-1).values.cpu().numpy()
    conf_wo   = probs_wo.max(dim=-1).values.cpu().numpy()
    conf_drop = conf_w - conf_wo

    # Semantic entropy: uncertainty over top-5 tokens
    top5      = probs_w.topk(5, dim=-1).values
    top5      = top5 / top5.sum(dim=-1, keepdim=True)
    sem_ent   = (-torch.sum(top5 * torch.log(top5 + 1e-10), dim=-1)).cpu().numpy()

    return {
        "delta_H":   delta_H,
        "kl_div":    kl_div,
        "conf_drop": conf_drop,
        "sem_ent":   sem_ent,
    }

def composite_score(signals):
    """Scalar composite from mean of each signal (z-norm not applied for single sample)."""
    ig  = -float(np.nanmean(signals["delta_H"]))
    kl  =  float(np.nanmean(signals["kl_div"]))
    cd  = -float(np.nanmean(signals["conf_drop"]))
    se  =  float(np.nanmean(signals["sem_ent"]))
    return ig + kl + cd + se

def lexical_overlap(context, passage):
    c = set(context.lower().split())
    p = set(passage.lower().split())
    return len(c & p) / max(len(p), 1)

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve context
    if args.context_file:
        with open(args.context_file, "r", encoding="utf-8") as f:
            context = f.read().strip()
    else:
        context = args.context.strip()

    passage   = args.passage.strip()
    hf_token  = args.hf_token or HF_TOKEN

    print("\n" + "=" * 70)
    print("DYNAMIC UNCERTAINTY-AWARE HALLUCINATION DETECTION")
    print("=" * 70)
    print(f"Context : {context[:120]}{'...' if len(context) > 120 else ''}")
    print(f"Passage : {passage}")
    print("=" * 70)

    # Load model
    tokenizer, model = load_model(hf_token)

    # Build inputs
    text_with    = f"Context: {context}\nPassage: {passage}"
    text_without = f"Passage: {passage}"

    # Run inference
    print("\nRunning two-pass inference...")
    signals = get_token_signals(tokenizer, model, text_with, text_without)

    # Decode passage tokens
    enc    = tokenizer(passage, return_tensors="pt",
                       truncation=True, max_length=MAX_LEN)
    tokens = [tokenizer.decode([t]) for t in enc["input_ids"][0]]
    n      = min(len(tokens), len(signals["delta_H"]))

    # ── Token-level table ──────────────────────────────────────────────
    print("\nTOKEN-LEVEL UNCERTAINTY SIGNALS")
    print("-" * 74)
    print(f"{'Token':<22} {'delta_H':>9} {'kl_div':>9} {'conf_drop':>10} {'sem_ent':>9}  Flag")
    print("-" * 74)

    for i in range(n):
        tok = repr(tokens[i])
        dh  = signals["delta_H"][i]
        kl  = signals["kl_div"][i]
        cd  = signals["conf_drop"][i]
        se  = signals["sem_ent"][i]
        flag = "  ← RISK" if (dh < 0 and kl > 0.3) else ""
        print(f"{tok:<22} {dh:>9.4f} {kl:>9.4f} {cd:>10.4f} {se:>9.4f}{flag}")

    # ── Sequence-level summary ─────────────────────────────────────────
    comp = composite_score(signals)

    print("\nSEQUENCE-LEVEL SUMMARY")
    print("-" * 40)
    print(f"  Information Gain (delta_H) : {-float(np.nanmean(signals['delta_H'])):>8.4f}")
    print(f"  KL Divergence              : {float(np.nanmean(signals['kl_div'])):>8.4f}")
    print(f"  Confidence Drop            : {-float(np.nanmean(signals['conf_drop'])):>8.4f}")
    print(f"  Semantic Entropy           : {float(np.nanmean(signals['sem_ent'])):>8.4f}")
    print(f"  Composite Score            : {comp:>8.4f}")

    # ── Verdict ────────────────────────────────────────────────────────
    overlap = lexical_overlap(context, passage)
    if overlap > 0.8:
        verdict = "LIKELY FAITHFUL  (high lexical overlap with context)"
    elif comp > 0:
        verdict = "HALLUCINATION DETECTED"
    else:
        verdict = "LIKELY FAITHFUL"

    print("\n" + "=" * 70)
    print(f"  VERDICT: {verdict}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
