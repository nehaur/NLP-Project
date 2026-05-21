import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MAX_LEN = 600

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()


def get_token_signals(text_with, text_without):
    enc_w = tokenizer(
        text_with,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(model.device)

    enc_wo = tokenizer(
        text_without,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(model.device)

    with torch.no_grad():
        out_w = model(**enc_w)
        out_wo = model(**enc_wo)

    seq = min(out_w.logits.shape[1], out_wo.logits.shape[1])

    logits_w = out_w.logits[0, :seq].float()
    logits_wo = out_wo.logits[0, :seq].float()

    probs_w = torch.softmax(logits_w, dim=-1)
    probs_wo = torch.softmax(logits_wo, dim=-1)

    log_w = torch.log(probs_w + 1e-10)
    log_wo = torch.log(probs_wo + 1e-10)

    h_with = (-torch.sum(probs_w * log_w, dim=-1)).cpu().numpy()
    h_without = (-torch.sum(probs_wo * log_wo, dim=-1)).cpu().numpy()
    delta_h = h_without - h_with

    kl_div = torch.sum(
        probs_w * (log_w - log_wo),
        dim=-1
    ).cpu().numpy()

    conf_w = probs_w.max(dim=-1).values.cpu().numpy()
    conf_wo = probs_wo.max(dim=-1).values.cpu().numpy()
    conf_drop = conf_w - conf_wo

    top5 = probs_w.topk(5, dim=-1).values
    top5 = top5 / top5.sum(dim=-1, keepdim=True)

    sem_ent = (-torch.sum(
        top5 * torch.log(top5 + 1e-10),
        dim=-1
    )).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(enc_w["input_ids"][0][:seq])

    results = []

    for i in range(seq):
        results.append({
            "token": tokens[i],
            "H_with": float(h_with[i]),
            "H_without": float(h_without[i]),
            "delta_H": float(delta_h[i]),
            "info_gain": float(delta_h[i]),
            "kl_div": float(kl_div[i]),
            "conf_drop": float(conf_drop[i]),
            "semantic_entropy": float(sem_ent[i])
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passage", required=True)
    parser.add_argument("--context", required=True)

    args = parser.parse_args()

    text_with = f"Context: {args.context}\nPassage: {args.passage}"
    text_without = f"Passage: {args.passage}"

    scores = get_token_signals(text_with, text_without)

    for row in scores:
        print(row)
