"""Return one of the seven LegalSeg roles for a nonblank sentence or paragraph."""
import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from windowing import token_windows


def load_model(directory):
    tokenizer = BertTokenizerFast.from_pretrained(directory, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(directory, local_files_only=True, attn_implementation="eager")
    return tokenizer, model.eval()


@torch.inference_mode()
def classify(tokenizer, model, text):
    if not text.strip():
        raise ValueError("Supply a nonblank sentence or paragraph.")
    content = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    windows = token_windows(tokenizer, content, model.config.max_position_embeddings)
    summed = torch.zeros(model.config.num_labels, dtype=torch.float64)
    for ids, mass in windows:
        input_ids = torch.tensor([ids], dtype=torch.long)
        logits = model(input_ids, attention_mask=torch.ones_like(input_ids)).logits[0]
        summed += logits.double() * mass
    return {"label": model.config.id2label[int(summed.argmax())],
            "wordpieces": len(content), "windows": len(windows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("text", nargs="?", help="Read from stdin when omitted")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Use a positive thread count.")
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("Supply a nonblank sentence or paragraph.")
    torch.set_num_threads(args.threads)
    tokenizer, model = load_model(args.model)
    print(json.dumps(classify(tokenizer, model, text)))


if __name__ == "__main__":
    main()
