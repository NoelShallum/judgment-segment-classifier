"""Train a resumable compact single-excerpt challenger using only training labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from program_trial import metric, official_sample, save_json
from weight_edit import LABELS, ROOT, TRAIN_COUNTS, file_hash


BASE = ROOT / "references/minilm"
SEED = 8152403
ALPHAS = (0., .25, .5, .75, 1.)
GROUP_SIZE = 256
BATCH_SIZE = 16
MAX_LENGTH = 192


def make_batches(ids, lengths, seed, training):
    ids = list(ids)
    rng = random.Random(seed)
    if training:
        rng.shuffle(ids)
    blocks = []
    width = GROUP_SIZE if training else len(ids)
    for start in range(0, len(ids), width):
        ordered = sorted(ids[start:start+width], key=lengths.__getitem__)
        blocks.extend(ordered[i:i+BATCH_SIZE] for i in range(0, len(ordered), BATCH_SIZE))
    if training:
        rng.shuffle(blocks)
    return blocks


def tokenize(tok, texts):
    return tok(texts, truncation=True, max_length=MAX_LENGTH, padding=False,
               return_token_type_ids=False, return_attention_mask=False)["input_ids"]


def pack(tok, encoded, indices, device):
    return tok.pad([{"input_ids": encoded[i]} for i in indices],
                   padding=True, return_tensors="pt").to(device)


@torch.inference_mode()
def logits(model, tok, encoded, device):
    model.eval()
    values = np.empty((len(encoded), len(LABELS)), dtype=np.float32)
    lens = list(map(len, encoded))
    for indices in make_batches(range(len(encoded)), lens, SEED, False):
        values[indices] = model(**pack(tok, encoded, indices, device)).logits.float().cpu().numpy()
    return values


def adjusted(logits_, alpha):
    return logits_ + alpha*np.log(TRAIN_COUNTS / TRAIN_COUNTS.sum())


def save_state(run, model, optimizer, scheduler, step, history, fingerprint):
    path = run / "latest.pt"
    tmp = run / "latest.pt.partial"
    state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "history": history, "input_fingerprint": fingerprint}
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-role", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = ap.parse_args()
    if args.per_role < 32 or args.epochs < 1:
        ap.error("Use at least 32 examples per role and at least one epoch.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.getenv("ROLE_CPU_THREADS","2")))
    run = ROOT / "training-minilm" / f"balanced-{args.per_role}-e{args.epochs}"
    run.mkdir(parents=True, exist_ok=True)
    target = ROOT / "checkpoints" / f"minilm-{args.per_role}-e{args.epochs}"
    verified = json.loads((BASE / "verified.json").read_text())
    base_hash = next(x["sha256"] for x in verified if x["file"] == "model.safetensors")
    if file_hash(BASE / "model.safetensors") != base_hash:
        raise ValueError("Wrong pretrained MiniLM snapshot")
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.truncation_side = "left"
    rows, _, stats = official_sample("train", args.per_role)
    source_examples = json.loads((ROOT / "results/gemini-synthetic-review.json").read_text())
    synthetic = {ex["id"]: ex for ex in (json.loads(s) for s in (ROOT / "synthetic/original.jsonl").read_text().splitlines())}
    reviews = [json.loads(s) for s in (ROOT / "synthetic/gemini-reviews.jsonl").read_text().splitlines()]
    added = [synthetic[r["id"]] for r in reviews if r["agrees_with_original"]]
    assert len(added) == source_examples["agreed"] and len(reviews) == source_examples["requested"]
    texts = [r[2] for r in rows] + [ex["text"] for ex in added]
    y_train = torch.tensor([r[3] for r in rows] + [LABELS.index(ex["intended_role"]) for ex in added], dtype=torch.long)
    ids = tokenize(tok, texts)
    lengths = list(map(len, ids))
    input_hash = hashlib.sha256(json.dumps([[r[0],r[3]] for r in rows]+[[e["id"],e["intended_role"]] for e in added]).encode()).hexdigest()
    selection, weights, dev_stats = official_sample("select",160)
    dev_y = [r[3] for r in selection]
    dev_ids = tokenize(tok, [r[2] for r in selection])
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=7, id2label=dict(enumerate(LABELS)),
              label2id={r:i for i,r in enumerate(LABELS)}, local_files_only=True).to(device)
    for layer in (model.bert.embeddings, *model.bert.encoder.layer[:-3]):
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=4e-5, weight_decay=.01)
    batches = []
    ends = []
    for epoch in range(args.epochs):
        b = make_batches(range(len(ids)), lengths, SEED+epoch, True)
        offset = len(batches)
        ends.extend((offset + len(b)//2, offset + len(b)))
        batches.extend(b)
    total = len(batches)
    warmup = max(1, total//12)
    def rate(step):
        return max(0., min((step+1)/warmup, (total-step)/max(1,total-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    start = 0
    history = []
    if (run/"latest.pt").is_file():
        state = torch.load(run/"latest.pt", map_location=device, weights_only=True)
        if state["input_fingerprint"] != input_hash:
            raise ValueError("Cannot resume a different training sample")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history = state["step"], state["history"]
        del state
    spec={"base":"sentence-transformers/all-MiniLM-L6-v2","base_sha256":base_hash,"input":"one standalone excerpt",
          "train_documents":stats["sample_documents"],"gold_examples":len(rows),"reviewed_examples":len(added),
          "max_tokens":MAX_LENGTH,"batch_size":BATCH_SIZE,"epochs":args.epochs,"selected_validation_documents":dev_stats["sample_documents"],
          "trainable_parameters":sum(p.numel() for p in trainable),"parameters":sum(p.numel() for p in model.parameters()),
          "seed":SEED,"planned_steps":total,"device":device,"training_input_fingerprint":input_hash,
          "selection_protocol":"Choose checkpoint and class-prior coefficient by prevalence-weighted selection macro-F1. Evaluate selected checkpoint on case-disjoint confirmation. Repeatable per-step dropout seed and length-bucketed minibatches."}
    save_json(run/"protocol.json",spec)
    print("Training",device,"total",total,"starting",start,"gold",len(rows),"synthetic",len(added),flush=True)
    best = max((x["macro_f1"] for x in history), default=-1)
    first = time.monotonic()
    for ix in range(start, total):
        model.train()
        torch.manual_seed(SEED+ix)
        batch = batches[ix]
        optimizer.zero_grad(set_to_none=True)
        loss = model(**pack(tok,ids,batch,device), labels=y_train[batch].to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.)
        optimizer.step()
        scheduler.step()
        step = ix+1
        if step in ends:
            dev_logits = logits(model,tok,dev_ids,device)
            choices = [(metric(dev_y,adjusted(dev_logits,a).argmax(1),weights)["macro_f1"],a) for a in ALPHAS]
            score, alpha = max(choices,key=lambda pair:pair[0])
            history.append({"step":step,"macro_f1":score,"prior":alpha})
            if score > best:
                target.mkdir(parents=True,exist_ok=True)
                # Export the prior offset as ordinary classifier biases in a standalone model.
                bias = model.classifier.bias
                original_bias = bias.detach().clone()
                with torch.no_grad():
                    bias.add_(torch.as_tensor(alpha*np.log(TRAIN_COUNTS / TRAIN_COUNTS.sum()), dtype=bias.dtype,device=device))
                    model.save_pretrained(target)
                    bias.copy_(original_bias)
                tok.init_kwargs.update(truncation_side="left",model_max_length=MAX_LENGTH)
                tok.save_pretrained(target)
                save_json(target/"selection.json",history[-1])
                best=score
            print("Validation",history[-1],flush=True)
        if step % 125 == 0 or step in ends:
            save_state(run,model,optimizer,scheduler,step,history,input_hash)
        if step == 1 or step % 50 == 0:
            print(f"Step {step}/{total}, loss {float(loss.detach()):.3f}, elapsed {time.monotonic()-first:.0f}s",flush=True)
    del model, optimizer, scheduler, trainable
    frozen = AutoModelForSequenceClassification.from_pretrained(target,local_files_only=True).to(device)
    confirm, weights, confirm_stats = official_sample("confirm",160)
    cz = logits(frozen,tok,tokenize(tok,[r[2] for r in confirm]),device)
    saved = {"protocol":spec,"history":history,"selected":json.loads((target/"selection.json").read_text()),
             "confirmation_sample":confirm_stats,"confirmation":metric([r[3] for r in confirm],cz.argmax(1),weights),
             "target_sha256":file_hash(target/"model.safetensors"),"official_test_used":False}
    save_json(ROOT/"results"/f"minilm-{args.per_role}-e{args.epochs}-validation.json",saved)
    print("Frozen confirmation macro-F1",saved["confirmation"]["macro_f1"],flush=True)


if __name__ == "__main__":
    main()
