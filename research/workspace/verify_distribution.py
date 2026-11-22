"""Verify local distributed weights through their ordinary Transformers entry point."""
import gc
import json

import numpy as np
import torch

from encoder import components
from predict import JudgmentRoleClassifier
from program_trial import save_json
from weight_edit import LABELS, ORIGINAL, ROOT, file_hash, tensor
from windowing import content_windows, token_windows


SAMPLES = [
    "Notice was served on 3 May. We therefore allow the appeal.",
    "The case is listed before the Division Bench on Monday.",
    "The appellant submits that the order was made without notice.",
    "Learned counsel for the respondent supports the High Court's order.",
    "The question for determination is whether the appeal is within time.",
    "The evidence supports that inference because the two accounts agree.",
    "The claimant bought the property in 2004 and moved in the following year.",
    ("It was asserted that the document had been served. " * 110) + "We dismiss the petition.",
]


def main():
    folder = ROOT / "checkpoints" / "selected-direct"
    original, tokenizer, device = components(device="cpu")
    batch = tokenizer(SAMPLES, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        hidden = original.bert(**batch).pooler_output.cpu().numpy()
        pristine = original(**batch).logits.cpu().numpy()
    old_w, old_b = [tensor(ORIGINAL, name) for name in ("classifier.weight", "classifier.bias")]
    new_w, new_b = [tensor(folder / "model.safetensors", name) for name in ("classifier.weight", "classifier.bias")]
    del original
    gc.collect()
    packaged = JudgmentRoleClassifier(folder)
    packaged_batch = packaged.tokenizer(SAMPLES, padding=True, truncation=True, max_length=512, return_tensors="pt")
    assert all(torch.equal(packaged_batch[k], v) for k, v in batch.items())
    with torch.inference_mode():
        reloaded = packaged.model(**packaged_batch).logits.cpu().numpy()
    expected_delta = hidden @ (new_w - old_w).T + (new_b - old_b)
    error = float(np.max(np.abs(reloaded - pristine - expected_delta)))
    assert error < 0.0001, error
    one = [packaged.predict(text)["label"] for text in SAMPLES]
    short = [i for i, s in enumerate(SAMPLES) if len(tokenizer.tokenize(s)) <= 510]
    assert all(one[i] == LABELS[int(reloaded[i].argmax())] for i in short)
    content = packaged.tokenizer(SAMPLES[-1],add_special_tokens=False,truncation=False,verbose=False)["input_ids"]
    spans = content_windows(len(content),510)
    assert spans[0][0]==0 and spans[-1][1]==len(content)
    assert sum(item[2] for item in spans)==len(content) and all(v[2]>0 for v in spans)
    windows = token_windows(packaged.tokenizer, content, 512)
    expected = np.zeros(7)
    with torch.inference_mode():
        for token_ids, weight in windows:
            result = packaged.model(input_ids=torch.tensor([token_ids]),attention_mask=torch.ones(1,len(token_ids),dtype=torch.long))
            expected += result.logits[0].cpu().numpy().astype(float) * weight
    assert one[-1] == LABELS[int(expected.argmax())]
    try:
        packaged.predict("  ")
    except ValueError:
        pass
    else:
        raise AssertionError("Blank input must fail")
    audit = {"sample_count": len(SAMPLES), "includes_overlength_excerpt": True,
             "empty_input_rejected": True, "single_window_batch_matches_individual_calls": True,
             "full_excerpt_matches_independent_window_aggregation": True,"overlength_window_count":len(windows),
             "target_sha256": file_hash(folder / "model.safetensors"), "parameters": sum(p.numel() for p in packaged.model.parameters()),
             "max_logit_delta_error": error,
             "examples": [{"text": text, "label": label} for text, label in zip(SAMPLES[:7], one[:7])]}
    save_json(ROOT / "results" / "distribution-check.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
