"""Compare written DeBERTa-v3-xsmall readouts on whole LegalSeg judgments."""
import numpy as np
from safetensors.numpy import load_file
from scipy.linalg import solve
import torch

from discourse_span_edit import text_features
from encoder_source_key_write import source_keys
import encoder_span_edit as study
from hier_model import CRF, HierModel, path_labels
from hier_trial import DONOR, OUT
from program_trial import fixture_rows, save_json
from role_fixtures import ANCHORS, CHECKS
from seven_paths import ROOT
from span_eval import counts, summarize

CONTEXT_KEYS_PER_ROLE = 48


def readouts(keys, roles, source):
    """Write cosine prototypes and a single weighted ridge correction."""
    means = np.stack([keys[roles == role].mean(0) for role in range(7)])
    prototype = study.normalized(means)
    yield f"{source} prototypes", prototype, np.zeros(7)
    keys_augmented = np.column_stack((keys, np.ones(len(keys)) * .3))
    scores = keys @ prototype.T
    target = np.eye(7)[roles] - scores
    scale = (keys_augmented ** 2).sum() / keys_augmented.shape[1]
    gram = keys_augmented.T @ keys_augmented
    for damping in (.1, 1.):
        delta = solve(gram + damping * scale * np.eye(len(gram)),
                      keys_augmented.T @ target, assume_a="pos")
        corrected = prototype + delta[:-1].T
        yield f"{source} residual ridge {damping}", corrected, .3 * delta[-1]


def compare_discourse(tasks):
    model = HierModel(DONOR)
    control = {}
    for task, groups in tasks.items():
        w = load_file(str(OUT / f"{task}-discourse-exact.safetensors"))
        results = {}
        for split, cases in groups.items():
            by_case = []
            for d in cases.values():
                cue, pos = text_features(d)
                words = model.words([d["text"][a:b] for a, b in d["units"]])
                logits = model.logits(model.contextualize(words))
                emission = logits + cue @ w["cue_head"] + pos @ w["position_head"]
                by_case.append(counts(d, d["units"], path_labels(emission, w[CRF])))
            results[split] = summarize(np.sum(by_case, axis=0))
        control[task] = results
    return control


def main():
    torch.set_num_threads(2)
    study.CACHE = study.CACHE / "deberta-v3-xsmall"
    source, targets = source_keys()
    authors = fixture_rows(ANCHORS) + fixture_rows(CHECKS)
    authored = {"author-" + str(i): [row[2]] for i, row in enumerate(authors)}
    authored_roles = np.asarray([row[3] for row in authors])
    groups, documents = study.make_documents()
    # Whole judgment is the entry point with clean, full-case prior evidence.
    groups = {"whole": {k: v for k, v in groups["whole"].items() if k != "cloud"}}
    documents = {k: v for k, v in documents.items() if k.startswith("whole/") and not k.startswith("whole/cloud/")}
    encoder = study.Encoder("deberta")
    print("Frozen DeBERTa encoder", encoder.params, "parameters", flush=True)
    modes = {}
    banks = {}
    contextual_names = [key for role in range(7) for key in
                        [k for k in source if targets[k][1] == role][::2][:CONTEXT_KEYS_PER_ROLE]]
    contextual = {}
    anchors = {}
    for key in contextual_names:
        center, _ = targets[key]
        start = max(0, center - 1)
        contextual[key] = source[key][start:center + 2]
        anchors[key] = center - start
    isolated = {key: [texts[targets[key][0]]] for key, texts in source.items()}
    for mode in study.MODES:
        print("Writing", mode, "from source and authored keys", flush=True)
        examples = isolated if mode == "isolated" else contextual
        keys = encoder.cache(examples, mode)
        text_keys = np.stack([keys[key][0 if mode == "isolated" else anchors[key]] for key in examples])
        labels = np.asarray([targets[key][1] for key in examples])
        authored_bank = encoder.cache(authored, mode)
        authored_keys = np.stack([authored_bank[key][0] for key in authored])
        modes[mode] = [*readouts(text_keys, labels, "official train"),
                       *readouts(authored_keys, authored_roles, "authored")]
        banks[mode] = encoder.cache(documents, mode)
    result = study.evaluate("deberta-v3-xsmall", groups, banks, modes)
    control = compare_discourse(groups)
    payload = {
        "model": study.MODELS["deberta"], "encoder_parameters": encoder.params,
        "method": "Write seven readout rows with authored prototypes or source-label prototypes and closed-form residual associations. The DeBERTa backbone stays frozen. Choose the profile and CRF scale using case-disjoint validation, before scoring the screening test. Includes both encoder-only emissions and blends, all under one billion parameters.",
        "source_keys": {"isolated": len(isolated), "bidirectional": len(contextual)},
        "authored_keys": len(authors), "steps": 0,
        "validation_selected": result, "current_discourse_control": control,
    }
    save_json(ROOT / "results/deberta-span-write.json", payload)
    print({label: (p["edit"], p["scores"]["test"]["candidate"]["exact_labeled_spans"]["f1"])
           for label, p in result["whole"].items() if label != "grid"}, flush=True)
    print("Current discourse control", control["whole"]["test"]["exact_labeled_spans"]["f1"], flush=True)


if __name__ == "__main__":
    main()
