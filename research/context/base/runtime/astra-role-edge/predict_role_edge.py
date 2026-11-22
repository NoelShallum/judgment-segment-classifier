"""Assign rhetorical roles to raw judgment spans or supplied sentence units."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from safetensors.numpy import load_file
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hier_model import CRF, HierModel, LABELS
from role_cues import cue_features
from role_edge_runtime import decode, decompose, transition_scores
from span_model import merge, units


def unit_features(text, spans, model, *, with_states=False):
    pieces = [text[a:b] for a, b in spans]
    places = [(a + b) * .5 / len(text) for a, b in spans]
    position = np.eye(5)[np.digitize(places, [.05, .25, .60, .91])]
    states = model.contextualize(model.words(pieces))
    features=model.logits(states),cue_features(pieces),position
    return (*features,states) if with_states else features


def label_units(text, spans, model, parent, vocab, memory, mode, gain):
    joint="joint.settings" in parent
    base,cue,position,*states = unit_features(text,spans,model,with_states=joint)
    logits = base + cue @ parent["cue_head"] + position @ parent["position_head"]
    edge = transition_scores(text, spans, vocab, memory)
    edge = decompose(edge, mode)
    result = decode_profile(logits, edge, parent, gain)
    if joint:
        from sentence_joint_runtime import joint_labels
        lengths=[len(text[a:b].strip()) for a,b in spans]
        if "local_readout.settings" in parent:
            from sentence_local_readout_runtime import labels
            return labels(logits,edge,parent,gain,result,lengths,states[0])
        return joint_labels(logits,edge,parent,gain,result,lengths,states[0])
    if "uncertainty.settings" in parent:
        from sentence_uncertainty_runtime import uncertainty_labels
        return uncertainty_labels(logits,edge,parent,gain,result,[len(text[a:b].strip()) for a,b in spans])
    return result


def decode_profile(logits, edge, parent, gain):
    preceding=decode(logits,parent[CRF],edge,gain)
    if "label_change.utility" not in parent:
        return preceding
    calibrated=logits@parent["emission.weight"]+parent["emission.bias"]
    alternative=decode(calibrated,parent[CRF],edge,gain)
    utility=parent["label_change.utility"]
    if utility.ndim==3:
        i=np.arange(len(preceding))
        margin=calibrated[i,alternative]-calibrated[i,preceding]
        buckets=np.digitize(margin,parent["label_change.margin_bins"])
        value=utility[preceding,alternative,buckets]
    else:
        value=utility[preceding,alternative]
    return np.where(value>parent["label_change.threshold"].item(),alternative,preceding)


def infer(text, model, parent, vocab, memory, mode, gain):
    spans = units(text)
    labels = label_units(text, spans, model, parent, vocab, memory, mode, gain)
    return {"spans": merge(spans, labels, LABELS)}


def infer_segments(text, model, parents, vocab, weights, dense=None):
    from segment_span_runtime import density_labels, segment_labels

    bounds = units(text)
    raw,cue,positions = unit_features(text,bounds,model)
    logits = {name:raw+cue@patch["cue_head"]+positions@patch["position_head"]
              for name,patch in parents.items()}
    offsets = [a for a,_ in bounds]
    tokens = np.fromiter((m.start() for m in re.finditer(r"\S+",text)),dtype=np.int64)
    counts = np.bincount(np.searchsorted(offsets,tokens,side="right")-1,minlength=len(bounds))
    edge = transition_scores(text,bounds,vocab,parents["exact"])
    labels=(segment_labels(logits,edge,parents,counts,weights) if dense is None
            else density_labels(logits,edge,parents,counts,weights,dense))
    return {"spans":merge(bounds,labels,LABELS)}


def infer_sentences(sentences, model, parent, vocab, memory, mode, gain):
    if not isinstance(sentences,list) or not sentences or any(not isinstance(s,str) or not s.strip() for s in sentences):
        raise ValueError("Supply a nonempty JSON array of nonblank sentence strings")
    pieces=[s.strip() for s in sentences]
    text=" ".join(pieces)
    starts=[0]
    for piece in pieces[:-1]:
        starts.append(starts[-1]+len(piece)+1)
    bounds=list(zip(starts,starts[1:]+[len(text)]))
    labels=label_units(text,bounds,model,parent,vocab,memory,mode,gain)
    return {"labels":[LABELS[int(label)] for label in labels]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_file", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--profile", default="balanced", help="A profile listed in deployment.json")
    parser.add_argument("--sentences", action="store_true", help="Read a JSON array of supplied sentences from one judgment")
    parser.add_argument("--patch-dir", type=Path, default=HERE)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest=json.loads((args.patch_dir / "deployment.json").read_text())
    if args.profile not in manifest["profiles"]:
        parser.error(f"Unknown profile {args.profile!r}. Choose from {', '.join(manifest['profiles'])}.")
    selected=manifest["profiles"][args.profile]
    if selected.get("input_format")=="sentences" and not args.sentences:
        parser.error("This profile requires --sentences and a JSON array of supplied sentences")
    if selected.get("input_format")=="raw" and args.sentences:
        parser.error("This profile requires raw judgment text")
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    patch_path=args.patch_dir / selected["file"]
    vocab_path=args.patch_dir / "vocab.json"
    checks=[(patch_path, selected["sha256"]), (vocab_path, manifest["vocab_sha256"]),
            (args.model_dir / "hier.safetensors", manifest["base_sha256"])]
    sources={}
    for name in selected.get("parent_profiles",[]):
        spec=manifest["profiles"][name]
        path=args.patch_dir/spec["file"]
        sources[name]=path
        checks.append((path,spec["sha256"]))
    for path, expected in checks:
        if digest(path)!=expected:
            raise ValueError(f"Expected different model data at {path}")
    patch=load_file(str(patch_path))
    vocab=json.loads(vocab_path.read_text())
    text=args.text_file.read_text()
    model=HierModel(args.model_dir)
    if sources:
        parents={name:load_file(str(path)) for name,path in sources.items()}
        print(json.dumps(infer_segments(text,model,parents,vocab,patch),ensure_ascii=False))
        return
    if "isolated.matrix" in patch:
        from isolated_role_readout_runtime import infer as infer_isolated
        print(json.dumps(infer_isolated(json.loads(text),model,patch),ensure_ascii=False))
        return
    mode="interaction" if patch["edge.mode"].item() else "joint"
    if args.sentences:
        output=infer_sentences(json.loads(text),model,patch,vocab,patch,mode,patch["edge.gain"].item())
    else:
        output=infer(text,model,patch,vocab,patch,mode,patch["edge.gain"].item())
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
