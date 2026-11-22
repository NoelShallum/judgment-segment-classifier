"""Measure the real inputs accepted by the released CPU sentence interface."""
from collections import defaultdict
import csv
import json
from resource import getrusage, RUSAGE_SELF
from time import perf_counter

from hier_model import PUNCTUATION
from sentence_document_graph_write import DONOR, EDGE, OUT, ROOT, HierModel, file_hash, load_file, save_json, torch
from predict_role_edge import infer_sentences


def main():
    path = ROOT/"references/official/test.csv"
    rows = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if text := row["Text"].strip():
                rows[row["Index"]].append(text)
    lengths = {k:[max(1,len(x.lower().translate(PUNCTUATION).split())) for x in v] for k,v in rows.items()}
    token_case = max(lengths,key=lambda k:sum(lengths[k]))
    sentence_case = max(lengths,key=lambda k:len(lengths[k]))
    individual_case = max(lengths,key=lambda k:max(lengths[k]))
    individual_index = lengths[individual_case].index(max(lengths[individual_case]))
    torch.set_num_threads(2)
    model = HierModel(DONOR)
    source = OUT/"sentence-document-readout-portable.safetensors"
    weights = load_file(str(source))
    vocab = json.loads((EDGE/"vocab.json").read_text())
    gain = float(weights["edge.gain"])
    probes = (
        ("Most tokens in a judgment",token_case,rows[token_case],lengths[token_case]),
        ("Most supplied sentences in a judgment",sentence_case,rows[sentence_case],lengths[sentence_case]),
        ("Longest single supplied sentence",individual_case,[rows[individual_case][individual_index]],
         [lengths[individual_case][individual_index]]),
    )
    infer_sentences(rows[token_case][:2],model,weights,vocab,weights,"joint",gain)
    report = []
    for purpose,case,content,count in probes:
        start = perf_counter()
        result = infer_sentences(content,model,weights,vocab,weights,"joint",gain)
        elapsed = perf_counter()-start
        assert len(result["labels"])==len(content)
        assert [len(model.tokenize(x)) for x in content] == count
        report.append({"case":case,"purpose":purpose,"tokens":sum(count),"sentences":len(content),
                       "labels_returned":len(result["labels"]),"cpu_seconds":elapsed})
    save_json(ROOT/"results/sentence-input-size.json",{"csv_sha256":file_hash(path),"weights_sha256":file_hash(source),
        "cases":len(rows),"sentences":sum(len(v) for v in rows.values()),"profile":"sentence-document",
        "encoder_parameters":model.parameters,"cpu_threads":2,"peak_process_mib":getrusage(RUSAGE_SELF).ru_maxrss/1024,
        "token_definition":"Lowercased whitespace words after replacement of ASCII punctuation with spaces; unknown words still occupy a token.",
        "probes":report})
    print(report,flush=True)


if __name__=="__main__":
    main()
