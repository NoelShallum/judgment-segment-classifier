"""A deterministic standalone classifier must give byte-identical inputs the same label."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
out={}
records=json.loads((root/"references/official/verified-local.json").read_text())
for split in ("test","val"):
    path=root/"references"/"official"/(split+".csv")
    groups=defaultdict(Counter);n=0;blanks=0
    with path.open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            t=row["Text"] or ""
            if not t.strip():blanks+=1;continue
            groups[t][row["Label"]]+=1;n+=1
    conflicts=[(t,v) for t,v in groups.items() if len(v)>1]
    losses=sum(sum(g.values())-max(g.values()) for _,g in conflicts)
    # Expose hashes and frequencies for reproducibility without republishing legal text.
    examples=[{"sha256":hashlib.sha256(t.encode()).hexdigest(),"gold_counts":dict(g)} for t,g in
            sorted(conflicts,key=lambda kv:(-sum(kv[1].values()),kv[0]))[:12]]
    out[split]={"nonblank_rows":n,"blank_rows":blanks,"unique_raw_inputs":len(groups),
        "raw_inputs_with_incompatible_labels":len(conflicts),"affected_rows":sum(sum(g.values()) for _,g in conflicts),
        "minimum_errors_on_nonblank_rows":losses,"empirical_maximum_accuracy":(n-losses)/n,
        "examples":examples}
out["interpretation"]="For identical nonblank text bytes and a single deterministic output, at least total minus the most frequent label in each conflicting group must be wrong. This upper bound applies only to these exact recorded splits and says nothing about the attainability of the remaining accuracy."
p=root/"results/identical-inputs-ceiling.json";tmp=p.with_suffix(".part");tmp.write_text(json.dumps(out,indent=2));tmp.replace(p)
print([(k,v["minimum_errors_on_nonblank_rows"],v["empirical_maximum_accuracy"]) for k,v in out.items() if k!="interpretation"])
