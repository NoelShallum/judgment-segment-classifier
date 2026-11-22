"""Freeze train-only, case-separated fit, edit-fit, selection and confirmation views."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGAL = ROOT.parents[1]
SPLIT = LEGAL / "references/official"
LABELS = ["None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision"]
SALT = "ultrabert-legal-corpus-folds-2026-11-22-v1"


def hash_int(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def save(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    temp.replace(path)


def docs(name):
    with (SPLIT / name).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for cid, group in itertools.groupby(enumerate(reader, start=2), lambda row: row[1]["Index"]):
            rows = []
            offset = 0
            for row_number, row in group:
                text = row["Text"]
                if not text.strip():
                    continue
                rows.append({"row": row_number, "ix":len(rows), "start":offset, "end":offset+len(text),
                             "text":text, "y":LABELS.index(row["Label"])})
                offset += len(text) + 1
            yield cid, rows


def main():
    candidates = {split:{label:[] for label in range(7)} for split in ("fit","edit")}
    natural = {split:{"short":[],"mixed":[]} for split in ("select","confirm")}
    for cid, rows in docs("train.csv"):
        if not rows:
            continue
        whole = "\n".join(" ".join(r["text"].casefold().split()) for r in rows)
        digest = hashlib.sha256(whole.encode()).hexdigest()
        fold = hash_int(SALT+digest) % 100
        split = "fit" if fold < 60 else "edit" if fold < 76 else "select" if fold < 88 else "confirm"
        if split in candidates:
            for label, group in candidates[split].items():
                offset = hash_int(SALT+digest+str(label))
                indices = [i for i,row in enumerate(rows) if row["y"]==label]
                if not indices or (len(group)>=48 and offset>=group[-1][0]):
                    continue
                center = indices[hash_int(digest+SALT+str(label)) % len(indices)]
                snippet = rows[max(center-2,0):center+3]
                block = {"id":cid,"group_sha256":digest,"ncase":len(rows),"center":center,
                         "anchor_role":label,"rows":snippet}
                group.append((offset,block))
                group.sort(key=lambda item:item[0])
                del group[48:]
        else:
            key = hash_int(SALT+"natural"+digest)
            doc = {"id":cid,"group_sha256":digest,"ncase":len(rows),"rows":rows}
            for tag in ["mixed", *(["short"] if 20<=len(rows)<=65 else [])]:
                group=natural[split][tag]
                if len(group)>=24 and key>=group[-1][0]:
                    continue
                group.append((key,doc)); group.sort(key=lambda item:item[0]); del group[24:]
    pools = {s:{str(k):[x[1] for x in v] for k,v in roles.items()} for s,roles in candidates.items()}
    pools.update({s:{k:[x[1] for x in v] for k,v in views.items()} for s,views in natural.items()})
    p = {"source":"L-NLProc/LegalSeg_CSV train.csv",
         "sha256":"60028fc085e3cb018e8d49a8879ba796dd03a0838367891f3312f9e576921794",
         "roles":LABELS,"salt":SALT,"fold_basis":"casefolded whitespace-normalized whole-document SHA-256",
         "fit_edit":"separate case hashes, then anchors grouped by human role for fit only",
         "select_confirm":"separate case hashes then label-blind choice among short or all cases",
         "offsets":"case text reconstructed with newline between nonblank CSV rows; row numbers count header as line one",
         "pools":pools}
    save(ROOT/"pools.json",p)
    print("pools", len(json.dumps(p)),[(s,len(pools[s])) for s in pools],flush=True)
    for s in natural: print(s,[(x["id"],x["ncase"]) for x in pools[s]["short"][:6]],flush=True)


if __name__=="__main__":
    main()
