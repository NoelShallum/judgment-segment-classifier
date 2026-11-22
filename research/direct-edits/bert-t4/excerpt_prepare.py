"""Extract the exact deployed route for four already-scored Gemini excerpt panels."""
from functools import lru_cache
import hashlib
import json

import numpy as np

from prepare import HERE, ROOT, BertTokenizerFast, pair, route_capture, sequence


PANELS = (
    ("balanced isolated sentences", "gemini-seven-pilot"),
    ("independent singleton case sample", "gemini-confirmation"),
    ("single human-labeled multi-statement rows", "gemini-seven-paragraphs"),
    ("joined same-role paragraphs", "gemini-seven-joined"),
)


def main():
    sequence.phrase.load_predictor = lru_cache(maxsize=1)(sequence.phrase.load_predictor)
    parent = pair.predictor("direct")
    tok = BertTokenizerFast.from_pretrained(ROOT/"references/tokenizer",local_files_only=True)
    received = json.loads((HERE.parent/"gemini/matched-excerpt-ledger.json").read_text())
    bank = {}
    by_panel = {}
    for name,folder in PANELS:
        approved = json.loads((ROOT/folder/"approved.json").read_text())
        refs = {r["id"]:r for r in received[name]}
        positions=[]
        for r in approved:
            digest=hashlib.sha256(r["text"].encode()).hexdigest()
            assert digest==refs[r["id"]]["input_sha256"]==r["sha256"]
            if digest not in bank:
                features=route_capture([r["text"]],tok,parent)
                assert features["positions"].tolist()==[0]
                bank[digest]=features
            positions.append({**refs[r["id"]],"row_sha256":digest})
        by_panel[name]=positions
    values=list(bank.values())
    np.savez_compressed(HERE/"excerpt-route.npz",doc_sha=list(bank),**{name:np.concatenate([row[name] for row in values]) for name in values[0] if name!="positions"})
    (HERE/"excerpt-ledger.json").write_text(json.dumps(by_panel,indent=2)+"\n")
    print("Escrowed excerpt route",len(bank),"unique uploads; panel sizes",{key:len(x) for key,x in by_panel.items()},flush=True)


if __name__=="__main__":
    main()
