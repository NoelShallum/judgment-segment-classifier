"""Score a sentence or an ordered list of sentences from a compiled native-embedding readout."""
import argparse
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

ROLES=["None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision"]

def classify(texts,table,tokenizer,document):
    enc=tokenizer.encode_batch(texts,add_special_tokens=False)
    if not texts or any(not t.ids or not s.strip() for s,t in zip(texts,enc)):raise ValueError("Every sentence must contain text.")
    key="structure" if document else "text"
    matrix=table[key+".token_to_class"]
    ids=[np.unique(t.ids) for t in enc]
    emissions=np.stack([matrix[r].astype(float).sum(0)/np.sqrt(len(r)) for r in ids])
    if not document:return emissions.argmax(1)
    n=len(texts);b=np.minimum(23,24*np.arange(n)//n)
    emissions+=table["structure.position_gain"][0]*table["position.logp"][b]
    trans=table["structure.transition_gain"][0]*table["transition.logp"]
    scores=emissions[0].copy();parents=[]
    for row in emissions[1:]:
        options=scores[:,None]+trans;parents.append(options.argmax(0));scores=options.max(0)+row
    cur=int(scores.argmax());out=[cur]
    for state in reversed(parents):cur=int(state[cur]);out.append(cur)
    return np.asarray(out[::-1])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--sentence");group.add_argument("--document-json",type=Path)
    a=p.parse_args()
    docs=json.loads(a.document_json.read_text()) if a.document_json else [a.sentence]
    if not isinstance(docs,list) or not all(isinstance(row,str) for row in docs):raise ValueError("Document JSON must contain an ordered array of sentence strings.")
    pred=classify(docs,load_file(str(a.weights)),Tokenizer.from_file(str(a.tokenizer)),a.document_json is not None)
    print(json.dumps([{"index":i,"role":ROLES[int(v)]} for i,v in enumerate(pred)],ensure_ascii=False))
if __name__=="__main__":main()
