"""Read the existing cross-fitted 144-case training reviewer without writing historical files."""
import json
import sys
from pathlib import Path
import numpy as np
from prepare import ROOT
HISTORY=ROOT.parents[1]
sys.path.insert(0,str(HISTORY))
import sentence_bert_reviewer_gain as g

def main():
    path=HISTORY/'cache/sentence-bert-referral/fit-views-v1.json'
    old=json.loads(path.read_text())
    docs=[]
    for cid,y,current,q,x,p,texts in old:
        clean=[t.strip() for t in texts]
        assert len(y)==len(clean) and all(clean)
        concat=' '.join(clean);starts=np.cumsum([0]+[len(v)+1 for v in clean[:-1]]).tolist()
        docs.append({'id':cid,'text':concat,'reference_units':list(zip(starts,starts[1:]+[len(concat)])),'reference_roles':y})
    assert (HISTORY/'cache/sentence-minilm-whitened-reviewer-gain-fit-dense.npz').is_file()
    g.B.docs_for_fit=lambda:docs
    a=g.get_features('fit',g.plans())
    bank,base,cross,loc=a[:4]
    np.savez_compressed(ROOT/'historical-incumbent-crossfit.npz',cases=bank[0],ends=bank[1],gold=bank[2],
                        predicted=cross,sequence_only=base,eligible_positions=loc)
    print(len(bank[0]),len(cross),len(loc))
if __name__=='__main__':main()
