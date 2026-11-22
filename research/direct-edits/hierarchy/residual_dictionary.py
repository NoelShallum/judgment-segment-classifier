"""Write token correction keys by a closed-form solve of public train-label residuals."""
import csv
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import softmax
from native_memory import HERE,ROOT,DONOR,HierModel,TO_ID,dump,file_hash,store_state

def write():
    k=0
    source=store_state("fit","base",240)
    with np.load(source) as f:
        keys=f["keys"].tolist(); slices=f["ends"]; states=f["states"]; gold=f["gold"]
    ids=set(keys); rows={k:[] for k in keys}
    model=HierModel(DONOR)
    with (ROOT/"references/official/train.csv").open(newline="",encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row["Index"] in ids and row["Text"].strip():
                rows[row["Index"]].append(model.tokenize(row["Text"].strip()))
    freq=np.load(HERE/"counts.npz")["counts"][0].sum(0)
    chosen=np.argsort(-freq,kind="stable")[:4608]
    chosen=chosen[chosen>=2]
    index=np.full(len(freq),-1,int);index[chosen]=np.arange(len(chosen))
    ptr=[0];allrows=[]; vals=[]
    for key,st,ed in zip(keys,slices,slices[1:]):
        assert len(rows[key])==ed-st
        for tok in rows[key]:
            t=np.array(tok[-60:])
            # Analytic linearization of the native word LSTM forget gate, 0.9309 per token.
            w=.93**np.arange(len(t)-1,-1,-1)
            inds=index[t]; valid=inds>=0
            allrows.extend(inds[valid]);vals.extend(w[valid])
            ptr.append(len(allrows))
    X=csr_matrix((vals,allrows,ptr),shape=(len(gold),len(chosen)))
    X.sum_duplicates()
    prob=softmax(model.logits(states)/2.2,axis=1)
    count=np.bincount(gold,minlength=7)
    sample=(count[gold]/count.sum())**-.35;sample/=sample.mean()
    gram=(X.T@X.multiply(sample[:,None])).toarray()
    scale=np.sqrt(np.maximum(gram.diagonal(),1))
    gram/=scale[None,:];gram/=scale[:,None]
    d=.25+.5/scale
    gram.flat[::len(gram)+1]+=d
    from scipy.linalg import cho_factor,cho_solve
    target=np.eye(7)[gold]-prob
    rhs=(X.T@(sample[:,None]*target))/scale[:,None]
    coeff=cho_solve(cho_factor(gram,check_finite=False),rhs,check_finite=False)/scale[:,None]
    values=np.zeros((len(freq),7),np.float32)
    values[chosen]=coeff
    # Scale the response into the approximately linear range of the physically programmed gates.
    values*=.55/np.percentile(np.abs(coeff),94.)
    values=np.clip(values,-2.5,2.5)
    tmp=HERE/"residual-dictionary.part"
    with tmp.open("wb") as fp:
        np.savez_compressed(fp,word_scores=values,keys=chosen)
    tmp.replace(HERE/"residual-dictionary.npz")
    dump(HERE/"residual-dictionary.json",dict(train_matrix_shape=list(X.shape),train_case_ids=keys,
           source_sha256=file_hash(source),nnz=int(X.nnz),ridge="0.25 + 0.5 / sqrt(diagonal), unit-Gram coordinates",
           gradient_steps=0,changed_word_tokens=len(chosen)))
    print("wrote residual dictionary",len(chosen),len(gold),flush=True)
if __name__=="__main__":write()
