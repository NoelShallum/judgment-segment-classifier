"""Recalculate official metrics from row-level predictions and the public LegalSeg CSV."""
import argparse,csv,hashlib,io,json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import numpy as np

H=Path(__file__).resolve().parent
WHOLE=H.parent
NAMES=['None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision']
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def cm(y,p):
    return np.bincount(np.asarray(y,np.int64)*7+np.asarray(p,np.int64),minlength=49).reshape(7,7)
def stats(c):
    f=2*c.diagonal()/np.maximum(c.sum(0)+c.sum(1),1)
    return dict(rows=int(c.sum()),macro_f1=float(f.mean()),accuracy=float(c.trace()/c.sum()),correct=int(c.trace()),by_role_f1=dict(zip(NAMES,f.tolist())),none_to_reasoning=int(c[0,5]),reasoning_to_none=int(c[5,0]),pair_errors=int(c[0,5]+c[5,0]),confusion=c.tolist())
def intervals(gold,pred,base,off):
    def docs(q):return np.stack([cm(gold[a:b],q[a:b]) for a,b in zip(off,off[1:])])
    aa,bb=docs(pred),docs(base);rng=np.random.default_rng(24611057);rows=[]
    for _ in range(40):
        ix=rng.integers(0,len(aa),(100,len(aa)));a,b=aa[ix].sum(1),bb[ix].sum(1)
        def m(c):
            hit=np.diagonal(c,axis1=1,axis2=2);f=(2*hit/np.maximum(c.sum(1)+c.sum(2),1)).mean(1)
            return np.column_stack((f,hit.sum(1)/c.sum((1,2)),c[:,0,5]+c[:,5,0],c[:,0,5],c[:,5,0]))
        rows.append(m(a)-m(b))
    ci=np.quantile(np.concatenate(rows),[.025,.975],axis=0)
    return {k:ci[:,i].tolist() for i,k in enumerate(['macro_delta','accuracy_delta','pair_delta','none_to_reasoning_delta','reasoning_to_none_delta'])}
def arrays(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def main():
    cli=argparse.ArgumentParser();cli.add_argument('--csv',type=Path,required=True);cli.add_argument('--prediction-file',type=Path,required=True);cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    z=arrays(a.prediction_file);keys=z['case_ids'].astype(str);off=z['offsets'];y=z['gold'].astype(np.int64);g=defaultdict(list)
    with a.csv.open(encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f)):g[r['Index']].append((i,NAMES.index(r['Label']),bool(r['Text'].strip())))
    assert set(keys)==set(g) and len(set(keys))==len(keys)==712
    allgold=[];blank=[]
    for j,k in enumerate(keys):
        current=g[k];ys=np.array([u[1] for u in current if u[2]],int);np.testing.assert_array_equal(y[off[j]:off[j+1]],ys)
        allgold.extend(u[1] for u in current);blank.extend(not u[2] for u in current)
    blank=np.array(blank);full_y=np.array(allgold);np.testing.assert_array_equal(full_y[~blank],y)
    assert len(y)==149868 and blank.sum()==13 and len(full_y)==149881
    models={};base=z['original']
    for name,p in z.items():
        if name in ('gold','offsets','case_ids'):continue
        assert len(p)==len(y) and np.issubdtype(p.dtype,np.integer) and p.min()>=0 and p.max()<7
        c=cm(y,p);oracle=c.copy()
        for r,k in ((0,5),(5,0)):oracle[r,r]+=oracle[r,k];oracle[r,k]=0
        complete=np.zeros(len(blank),np.int8);complete[~blank]=p
        models[name]={'nonblank':stats(c),'all_csv_rows':stats(cm(full_y,complete)),'pair_only_oracle':stats(oracle),
                      'paired_against_original':intervals(y,p,base,off),'changed':int((p!=base).sum()),'fixed':int(((p==y)&(base!=y)).sum()),'harmed':int(((p!=y)&(base==y)).sum())}
    out=dict(test_csv_sha256=digest(a.csv),predictions_sha256=digest(a.prediction_file),blank_count=int(blank.sum()),blank_gold=np.bincount(full_y[blank],minlength=7).tolist(),blank_fallback='None, the modal blank label on official train; skipped during sentence-context assembly.',cases=len(keys),models=models,bootstrap='4,000 paired resamples of whole judgments, pooled seven-class macro-F1 on original nonblank rows. Intervals are conditional on this historical benchmark.')
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    for k,v in models.items():print(k,round(100*v['nonblank']['macro_f1'],4),round(100*v['all_csv_rows']['macro_f1'],4),v['nonblank']['pair_errors'])
if __name__=='__main__':main()
