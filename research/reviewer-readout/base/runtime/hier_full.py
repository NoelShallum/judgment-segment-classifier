"""Score all public test judgments with frozen hierarchy profiles using exact case identities."""
from collections import defaultdict
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
from hier_model import HierModel,PUNCTUATION,EMITTER,CRF,path_labels
from program_trial import NAMES,TO_ID,save_json
from span_eval import document,counts,summarize
from weight_edit import ROOT,file_hash
from span_model import units
from seven_paths import LABELS

DONOR=ROOT/'references/hier'
PROFILE=ROOT/'checkpoints/hierarchical'
CACHE=ROOT/'cache/hier-zero-all-test-v1'

def gold_cases():
    cases=defaultdict(list)
    with (ROOT/'references/official/test.csv').open(encoding='utf-8-sig',newline='') as stream:
        for i,r in enumerate(csv.DictReader(stream)):
            cases[r['Index']].append((i,r['Text'],TO_ID[r['Label']]))
    return cases

def prepare(cases,model):
    CACHE.mkdir(exist_ok=True)
    for i,(case,rows) in enumerate(cases.items(),1):
        p=CACHE/(case+'.npz')
        if p.exists():continue
        d=document(case,rows);sentences=[r[1].strip() for r in rows]
        chunks=[d['text'][a:b].strip() for a,b in d['units']]
        unique=list(dict.fromkeys(sentences+chunks));index={s:j for j,s in enumerate(unique)}
        allwords=model.words(unique)
        words=allwords[[index[s] for s in sentences]]
        base=model.contextualize(words,True)
        ctx=model.contextualize(words,False)
        subset=[j for j,(_,s,_) in enumerate(rows) if '\t' not in s and s.lower().translate(PUNCTUATION).split()]
        filtered=ctx if len(subset)==len(rows) else model.contextualize(words[subset],False)
        raw=model.contextualize(allwords[[index[s] for s in chunks]],False)
        tmp=p.with_suffix('.part')
        with tmp.open('wb') as stream:np.savez(stream,excerpt=base,context=model.logits(ctx),filtered=model.logits(filtered),raw=model.logits(raw),subset=subset)
        tmp.replace(p)
        if i%30==0 or i==len(cases):print('Forwarded full case',i,'/',len(cases),flush=True)

def cm(y,p):
    return np.bincount(7*np.asarray(y)+np.asarray(p),minlength=49).reshape(7,7)

def row_summary(m):
    hits=m.diagonal();den=m.sum(0)+m.sum(1);f1=np.divide(2*hits,den,out=np.zeros(7),where=den>0)
    return {'rows':int(m.sum()),'correct':int(hits.sum()),'accuracy':float(hits.sum()/m.sum()),'macro_f1':float(f1.mean()),
            'per_role_f1':dict(zip(LABELS,f1.tolist())),'confusion':m.tolist()}

def score(cases,model):
    short=load_file(str(PROFILE/'excerpt.safetensors'));doc=load_file(str(PROFILE/'document.safetensors'))
    rowscores=defaultdict(lambda:np.zeros((7,7),int));spans={};same=total=0
    pub={}
    for s in (ROOT/'references/baseline_document/predictions.txt').read_text().splitlines():
        k,seq=s.split('\t',1);pub[str(int(k.removeprefix('file_'))-1)]=[TO_ID[l] for l in seq.rstrip(',').split(',')]
    for k,rows in cases.items():
        d=document(k,rows)
        y=np.array([r[2] for r in rows]);nonblank=np.array([bool(r[1].strip()) for r in rows]);z=np.load(CACHE/(k+'.npz'))
        for tag,t in [('parent',None),('programmed',short)]:
            tr=model.transitions if t is None else t[CRF]
            log=model.logits(z['excerpt']) if t is None else z['excerpt']@t[EMITTER+'weight'][3:].T+t[EMITTER+'bias'][3:]
            pred=(log+tr[1,3:]+tr[3:,2]).argmax(1)
            rowscores['excerpt_'+tag]+=cm(y[nonblank],pred[nonblank])
            rowscores['all_rows_with_blank_as_unknown_'+tag]+=cm(y,pred)
        for tag,tr in [('parent',model.transitions),('programmed',doc[CRF])]:
            pred=path_labels(z['context'],tr);rowscores['known_rows_'+tag]+=cm(y,pred)
            pred=path_labels(z['filtered'],tr);subset=z['subset']
            assert len(pred)==len(pub[k])
            rowscores['author_filtered_'+tag]+=cm(y[subset],pred)
            if tag=='parent':same+=np.sum(pred==pub[k]);total+=len(pred)
            pred=path_labels(z['raw'],tr);v=counts(d,d['units'],pred)
            spans[tag]=spans.get(tag,np.zeros_like(v))+v
    report={'cases':len(cases),'corpus':'Official LegalSeg test split. Rows with no textual content are excluded by the excerpt product. The 149,881-row diagnostic feeds an unknown token to a blank input.',
            'model_sha256':file_hash(DONOR/'hier.safetensors'),'excerpt_sha256':file_hash(PROFILE/'excerpt.safetensors'),'document_sha256':file_hash(PROFILE/'document.safetensors'),
            'row_classification':{k:row_summary(v) for k,v in rowscores.items()},'raw_full_case_segmentation':{k:summarize(v) for k,v in spans.items()},
            'named_release_alignment':{'same_parent_outputs':int(same),'author_filtered_rows':total,'fraction':same/total,
             'protocol':'Public file IDs minus one align every judgment. For this particular author comparison only, remove tab-containing rows and rows empty after stripping ASCII punctuation, then recompute all document activations.'}}
    save_json(ROOT/'results/hierarchical-official-test.json',report)
    print('Full hierarchy',[(k,v['accuracy'],v['macro_f1']) for k,v in report['row_classification'].items()],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);model=HierModel(DONOR);cases=gold_cases();prepare(cases,model)
    if not args.prepare_only:score(cases,model)
