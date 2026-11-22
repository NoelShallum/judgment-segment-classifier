"""Physically edit LegalSeg's hierarchical emission and CRF tensors, using validation for selection."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import save_file
from associative_edit import correction
from hier_extract import REV,SOURCE_SHA,sha
from hier_model import CRF,EMITTER,HierModel,path_labels
from program_trial import fixture_rows,metric,official_sample,save_json
from role_fixtures import ANCHORS,CHECKS
from seven_paths import LABELS,FOLDERS,ROOT,folder_for
from span_eval import counts,sets,summarize
from span_model import merge

DONOR=ROOT/'references/hier'
OUT=ROOT/'checkpoints/hierarchical'
IDENTITY=np.eye(7,dtype=np.float32)

def features(model,texts,independent=True):
    digest=hashlib.sha256()
    for t in texts:
        b=t.encode();digest.update(len(b).to_bytes(8,'little'));digest.update(b)
    p=ROOT/'cache'/('hier-zero-words-'+digest.hexdigest()[:24]+'.npy')
    if p.exists():
        words=torch.from_numpy(np.load(p))
    else:
        words=model.words(texts)
        np.save(p,words.numpy())
    return model.contextualize(words,independent)

def rewrite(model,name,w,b,tr):
    weights=model.weight.copy();bias=model.bias.copy()
    weights[3:]=w;bias[3:]=b
    path=OUT/(name+'.safetensors')
    save_file({EMITTER+'weight':weights,EMITTER+'bias':bias,CRF:tr},str(path))
    return {'sha256':sha(path),'edited_scalars':int(np.sum(weights!=model.weight)+np.sum(bias!=model.bias)+np.sum(tr!=model.transitions)),
            'stored_scalars':int(weights.size+bias.size+tr.size),'bytes':path.stat().st_size}

def short(model):
    synth=fixture_rows(ANCHORS)+fixture_rows(CHECKS)
    keys=features(model,[r[2] for r in synth]);desired=np.array([r[3] for r in synth])
    parent_w,parent_b=model.weight[3:],model.bias[3:]
    heads=[(0,0.,parent_w,parent_b)]
    for margin,damping in itertools.product((1.5,3.,5.),(.1,.5)):
        w,b=correction(keys,desired,parent_w,parent_b,margin,damping)
        heads.append((margin,damping,w,b))
    split={}
    for name in ('select','confirm'):
        rows,sw,info=official_sample(name,160)
        split[name]=(features(model,[r[2] for r in rows]),np.array([r[3] for r in rows]),sw,info)
    x,y,sw,_=split['select'];candidates=[];best=None
    edge=model.transitions[1,3:]+model.transitions[3:,2]
    for margin,damping,w,b in heads:
        for scale in (1.,.5,.25,0.):
            m=metric(y,(x@w.T+b+scale*edge).argmax(1),sw)
            candidates.append([margin,damping,scale,m['accuracy'],m['macro_f1']])
            if m['macro_f1']+1e-12>=candidates[0][4]:
                score=m['accuracy'],m['macro_f1']
                if best is None or score>best[:2]:best=(*score,margin,damping,scale,w,b)
    _,_,margin,damping,scale,w,b=best
    tr=model.transitions.copy();tr[1,3:]*=scale;tr[3:,2]*=scale
    rec=rewrite(model,'excerpt',w,b,tr)
    rec.update(margin=margin,damping=damping,boundary_multiplier=scale,candidates=candidates,chosen_on='Maximize weighted validation accuracy, requiring at least the unchanged donor macro-F1 on the selection split.')
    rec['groups']={}
    for name,(xx,yy,ww,info) in split.items():
        rec['groups'][name]={'sampling':info,'parent':metric(yy,(xx@parent_w.T+parent_b+edge).argmax(1),ww),
                             'programmed':metric(yy,(xx@w.T+b+scale*edge).argmax(1),ww)}
    for g in FOLDERS:
        f=folder_for(g);rows=json.loads((f/'approved.json').read_text());x=features(model,[r['text'] for r in rows])
        rr={}
        for name,pred,h in [('hier_parent',(x@parent_w.T+parent_b+edge).argmax(1),SOURCE_SHA),
                            ('hier_edited',(x@w.T+b+scale*edge).argmax(1),rec['sha256'])]:
            rr[name]={'sha256':h,'predictions':{r['id']:LABELS[j] for r,j in zip(rows,pred)}}
        save_json(f/'hier.json',rr)
    return rec

def whole(model):
    groups=sets();raw={};oracle={}
    for docs in groups.values():
        for d in docs:
            for db,field in ((raw,'units'),(oracle,'reference_units')):
                snippets=[d['text'][a:b].strip() for a,b in d[field]]
                db[d['id']]=model.logits(features(model,snippets,False))
    base=model.transitions
    def altered(inner,edge,sticky):
        t=base.copy();t[1,3:]*=edge;t[3:,2]*=edge;t[3:,3:]=inner*base[3:,3:]+sticky*IDENTITY
        return t
    def score(name,t,db=raw,field='units'):
        per=[counts(d,d[field],path_labels(db[d['id']],t)) for d in groups[name]]
        return summarize(np.sum(per,0)),np.stack(per)
    candidates=[];best=None
    for inner,edge,sticky in itertools.product((1.,.5,0.,1.5), (1.,.5,0.),(0.,.75,1.5)):
        t=altered(inner,edge,sticky);res,_=score('select',t)
        key=res['exact_labeled_spans']['f1'],res['token_accuracy']
        candidates.append([inner,edge,sticky,*key])
        if best is None or key>best[:2]:best=(*key,inner,edge,sticky)
    _,_,inner,edge,sticky=best;t=altered(inner,edge,sticky)
    rec=rewrite(model,'document',model.weight[3:],model.bias[3:],t)
    rec.update(inner_multiplier=inner,edge_multiplier=edge,diagonal_addition=sticky,candidates=candidates,
               chosen_on='Maximize exact labeled-span F1 on ten complete validation judgments, with token accuracy as tiebreaker.',groups={})
    for name,docs in groups.items():
        m,mm=score(name,t);p,pp=score(name,base);o,_=score(name,t,oracle,'reference_units')
        rng=np.random.default_rng(84071);draw=rng.integers(len(docs),size=(2000,len(docs)));deltas=[]
        for ix in draw:
            a=summarize(mm[ix].sum(0));b=summarize(pp[ix].sum(0))
            deltas.append([a['token_accuracy']-b['token_accuracy'],a['exact_labeled_spans']['f1']-b['exact_labeled_spans']['f1']])
        rec['groups'][name]={'cases':len(docs),'rows':sum(len(d['reference_roles']) for d in docs),'case_ids':[d['id'] for d in docs],
                             'units':sum(len(d['units']) for d in docs),'programmed':m,'parent':p,'programmed_known_cuts':o,
                             'parent_gap_ci_token_accuracy_exact_f1':np.quantile(deltas,[.025,.975],axis=0).T.tolist()}
    rows=[]
    for d in groups['cloud']:
        y=path_labels(raw[d['id']],t)
        rows.append({'doc':d['id'],'unit_labels':[LABELS[i] for i in y],'spans':merge(d['units'],y,LABELS)})
    save_json(ROOT/'seven-spans/hier-cloud.json',rows)
    return rec

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    model=HierModel(DONOR)
    result={'revision':REV,'donor_sha256':SOURCE_SHA,'vocabulary_sha256':sha(DONOR/'word2idx.json'),'parameters':model.parameters,
            'model_bytes':(DONOR/'hier.safetensors').stat().st_size,'vocabulary_bytes':(DONOR/'word2idx.json').stat().st_size,'labels':LABELS,
            'implementation':'The same public word embeddings, both bidirectional LSTMs, emission matrix and CRF. All deployed LSTM states initialize to zero. The original implementation samples random hidden and cell states per forward pass. The unchanged comparator also uses zero states. Punctuation-only units get an unknown word to preserve document offsets.',
            'selection':'No gradient steps. Public seven-role development cases choose each profile before the four paired samples are scored. The hierarchy family was introduced after aggregate results for earlier families were available.'}
    result['excerpt']=short(model)
    save_json(ROOT/'results/hierarchical-excerpt.json',{k:v for k,v in result.items()})
    print('Hierarchical excerpt frozen',result['excerpt']['boundary_multiplier'],result['excerpt']['groups']['confirm']['programmed']['accuracy'],flush=True)
    result['document']=whole(model)
    save_json(ROOT/'results/hierarchical-validation.json',result)
    m=result['document']['groups']['test']['programmed']
    print('Hierarchical document frozen',m['token_accuracy'],m['exact_labeled_spans']['f1'],flush=True)
if __name__=='__main__':main()
