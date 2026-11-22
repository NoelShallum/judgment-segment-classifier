"""Write one classifier over BERT and hierarchical activations by direct block concatenation."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file,save_file
from graft_rows import data
from hier_model import CRF,EMITTER,HierModel,path_labels
from hier_trial import DONOR,OUT as HIER,features
from matched_local import vectors
from program_trial import metric,official_sample,save_json
from seven_paths import ROOT,LABELS,FOLDERS,folder_for
from span_eval import EncoderCache,MODELS,counts,sets,summarize
from span_model import merge
from weight_edit import file_hash,tensor

OUT=ROOT/'checkpoints/fusion'
FRACTIONS=(0.,.15,.30,.5,.7,.85,1.)
def write(name,source,alpha,bw,bb,hw,hb,tr):
    w=np.concatenate((alpha*bw,(1-alpha)*hw),axis=1).astype(np.float32)
    b=(alpha*bb+(1-alpha)*hb).astype(np.float32)
    t=OUT/(name+'.safetensors')
    save_file({'classifier.weight':w,'classifier.bias':b,'transition':tr},str(t))
    return {'sha256':file_hash(t),'bytes':t.stat().st_size,'bert':source,'bert_fraction':alpha,'head_shape':list(w.shape)}

def short(model,heads):
    hdict=load_file(str(HIER/'excerpt.safetensors'));hw=hdict[EMITTER+'weight'][3:];hb=hdict[EMITTER+'bias'][3:]+hdict[CRF][1,3:]+hdict[CRF][3:,2]
    folds={}
    for fold in ('select','confirm'):
        y,ww,bert,info=data(fold)
        rows,_,_=official_sample(fold,160)
        h=features(model,[r[2] for r in rows]);folds[fold]=(y,ww,bert,h,info)
    y,ww,bert,h,info=folds['select']
    hh=h@hw.T+hb;candidates=[];best=None
    for name,(bw,bb) in heads.items():
        z=bert[name]@bw.T+bb
        for a in FRACTIONS:
            m=metric(y,(a*z+(1-a)*hh).argmax(1),ww);score=(m['accuracy'],m['macro_f1'])
            candidates.append([name,a,*score])
            if best is None or score>best[:2]:best=(*score,name,a)
    _,_,name,a=best;bw,bb=heads[name]
    rec=write('excerpt',name,a,bw,bb,hw,hb,np.zeros((10,10),np.float32));rec.update(candidates=candidates,groups={})
    for fold,(y,ww,bert,h,info) in folds.items():
        bz=bert[name]@bw.T+bb;hz=h@hw.T+hb
        rec['groups'][fold]={'sampling':info,'bert':metric(y,bz.argmax(1),ww),'hier':metric(y,hz.argmax(1),ww),'combined':metric(y,(a*bz+(1-a)*hz).argmax(1),ww)}
    for group in FOLDERS:
        folder=folder_for(group);rows=json.loads((folder/'approved.json').read_text());texts=[r['text'] for r in rows]
        bx=vectors(MODELS[name]['source'],texts)
        hx=features(model,texts)
        w=load_file(str(OUT/'excerpt.safetensors'));pred=(np.column_stack((bx,hx))@w['classifier.weight'].T+w['classifier.bias']).argmax(1)
        save_json(folder/'fusion.json',{'combined_excerpt':{'sha256':rec['sha256'],'predictions':{r['id']:LABELS[j] for r,j in zip(rows,pred)}}})
    return rec

def whole(model,heads):
    groups=sets();all_docs=sum(groups.values(),[]);h={};bs={}
    for d in all_docs:
        snippets=[d['text'][i:j].strip() for i,j in d['units']]
        h[d['id']]=model.logits(features(model,snippets,False))
    for name,(w,b) in heads.items():
        enc=EncoderCache(MODELS[name]['source'],'intel');by={}
        for d in all_docs:
            snippets=[d['text'][i:j].strip() for i,j in d['units']]
            by[d['id']]=enc.encode(snippets)@w.T+b
        enc.close();bs[name]=by
    tt=model.transitions
    def transition(g,edge):
        t=tt.copy();t[3:,3:]*=g;t[1,3:]*=edge;t[3:,2]*=edge
        return t
    def score(group,name,a,g,edge,detail=False):
        tr=transition(g,edge);per=[];predictions=[]
        for d in groups[group]:
            k=d['id'];ll=a*bs[name][k]+(1-a)*h[k];y=path_labels(ll,tr)
            per.append(counts(d,d['units'],y))
            if detail:predictions.append({'doc':k,'unit_labels':[LABELS[i] for i in y],'spans':merge(d['units'],y,LABELS)})
        return summarize(np.sum(per,0)),predictions
    best=None;grid=[]
    for name,a,g,edge in itertools.product(heads,FRACTIONS,(0.,.25,.5,1.),(0.,1.)):
        m,_=score('select',name,a,g,edge);rank=(m['exact_labeled_spans']['f1'],m['token_accuracy'])
        grid.append([name,a,g,edge,*rank])
        if best is None or rank>best[:2]:best=(*rank,name,a,g,edge)
    _,_,name,a,g,edge=best
    bw,bb=heads[name];rec=write('document',name,a,bw,bb,model.weight[3:],model.bias[3:],transition(g,edge))
    rec.update(transition_multiplier=g,boundary_multiplier=edge,candidates=grid,groups={})
    for group,docs in groups.items():
        m,preds=score(group,name,a,g,edge,True);rec['groups'][group]={'case_ids':[d['id'] for d in docs],'combined':m}
        if group=='cloud':save_json(ROOT/'seven-spans/fusion-cloud.json',preds)
    return rec
def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True);m=HierModel(DONOR)
    heads={k:(tensor(p['head'],'classifier.weight'),tensor(p['head'],'classifier.bias')) for k,p in MODELS.items()}
    report={'method':'Direct block concatenation of two independently public encoders’ seven-role emission heads. An analytical coefficient fixes each block; validation chooses among fixed fractions. The document variant replaces all role-to-role and edge transitions with scaled donor CRF weights. There are no gradient updates, fitted probe or teacher labels. This family was proposed after earlier aggregate results.','parameters_of_frozen_backbones_including_original_heads':m.parameters+109487623,'labels':LABELS}
    report['excerpt']=short(m,heads);save_json(ROOT/'results/fusion-excerpt.json',report)
    print('Combined excerpt',report['excerpt']['bert'],report['excerpt']['bert_fraction'],report['excerpt']['groups']['confirm']['combined']['accuracy'],flush=True)
    report['document']=whole(m,heads);save_json(ROOT/'results/fusion-validation.json',report)
    print('Combined document',report['document']['bert'],report['document']['bert_fraction'],report['document']['groups']['test']['combined']['exact_labeled_spans']['f1'],flush=True)
if __name__=='__main__':main()
