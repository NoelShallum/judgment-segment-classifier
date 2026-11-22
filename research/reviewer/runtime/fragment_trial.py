"""Select direct recurrent/standalone readout and CRF writes for text clipped from a judgment."""
from collections import defaultdict
import csv,itertools,json,random
import numpy as np
import torch
from safetensors.numpy import save_file,load_file
from hier_model import HierModel,EMITTER,CRF,path_labels
from hier_trial import DONOR,OUT
from program_trial import fold,TO_ID,save_json
from seven_paths import ROOT,LABELS
from span_eval import document,sets,counts,summarize
from span_model import merge
from weight_edit import file_hash
def passages():
    previous=sets();groups={}
    skip={d['id'] for docs in previous.values() for d in docs}
    for name in ('gemini-seven-pilot','gemini-confirmation','gemini-seven-paragraphs','gemini-seven-joined'):
        skip.update(r['doc'] for r in json.loads((ROOT/name/'approved.json').read_text()))
    for kind,src,n,seed in [('select','val',84,89104),('confirm','val',84,89105),('test','test',120,89106)]:
        by=defaultdict(list)
        with (ROOT/'references/official'/(src+'.csv')).open(encoding='utf-8-sig',newline='') as f:
            for i,r in enumerate(csv.DictReader(f)):
                case=r['Index']
                if case in skip or (src=='val' and (fold(case)==0)!=(kind=='confirm')):continue
                by[case].append((i,r['Text'],TO_ID[r['Label']]))
        rng=random.Random(seed);ids=rng.sample(sorted(k for k,rows in by.items() if len(rows)>=25),n);items=[]
        for k in ids:
            number=rng.randint(4,9);first=rng.randint(1,len(by[k])-number-1)
            rows=by[k][first:first+number];d=document(k,rows);d['row_ids']=[r[0] for r in rows]
            items.append(d)
        groups[kind]=items
    groups['cloud']=previous['cloud']
    return groups
def main():
    torch.set_num_threads(1);m=HierModel(DONOR);groups=passages()
    short=load_file(str(OUT/'excerpt.safetensors'));contexts={};isolated={}
    for docs in groups.values():
        for d in docs:
            words=m.words([d['text'][a:b].strip() for a,b in d['units']])
            contexts[d['id']]=m.logits(m.contextualize(words))
            isolated[d['id']]=m.contextualize(words,True)@short[EMITTER+'weight'][3:].T+short[EMITTER+'bias'][3:]
    def altered(inner,edge):
        t=m.transitions.copy();t[3:,3:]*=inner;t[1,3:]*=edge;t[3:,2]*=edge
        return t
    def score(group,a,inner,edge):
        c=[];preds=[];tr=altered(inner,edge)
        for d in groups[group]:
            id=d['id'];y=path_labels(a*contexts[id]+(1-a)*isolated[id],tr)
            c.append(counts(d,d['units'],y))
            preds.append({'doc':id,'unit_labels':[LABELS[i] for i in y],'spans':merge(d['units'],y,LABELS)})
        return summarize(np.sum(c,0)),np.stack(c),preds
    grid=[];best=None
    for a,inner,edge in itertools.product((1.,.75,.5,.25,0.),(1.,.5,.25,0.),(1.,.25,0.)):
        r,_,_=score('select',a,inner,edge);rank=(r['exact_labeled_spans']['f1'],r['token_accuracy'])
        grid.append([a,inner,edge,*rank])
        if best is None or rank>best[:2]:best=(*rank,a,inner,edge)
    _,_,a,inner,edge=best
    path=OUT/'fragment.safetensors';save_file({'classifier.weight':np.column_stack((a*m.weight[3:],(1-a)*short[EMITTER+'weight'][3:])).astype('f4'),
                'classifier.bias':(a*m.bias[3:]+(1-a)*short[EMITTER+'bias'][3:]).astype('f4'),'transition':altered(inner,edge)},str(path))
    res={'sha256':file_hash(path),'context_fraction':a,'internal_crf_multiplier':inner,'edge_crf_multiplier':edge,'candidates':grid,
         'protocol':'Select on 84 interior passages of 4–9 adjacent official rows from distinct development judgments. Confirm on 84 separate validation cases. Evaluate on 120 new cases from the original test set without role quotas or model-based filtering. All 288 are disjoint from earlier full-case, sentence and four matched cohorts and the 12 cloud passages. This follow-up was suggested by the earlier short-passage errors. Both ends of every sampled passage lie inside its source case.','groups':{}}
    for group,docs in groups.items():
        r,rr,preds=score(group,a,inner,edge);parent,pp,_=score(group,1,1,1)
        rng=np.random.default_rng(4981);gaps=[]
        for ix in rng.integers(len(rr),size=(1000,len(rr))):
            q=summarize(rr[ix].sum(0));b=summarize(pp[ix].sum(0));gaps.append([q['token_accuracy']-b['token_accuracy'],q['exact_labeled_spans']['f1']-b['exact_labeled_spans']['f1']])
        res['groups'][group]={'cases':len(docs),'rows':sum(len(d['reference_roles']) for d in docs),'multi_role_cases':sum(len(d['gold_spans'])>1 for d in docs),
                              'parent':parent,'programmed':r,'gap_ci_token_accuracy_exact_f1':np.quantile(gaps,[.025,.975],axis=0).T.tolist(),
                              'case_rows':[[d['id'],d.get('row_ids',[])] for d in docs]}
        if group=='cloud':save_json(ROOT/'seven-spans/fragment-cloud.json',preds)
    save_json(ROOT/'results/fragment-validation.json',res)
    print('Interior program',a,inner,edge,res['groups']['test']['programmed']['exact_labeled_spans']['f1'],flush=True)
if __name__=='__main__':main()
