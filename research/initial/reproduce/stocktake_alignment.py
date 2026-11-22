"""Summarize existing model outputs on identical inputs without running either model."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from seven_paths import ROOT,folder_for,LABELS
NAMES=('None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision')
def summarize(cm):
    hit=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-2)+cm.sum(-1)
    f=np.divide(2*hit,den,out=np.zeros_like(hit,dtype=float),where=den>0)
    return hit.sum(-1)/cm.sum((-2,-1)),f.mean(-1)
def scores(official,stored,local,keys):
    cases=sorted(set(r['doc'] for r in keys));idx={v:i for i,v in enumerate(cases)}
    cms={};details={}
    for name in ['archived_standalone',*local]:
        cube=np.zeros((len(cases),7,7),dtype=int)
        for r in keys:
            j=int(r['id'].split('-')[-1]);actual=NAMES.index(official[j]['Label'])
            assert actual==LABELS.index(r['gold'])==stored[j][0]
            pred=stored[j][1] if name=='archived_standalone' else LABELS.index(local[name][r['id']])
            cube[idx[r['doc']],actual,pred]+=1
        cms[name]=cube
        a,f=summarize(cube.sum(0));details[name]={'correct':int(cube.sum(0).trace()),'accuracy':float(a),'macro_f1':float(f)}
    rng=np.random.default_rng(7413);draw=rng.multinomial(len(cases),np.ones(len(cases))/len(cases),size=4000)
    ref=cms['archived_standalone'];a0,f0=summarize((draw@ref.reshape(len(cases),49)).reshape(-1,7,7))
    for n in local:
        a1,f1=summarize((draw@cms[n].reshape(len(cases),49)).reshape(-1,7,7))
        details[n]['paired_gap_accuracy_macro_f1_ci_95']=np.quantile(np.stack([a1-a0,f1-f0]),[.025,.975],axis=1).T.tolist()
    return {'rows':len(keys),'cases':len(cases),'systems':details}
def main():
    root=ROOT/'references'
    with (root/'official/test.csv').open(encoding='utf-8-sig',newline='') as f:official=list(csv.DictReader(f))
    with (root/'baseline_sentence/InLegalBert(i)_predictions.csv').open(encoding='utf-8-sig',newline='') as f:stored=[(int(r['true']),int(r['pred'])) for r in csv.DictReader(f)]
    assert len(stored)==len(official) and all(NAMES.index(x['Label'])==y for x,(y,p) in zip(official,stored))
    result={}
    for name in ('pilot','confirmation'):
        cohort=json.loads((folder_for(name)/'approved.json').read_text())
        data=json.loads((ROOT/'results'/('gemini-seven-'+name+'.json')).read_text())['rows']
        preds={n:{r['id']:r[n] for r in data} for n in ('internal_mlp','associative_direct','hier_edited','gemini')}
        exact=[r for r in cohort if official[int(r['id'].split('-')[-1])]['Text']==r['text']]
        if not exact:raise ValueError('No exact records')
        result[name]=scores(official,stored,preds,exact)
        result[name]['eligible_cohort']=len(cohort)
    ids=[i for i,r in enumerate(official) if r['Text'].strip()]
    cm=np.bincount([7*stored[i][0]+stored[i][1] for i in ids],minlength=49).reshape(7,7);a,f=summarize(cm)
    result['all_nonblank_standalone_archive']={'rows':len(ids),'accuracy':float(a),'macro_f1':float(f),'correct':int(cm.trace())}
    result['protocol']='Offline comparison of previously frozen predictions. Match original CSV row ID, its gold label and the complete UTF-8 excerpt byte for byte. The authors’ current-sentence prediction CSV contains only labels; we verified that all 149,881 gold rows agree in sequence. Release forecasts rely on that alignment, since text hashes are absent from the prediction file. Exclude any excerpt whose leading enumerator, formatting or text changed. The paired CIs resample whole cases.'
    dest=ROOT/'results/stocktake-standalone-alignment.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    for k in ('pilot','confirmation'):
        print(k,result[k]['rows'],result[k]['cases'],[(n,v['correct'],round(v['macro_f1'],3)) for n,v in result[k]['systems'].items()])
    print('Full archive nonblank',result['all_nonblank_standalone_archive'])
if __name__=='__main__':main()
