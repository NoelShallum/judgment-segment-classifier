"""Apply frozen exact and near-text gates to the legal contextual model; quantify novel-text slices."""
import argparse,hashlib,io,json,sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from memory import H,sha,save
from score import bank,cm,metric,predict,paired
E=H.parent/'doc-emission';sys.path.insert(0,str(E))
import exact_memory as exact
def lock():
    p=H/'composition-lock.json'
    if p.exists():return
    save('composition-lock.json',dict(doc_arm='doc-semantic',near_freeze=sha(H/'freeze.json'),exact_freeze=sha(E/'exact-memory/freeze.json'),legal_development=sha(H.parent/'legal-context/legal-results.zip'),
        primary='Use the frozen doc-semantic model for every row. Use exact training vote if context width >=3, else use preselected near-text aligned vote if eligible, else use matching single-sentence exact vote. Exclude full duplicate judgments from both writes.',
        note='Composition was specified after the old reviewer exact-memory result and earlier historical baseline test audits were known. No official doc-semantic predictions informed the rule. Compare unseen text and report each model separately.'))
def getexact(d,g):
    c='official' if g=='test' else g
    if g!='select':
        with np.load(E/'exact-memory'/(c+'-predictions.npz'),allow_pickle=False) as f:
            np.testing.assert_array_equal(f['case_ids'],d.case_ids);np.testing.assert_array_equal(f['gold'],d.gold)
            return f['all-role/all-seven'],f['all-role/all-seven.context_width']
    p=exact.table()
    _,_,_,base,values,_=exact.on_cohort(p,c)
    cfg=exact.read(exact.FREEZE)['arms']['all-seven']['parameters']
    y,w,_,_=exact.propose(base['all-role'],values,'all-seven',cfg)
    return y,w
def getlegal(d,g):
    arch=H.parent/'legal-context'/('legal-official-results.zip' if g=='test' else 'legal-results.zip')
    with ZipFile(arch) as z:
        name=g+'-predictions.npz';raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==json.loads(z.read('SHA256SUMS.json'))[name]
        with np.load(io.BytesIO(raw)) as f:
            np.testing.assert_array_equal(f['case_ids'],d.case_ids);np.testing.assert_array_equal(f['gold'],d.gold);np.testing.assert_array_equal(f['incumbent'],d.incumbent)
            return f['doc-semantic'],f['doc-authors']
def run(groups):
    lock();spec=json.loads((H/'composition-lock.json').read_text());assert spec['near_freeze']==sha(H/'freeze.json') and spec['exact_freeze']==sha(E/'exact-memory/freeze.json')
    cfg=json.loads((H/'freeze.json').read_text())['arms']['all']['config'];results={}
    for g in groups:
        d=bank(g);base,auth=getlegal(d,g);near,elig=predict(d,cfg);elig&=~d.same_case
        ex,width=getexact(d,g);allow=(width>0)&~d.same_case
        exact_only=np.where(allow,ex,base)
        near_only=np.where(elig,near,base)
        combo=np.where((width>=3)&allow,ex,np.where(elig,near,exact_only))
        hit=allow|elig;z={'original':d.incumbent,'reviewer':d.base,'authors':auth,'context':base,'context_exact':exact_only,'context_near':near_only,'context_both':combo}
        met={k:metric(cm(d.gold,v)) for k,v in z.items()};sub={}
        slices={'any_eligible_memory':hit,'no_eligible_memory':~hit,'no_detected_near_judgment':d.match_ratio==0,'at_least_80pct_near_judgment':d.match_ratio>=.8,'no_full_or_near_judgment':d.match_ratio<.8,'full_judgment_duplicate':d.same_case,'exact_only':allow&~elig,'near_only':elig&~allow,'both_sources':allow&elig}
        for k,loc in slices.items():
            sub[k]=dict(rows=int(loc.sum()),cases_with_rows=int(sum(loc[a:b].any() for a,b in zip(d.offsets,d.offsets[1:]))),
                        models={name:metric(cm(d.gold[loc],pred[loc])) for name,pred in z.items()} if loc.any() else {})
        results[g]=dict(cases=len(d.case_ids),rows=len(base),scores=met,slices=sub,ci_vs_context=paired(d,combo,base),ci_vs_original=paired(d,combo,d.incumbent),
                        fixed=int(((combo==d.gold)&(base!=d.gold)).sum()),harmed=int(((combo!=d.gold)&(base==d.gold)).sum()))
        np.savez_compressed(H/('composition-'+g+'-predictions.npz'),gold=d.gold,offsets=d.offsets,case_ids=d.case_ids,any_memory=hit,**z)
        print(g,{k:(round(v['macro_f1']*100,3),v['pair']) for k,v in met.items()},'eligible',int(hit.sum()),flush=True)
    save('composition-official.json' if groups==['test'] else 'composition-validation.json',dict(freeze=sha(H/'composition-lock.json'),groups=results))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('development','official'));a=ap.parse_args();run(['test'] if a.phase=='official' else ['select','confirm','fresh'])
