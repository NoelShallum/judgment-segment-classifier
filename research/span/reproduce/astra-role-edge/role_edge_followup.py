"""Replay the deployable role edge patches against published predictions and Gemini."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from predict_role_edge import infer
from run_role_edge import (CACHE, CRF, DONOR, HERE, ROOT, HierModel, bootstrap, counts,
                           decode, decompose, document, file_hash, gold_cases, load_file,
                           path_labels, prepare_score, save_json, torch, transition_scores)
from hier_model import LABELS
from span_eval import summarize


def replay_raw(d,logits,patches,vocab):
    score=prepare_score(d,logits,patches)
    edge=transition_scores(d['text'],d['units'],vocab,next(iter(patches.values())))
    outputs={}
    for name,p in patches.items():
        mode='interaction' if p['edge.mode'].item() else 'joint'
        path=decode(score.logits[name],p[CRF],decompose(edge,mode),p['edge.gain'].item())
        outputs[name]=score.score(path)[:55]
    return outputs


def gemini(model,patches,vocab):
    requests=json.loads((ROOT/'seven-spans/cloud.json').read_text())
    reference=json.loads((ROOT/'seven-spans/gemini-cloud.json').read_text())
    heldout={name:[] for name in patches};cloud=[]
    for b in requests:
        k=b['doc']; g=reference[k]
        d=document(k,[(r['row'],r['text'],LABELS.index(r['gold'])) for r in b['rows']])
        assert hashlib.sha256(d['text'].encode()).hexdigest()==g['raw_sha256']
        assert g['unit_offsets']==[list(x) for x in d['units']]
        actual=model.logits(model.contextualize(model.words([d['text'][a:z] for a,z in d['units']])))
        candidates=replay_raw(d,actual,patches,vocab)
        for name,scores in candidates.items():heldout[name].append(scores)
        cloud.append(counts(d,d['units'],[LABELS.index(a) for a in g['unit_labels']]))
    cloud=np.stack(cloud)
    return {'cases':len(cloud),'archived_gemini':summarize(cloud.sum(0)),
            'candidates':{k:{'scores':summarize(np.sum(v,axis=0)), 'paired_gap_95pct':bootstrap(np.stack(v),cloud)} for k,v in heldout.items()}}


def main():
    torch.set_num_threads(2);deploy=json.loads((HERE/'deployment.json').read_text())
    patches={k:load_file(str(HERE/v['file'])) for k,v in deploy['profiles'].items()}
    for k,v in deploy['profiles'].items():assert file_hash(HERE/v['file'])==v['sha256']
    vocab=json.loads((HERE/'vocab.json').read_text())
    references=json.loads((ROOT/'results/archived-sentence-span-references.json').read_text())['scored_release_files']
    key='previous_predicted_role';folder=ROOT/'references/all-predictions/LegalSeg_InLegalBERT_Predictions'
    official=folder/'InLegalBert(i-1, label_p, i).csv'
    assert file_hash(official)==references[key]['bytes_sha256']
    with official.open(encoding='utf-8-sig',newline='') as f:
        archive=np.asarray([(int(x['true']),int(x['pred'])) for x in csv.DictReader(f)],np.int8)
    output={'deployment_sha256':file_hash(HERE/'deployment.json'),'matched_gemini':gemini(HierModel(DONOR),patches,vocab)}
    save_json(HERE/'independent-replay.json',output)
    reference=[];records={k:[] for k in patches};words=[];ids=[]
    for k,rows in gold_cases().items():
        d=document(k,rows);active=[i for i,t,_ in rows if t.strip()]
        assert np.array_equal(archive[active,0],d['reference_roles'])
        reference.append(counts(d,d['reference_units'],archive[active,1]))
        with np.load(CACHE/(k+'.npz')) as f:scores=replay_raw(d,f['raw'],patches,vocab)
        for name,s in scores.items():records[name].append(s)
        words.append(len(d['text'].split()));ids.append(k)
        if len(ids)%200==0:print('Replay',len(ids),flush=True)
    records={k:np.stack(v) for k,v in records.items()};reference=np.stack(reference);w=np.asarray(words)
    np.savez_compressed(HERE/'per-case-published-comparison.npz',**records,archive=reference,words=w,case_ids=np.asarray(ids))
    output['official_test']={'cases':len(reference),'strongest_archive':key,'archived_score':summarize(reference.sum(0)),
        'profiles':{name:{'scores':summarize(mat.sum(0)), 'paired_gap_against_archive_95pct':bootstrap(mat,reference),
                         'by_length':{band:{'cases':int(flag.sum()),'scores':summarize(mat[flag].sum(0))} for band,flag in
                              [('under_2000',w<2000),('2000_to_7999',(w>=2000)&(w<8000)),('8000_or_more',w>=8000)]}} for name,mat in records.items()}}
    save_json(HERE/'independent-replay.json',output)

if __name__=='__main__':main()
