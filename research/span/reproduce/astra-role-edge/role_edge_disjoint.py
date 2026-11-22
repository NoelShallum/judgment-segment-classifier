"""Replay frozen write on the additional validation cohort, dropping identical raw texts."""
import json
import hashlib
from pathlib import Path

from run_role_edge import CRF, DONOR, HERE, OUT, HierModel, file_hash, independent_validation, load_file, measure, prepare, sample_whole, save_json, torch


def main():
    torch.set_num_threads(2)
    selection=json.loads((HERE/"frozen.json").read_text())
    controls=sample_whole()
    used={d['id'] for docs in controls.values() for d in docs}
    docs=independent_validation(used)
    seen={hashlib.sha256(d['text'].encode()).hexdigest() for ds in controls.values() for d in ds}
    retained=[];excluded=[]
    for case in docs:
        signature=hashlib.sha256(case['text'].encode()).hexdigest()
        if signature in seen:
            excluded.append(case['id'])
        else:
            retained.append(case)
        seen.add(signature)
    cached=json.loads((HERE/'overlap.json').read_text())['independent_exclusions_for_text_disjoint_check']
    assert excluded==cached
    weights=load_file(str(HERE/'role_edge.safetensors'));vocab=json.loads((HERE/'vocab.json').read_text())
    parents={k:load_file(str(OUT/f'whole-discourse-{k}.safetensors')) for k in selection['choices']}
    bank=prepare(retained,'independent',HierModel(DONOR),parents,vocab,weights)
    report={'cases':len(retained),'excluded_duplicate_ids':excluded,'frozen_sha256':file_hash(HERE/'frozen.json'),
            'results':measure(bank,parents,selection['choices'])}
    save_json(HERE/'text-disjoint-validation.json',report)
    print('Text-disjoint validation',len(retained),[(k,v['candidate']['exact_labeled_spans']['f1']) for k,v in report['results'].items()])

if __name__=='__main__':main()
