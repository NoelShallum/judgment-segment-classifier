"""Verify cached replay, live full-judgment inference, and an isolated CLI extraction."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from predict_role_edge import infer
from run_role_edge import (CACHE,CRF,DONOR,HERE,ROOT,HierModel,decode,decompose,document,file_hash,
                           gold_cases,load_file,save_json,text_features,torch,transition_scores)
from hier_model import LABELS
from span_model import merge


def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR); files=json.loads((HERE/'deployment.json').read_text())
    vocab=json.loads((HERE/'vocab.json').read_text())
    patches={k:load_file(str(HERE/p['file'])) for k,p in files['profiles'].items()}
    memory=load_file(str(HERE/'role_edge.safetensors'))
    frozen=json.loads((HERE/'frozen.json').read_text()); rows=gold_cases()
    sizes={k:sum(len(text.split()) for _,text,_ in items) for k,items in rows.items()}
    result={'deployment_sha256':file_hash(HERE/'deployment.json'),'cases':[],'cli':{}}
    for target in (550,4500,16000):
        case=min(sizes,key=lambda k:abs(sizes[k]-target)); d=document(case,rows[case]); q={}
        with np.load(CACHE/(case+'.npz')) as blob:raw=blob['raw']
        cues,position=text_features(d)
        edge=transition_scores(d['text'],d['units'],vocab,memory)
        for name,p in patches.items():
            mode,gain=frozen['choices'][name]
            logits=raw+cues@p['cue_head']+position@p['position_head']
            y=decode(logits,p[CRF],decompose(edge,mode),gain)
            expected=merge(d['units'],y,LABELS)
            actual=infer(d['text'],m,p,vocab,p,mode,p['edge.gain'].item())['spans']
            assert expected==actual,(case,name)
            q[name]=len(actual)
        result['cases'].append({'case':case,'whitespace_tokens':sizes[case],'candidate_units':len(d['units']),
                               'exactly_equal_cached_and_live_spans':q})
    with TemporaryDirectory(dir=HERE) as stage:
        root=Path(stage);edge=root/'edge';edge.mkdir()
        for name in ('hier_model.py','role_cues.py','span_model.py'):shutil.copyfile(ROOT/name,root/name)
        for name in ('role_edge_runtime.py','predict_role_edge.py','vocab.json','deployment.json',
                     *(d['file'] for d in files['profiles'].values())):
            shutil.copyfile(HERE/name,edge/name)
        text='The appellant submits that ₹1,000 was paid.\n\nThe court records the words “notice served”. We therefore allow the appeal. Costs follow.'
        source=root/'input.txt';source.write_text(text)
        for profile,p in patches.items():
            raw=subprocess.check_output([sys.executable,str(edge/'predict_role_edge.py'),str(source),'--model-dir',str(DONOR),
                                        '--profile',profile,'--threads','1'],text=True,cwd=root)
            actual=json.loads(raw)['spans']; mode,gain=frozen['choices'][profile]
            expected=infer(text,m,p,vocab,p,mode,gain)['spans']
            assert actual==expected and actual[0]['start']==0 and actual[-1]['end']==len(text)
            assert all(a['end']==b['start'] for a,b in zip(actual,actual[1:]))
            result['cli'][profile]={'spans':len(actual),'characters':len(text),'unicode_offset_coverage':True}
    save_json(HERE/'qa.json',result)
    print('Deployment QA',result,flush=True)

if __name__=='__main__':main()
