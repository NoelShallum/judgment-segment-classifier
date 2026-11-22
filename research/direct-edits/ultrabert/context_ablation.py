"""Prespecified same-row sentence, local-block, and long-block forward passes using frozen weights."""
import argparse
import hashlib
import json
import os
import numpy as np
import torch
from safetensors.numpy import load_file
from extract import Encoder,groups,path_for,read
from study import SOURCE,KEY_PERM,data,load_source,metrics,compare
from joint import position,decode
from prepare import ROOT,save


def main():
    a=argparse.ArgumentParser();a.add_argument('--n',type=int,default=2);a.add_argument('--device',default='cpu')
    args=a.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')));torch.set_num_interop_threads(1)
    gs=groups('screen')['confirm'][:args.n]
    pulse=ROOT/'context-ablation-protocol.json'
    if not pulse.exists():save(pulse,{'cohort':'first two, then first eight previously frozen label-blind short official-train confirmation IDs. All original sentences. No weight or readout refit.',
        'conditions':[['isolated',384,'alone'],['local',384,'context'],['long',2048,'context']],
        'models':'both frozen case-agnostic joint_text and joint_position (external document position/decoder identified), against matched unedited control',
        'pending':'No ablation scores read; native maximum 32768 tokens; long evaluation uses 2048 for CPU.'})
    model=None;source=load_source()
    seq=ROOT/f'context-ablation-{args.n}';seq.mkdir(exist_ok=True)
    out={'protocol_sha256':hashlib.sha256(pulse.read_bytes()).hexdigest(),'ids':[r['id'] for r in gs],'weights_fitted_at_context':384,'conditions':{}}
    for name,cap,mode in [('local',384,'context'),('alone',384,'alone'),('long',2048,'context')]:
        blocks=[];freq=[]
        for item in gs:
            p=path_for(item,cap,mode)
            if not p.exists():
                if model is None:model=Encoder(args.device)
                v=model.run(item,cap,mode);p.parent.mkdir(parents=True,exist_ok=True)
                partial=p.with_suffix('.partial')
                with partial.open('wb') as f:np.savez(f,**v)
                partial.replace(p)
            v=read(item,cap,mode);blocks+=v['block_lengths'].tolist();freq.append(len(v['block_lengths']))
        d=data(gs,cap,mode)
        r={'target_sentences':len(d['y']),'target_tokens':int(d['tok'].sum()),'number_of_forward_blocks':len(blocks),'longest_actual_forward':max(blocks),'forward_blocks_by_case':freq,'models':{}}
        blob={k:d[k] for k in ['row','ix','case','y']}
        for kind in ['text','position']:
            p=ROOT/f'joint-screen-context-384/{kind}.safetensors'
            ck=load_file(str(p));W=ck['classifier.weight'].T
            k=(ck[SOURCE]-source).astype(float).T@W.astype(float)
            z=d['e']@W+position(d)@ck['position.weight'].T+ck['classifier.bias']
            tx=ck['transition.log_probabilities'];ts=float(ck['transition.gain'][0])
            zer=decode(d,z,tx,ts);target=decode(d,z+d['k']@k,tx,ts)
            shuffled=decode(d,z+d['k']@k[KEY_PERM],tx,ts)
            r['models'][kind]={'physical_checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                               'edited':metrics(d['y'],target),'matched_zero':metrics(d['y'],zer),'shuffle':metrics(d['y'],shuffled),'vs_zero':compare(d['y'],zer,target)}
            blob[kind]=target;blob[kind+'_matched_zero']=zer;blob[kind+'_shuffle']=shuffled
        out['conditions'][name]=r;save(seq/'report.json',out)
        np.savez_compressed(seq/f'{name}.npz',**blob)
        print(name,r['target_sentences'],r['number_of_forward_blocks'],r['longest_actual_forward'],[round(r['models'][k]['edited']['macro_f1_7'],5) for k in ['text','position']],flush=True)
if __name__=='__main__':main()
