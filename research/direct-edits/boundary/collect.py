"""CPU-only extraction of pre-existing GLiNER2 native span-marginal features."""
import os, sys, json, csv, time, argparse
from pathlib import Path
P=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[k]='1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(P/'deps'));sys.path.append(str(P.parent/'ultrabert/deps311'))
import numpy as np, torch
from gliner2.models.boundary.engine import BoundaryExtractor
torch.set_num_threads(1);torch.set_num_interop_threads(1)
PROTO=json.loads((P/'protocol.json').read_text()); S=PROTO['schema']; R=PROTO['roles']; STORE=P/'features';STORE.mkdir(exist_ok=True)
CHANNELS=('start','end','inside')
PROJECTIONS=('start_boundary_projection','end_boundary_projection','inside_text_projection')

def load_model():
    m=BoundaryExtractor.from_pretrained(str(P/'model'),use_flashdeberta=False).eval();m.processor.change_mode(is_training=False)
    return m

def dataset(ids,split):
    out={k:[] for k in ids}
    with (P.parents[1]/'references'/'official'/f'{split}.csv').open() as f:
        for row in csv.DictReader(f):
            k=row['Index']; t=row['Text'].strip()
            if k in out and t: out[k].append((t,R.index(row['Label'])))
    return out

def get_blocks(m, texts):
    # The vendor word splitter supplies source character boundaries; schema consumes additional subwords.
    cur=[]; length=0
    for i,text in enumerate(texts):
        words=list(m.processor.word_splitter(text,lower=True)); assert words,(i,text)
        for start in range(0,len(words),80):
            part=words[start:start+80]; frag=text[part[0][1]:part[-1][2]]
            if cur and length+len(part)>96:
                yield cur;cur=[];length=0
            cur.append((i,frag,len(part)));length+=len(part)
    if cur:yield cur

def collect_case(m,schema,rows):
    texts,labels=zip(*rows);h=m.boundary_head;qh=h.boundary_query_head
    x=[[],[],[]]; q=[[],[],[]];scores=[];obs=[];weights=[]; qb=[]; token_sizes=[];parity=[]; blocks=list(get_blocks(m,texts))
    def batch_block(seq):
        joined=[' '.join(t for _,t,_ in block) for block in seq]
        b=m.processor.collate_fn_inference([(t,schema) for t in joined],architecture='boundary',max_len=None,error_policy='raise')
        n_sub=b.attention_mask.sum(1).tolist()
        # Long wordpiece blocks are split without losing rows or remaining words.
        if max(n_sub)>510:
            if len(seq)>1:
                k=len(seq)//2;batch_block(seq[:k]);batch_block(seq[k:]);return
            fraglist=seq[0]
            if len(fraglist)>1:
                k=len(fraglist)//2;batch_block([fraglist[:k]]);batch_block([fraglist[k:]]);return
            row,text,n=fraglist[0]; ww=list(m.processor.word_splitter(text,lower=True));assert n==len(ww) and n>1, (row,len(text),n,max(n_sub))
            k=n//2;batch_block([[(row,text[:ww[k-1][2]],k)]]);batch_block([[(row,text[ww[k][1]:],n-k)]]);return
        c=m._encode_core(b); enc=h.boundary_encoder(c['text_states'],c['text_mask']); native=qh(enc.states,enc.mask,c['text_states'],c['text_mask'],c['query_states'],c['query_mask']); heads=(native.start_logits,native.end_logits,native.inside_logits)
        for j,block in enumerate(seq):
            n_words=sum(z[2] for z in block); off=c['word_offsets'][j]; model_words=len(b.start_mappings[j]); extra=model_words-n_words; assert extra==0 or (extra==1 and b.start_mappings[j][-1]==b.end_mappings[j][-1]==len(joined[j])), (n_words,model_words,joined[j][:150]);assert int(c['text_mask'][j].sum())==model_words+off; assert [z['field_name'] for z in c['ext_specs'][j]]==S,(c['ext_specs'][j],S)
            query=c['query_states'][j]; query_id=len(q[0]); token_sizes.append(n_sub[j])
            qs=[]
            for t,name in enumerate(CHANNELS):
                dirs=getattr(qh,name+'_query_projection')(query);qs.append(dirs);q[t].append(dirs.cpu().numpy())
            current=off
            for row,_,n in block:
                end=current+n
                inp=[enc.states[j,current],enc.states[j,end],c['text_states'][j,current:end].mean(0)]
                score=[heads[0][j,:,current],heads[1][j,:,end],heads[2][j,:,current:end].mean(-1)]
                if len(parity)<2:
                    for t in range(3):
                        reconstructed=qs[t]@getattr(qh,PROJECTIONS[t])(inp[t])/np.sqrt(qh.boundary_dim)
                        parity.append(float((reconstructed-score[t]).abs().max()))
                for t in range(3):x[t].append(inp[t].cpu().numpy())
                scores.append(torch.stack(score).cpu().numpy());obs.append(row);weights.append(n);qb.append(query_id);current=end
    with torch.inference_mode():
        # Small no-padding-overhead batches keep a single CPU occupied.
        for i in range(0,len(blocks),3):batch_block(blocks[i:i+3])
    assert set(obs)==set(range(len(rows)))
    res={('x'+str(t)):np.array(x[t],'f4') for t in range(3)};res.update({('q'+str(t)):np.array(q[t],'f4') for t in range(3)})
    res.update(y=np.array(labels,np.int8),obs=np.array(obs,np.int32),weights=np.array(weights,np.float32),qb=np.array(qb,np.int32),native=np.array(scores,'f4'),subwords=np.array(token_sizes),parity=np.array(parity)); return res

def main():
    ap=argparse.ArgumentParser();ap.add_argument('part',choices=['train','four','rest','all']);args=ap.parse_args()
    p=PROTO;groups={'train':[('train',p['fit']+p['select']+p['check'])], 'four':[('val',p['interim'])], 'rest':[('val',p['validation'][4:])], 'all':[('train',p['fit']+p['select']+p['check']),('val',p['validation'])]}[args.part]
    m=load_model();schema=m.create_schema().entities(S).build()
    print('loaded',sum(v.numel() for v in m.parameters()),schema,flush=True);start=time.time()
    for split,ids in groups:
        pending=[k for k in ids if not (STORE/f'{split}-{k}.npz').exists()]
        if not pending:continue
        cases=dataset(pending,split)
        for k in pending:
            t=time.time();a=collect_case(m,schema,cases[k]); f=STORE/f'{split}-{k}.npz'; temp=f.with_suffix('.part.npz');np.savez(temp,**a);temp.replace(f)
            print(split,k,len(a['y']),len(a['qb']),len(a['subwords']),int(sum(a['subwords'])),round(time.time()-t,1),round(time.time()-start,1),float(max(a['parity'])),flush=True)
if __name__=='__main__':main()
