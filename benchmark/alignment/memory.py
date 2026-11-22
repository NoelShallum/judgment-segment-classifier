"""Program an associative table on official train and locate repeated partial judgments by monotone text alignment."""
import argparse,csv,difflib,hashlib,itertools,json,re,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file

H=Path(__file__).resolve().parent;ROOT=H.parents[1];DATA=ROOT/'references/official'
NAMES=['None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision']
RX=re.compile(r'[^\w]+',re.UNICODE);NUM=re.compile(r'\d+');TOK=re.compile(r'\w+',re.UNICODE)
def norm(s):return RX.sub(' ',s.casefold()).strip()
def masked(s):return NUM.sub('#',s)
def hashed(s):return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,x):
    p=H/name;t=p.with_suffix('.part');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def documents(split,keys=None,labels=False):
    seen=set();src=DATA/(split+'.csv')
    with src.open(encoding='utf-8-sig',newline='') as f:
        for k,rows in itertools.groupby(csv.DictReader(f),lambda r:r['Index']):
            assert k not in seen;seen.add(k)
            if keys is not None and k not in keys:continue
            words=[];gold=[];fp=hashlib.sha256()
            for row in rows:
                raw=row['Text']
                if not raw.strip():continue
                words.append(norm(raw));fp.update(' '.join(raw.casefold().split()).encode()+b'\0')
                if labels:gold.append(NAMES.index(row['Label']))
            yield k,words,gold,fp.hexdigest()
def slot(s):
    if len(s)<30 or s.count(' ')<5:return 0
    return hashed(masked(s))
def build():
    text=[];labels=[];keys=[];off=[0];fps=[]
    for k,v,y,fp in documents('train',labels=True):
        keys.append(k);fps.append(fp);text+=v;labels+=y;off.append(len(text))
    ids=np.fromiter((slot(s) for s in text),np.uint64,count=len(text))
    order=np.argsort(ids,kind='stable').astype(np.int32)
    table=np.eye(7,dtype=np.int8)[labels]
    save_file({'memory.weight':table},str(H/'train-values.safetensors'))
    np.savez(H/'index.npz',hashes=ids,order=order,sorted=ids[order],case_ids=np.array(keys),offsets=np.array(off),fingerprints=np.array(fps))
    save('source.json',dict(rows=len(text),cases=len(keys),train_csv_sha256=sha(DATA/'train.csv'),index_sha256=sha(H/'index.npz'),memory_sha256=sha(H/'train-values.safetensors'),
                           recipe='Each training sentence receives one seven-output one-hot value row. Stable 64-bit hashes propose candidate train cases; normalized text and ordered local alignment verify candidates before reading any value. No case identifier is a feature.'))
    return text
class Memory:
    def __init__(self,text):
        self.text=text;rec=json.loads((H/'source.json').read_text())
        assert sha(H/'index.npz')==rec['index_sha256'] and sha(H/'train-values.safetensors')==rec['memory_sha256']
        with np.load(H/'index.npz',allow_pickle=False) as f:
            self.keys=f['case_ids'];self.offsets=f['offsets'];self.sorted=f['sorted'];self.order=f['order'];self.fps=set(f['fingerprints'])
        self.value=load_file(str(H/'train-values.safetensors'))['memory.weight'];self.role=self.value.argmax(1)
        self.doc=np.repeat(np.arange(len(self.keys),dtype=np.int32),np.diff(self.offsets))
        r=np.random.default_rng(202611221);self.shuf=self.role[r.permutation(len(self.role))]
    def align(self,text,fp):
        n=len(text);query=[masked(x) for x in text];ids=np.array([slot(x) for x in text],np.uint64)
        lo=self.sorted.searchsorted(ids);hi=self.sorted.searchsorted(ids,side='right')
        votes=Counter();seen=set()
        for s,a,b in zip(ids,lo,hi):
            if not s or s in seen or b-a>60:continue
            seen.add(s)
            for doc in np.unique(self.doc[self.order[a:b]]):votes[int(doc)]+=float(1/np.sqrt(b-a))
        accum=np.zeros((n,7),np.float32);shuffle=accum.copy()
        quality=np.zeros(n,np.float32);exact=np.zeros(n,bool);block_len=np.zeros(n,np.int32);source_ids=np.full(n,-1,np.int32)
        best_overlap=0.
        for j,strength in votes.most_common(4):
            if strength<1.5:continue
            a,b=self.offsets[j:j+2];ref=self.text[a:b];maskedref=[masked(x) for x in ref]
            matcher=difflib.SequenceMatcher(None,query,maskedref,autojunk=False)
            blocks=[block for block in matcher.get_matching_blocks() if block.size>=2]
            anchors=sum(x.size for x in blocks);ratio=anchors/n;best_overlap=max(best_overlap,ratio)
            if anchors<3:continue
            proposed={}
            def propose(i,k,sim,length,ex=False):
                if not 0<=i<n or not 0<=k<len(ref):return
                q=sim*(.74+.26*min(1.,length/6))*(.85+.15*min(1.,ratio/.5))
                if i not in proposed or q>proposed[i][0]:proposed[i]=(q,k,ex,length)
            for block in blocks:
                i,k,length=block
                for h in range(length):
                    ex=text[i+h]==ref[k+h]
                    if len(text[i+h])<5 and length<4:continue
                    propose(i+h,k+h,1 if ex else .955,length,ex)
                for step in (-1,1):
                    edge_i,edge_k=(i,k) if step==-1 else (i+length-1,k+length-1)
                    for hop in (1,2,3):
                        ii,kk=edge_i+step*hop,edge_k+step*hop
                        if not 0<=ii<n or not 0<=kk<len(ref):break
                        u,v=text[ii],ref[kk]
                        if not u or not v:break
                        xs=set(u.split());ys=set(v.split());dice=2*len(xs&ys)/max(len(xs)+len(ys),1)
                        if min(len(u),len(v))<18 and u!=v:break
                        if dice<.58:break
                        # Token evidence must remain local to a verified contiguous aligned passage.
                        propose(ii,kk,min(.985,dice)*(.98**hop),length,u==v)
            for i,(q,k,ex,length) in proposed.items():
                val=q**8;accum[i,self.role[a+k]]+=val;shuffle[i,self.shuf[a+k]]+=val
                if q>quality[i]:quality[i]=q;source_ids[i]=j;block_len[i]=length;exact[i]=ex
        return dict(votes=accum,shuffled=shuffle,quality=quality,exact=exact,block=block_len,source_case=source_ids,match_ratio=np.full(n,best_overlap,np.float32),same_case=np.full(n,fp in self.fps,bool))
def score_inputs(groups):
    txt=None
    if not (H/'source.json').exists():txt=build()
    if txt is None:txt=[v for _,texts,_,_ in documents('train') for v in texts]
    mem=Memory(txt)
    for g in groups:
        base=H.parent/'reviewer'/(('official' if g=='test' else g)+'-predictions.npz')
        with np.load(base,allow_pickle=False) as b:
            ks=b['case_ids'].astype(str);offsets=b['offsets'];inc=b['incumbent'];new=b['all_role'] if 'all_role' in b else b['all-role']
            truth=b['gold'].copy()
        docs={k:(s,fp) for k,s,_,fp in documents('test' if g=='test' else 'val',set(ks))}
        result=[]
        for idx,key in enumerate(ks):
            s,fp=docs[key];assert len(s)==offsets[idx+1]-offsets[idx]
            result.append(mem.align(s,fp))
            if idx%100==99:print('align',g,idx+1,flush=True)
        bank={k:np.concatenate([v[k] for v in result]) for k in result[0]}
        np.savez_compressed(H/(g+'-bank.npz'),**bank,case_ids=ks,offsets=offsets,incumbent=inc,base=new,gold=truth)
        print('aligned',g,len(ks),len(truth),(bank['quality']>0).sum(),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('groups',nargs='+');args=ap.parse_args()
    if 'test' in args.groups:assert (H/'freeze.json').exists()
    score_inputs(args.groups)
