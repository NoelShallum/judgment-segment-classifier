"""Run the frozen public-data GLiNER feature extractor on a verified NVIDIA T4."""
import argparse,csv,hashlib,json,os,re,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
MODEL_REV='b5d42bf7dc7d01c40304ae4ed43c2f13c325a75b'
MODEL_FILES={'model.safetensors':'4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b','config.json':'1b4e63ac964dc4204165bd1420735cd4c2e4bce97394d0b69fa3f7ea192a8e97','encoder_config/config.json':'db837d0dc587f5858687ef860c1f400de10f3c3e44f88daef8cbda80d74e4c9c','tokenizer.json':'cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3','tokenizer_config.json':'0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1'}
CSV_HASH={'train':'60028fc085e3cb018e8d49a8879ba796dd03a0838367891f3312f9e576921794','val':'8abb2d789325418fc5ab4186c0dc63798e02b8fd82f9e00c324185e6bd86f816'}
PROTO_HASH='774935ce7a811363514d5d1703840aaba8372bd30833168362fb0b50034cc6ac'

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def matches(path,expected):
    got=digest(path)
    if got!=expected:raise ValueError(f'SHA256 mismatch for {path}: {got} != {expected}')
    return got

def arguments():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--part',choices=('fit','select','check','interim','validation'),required=True);a.add_argument('--csv',type=Path,required=True);a.add_argument('--model-dir',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--verify-only',action='store_true',help='Hash all source inputs without loading any accelerator');a.add_argument('--case-ids-file',type=Path,help='Optional JSON list from a separately frozen scale manifest');a.add_argument('--case-ids-sha256',help='Required checksum for the alternate case-ID list');return a.parse_args()

def main():
    args=arguments();split='train' if args.part in ('fit','select','check') else 'val';verified={k:matches(args.model_dir/k,v) for k,v in MODEL_FILES.items()};verified['csv']=matches(args.csv,CSV_HASH[split]);verified['protocol']=matches(P/'protocol.json',PROTO_HASH)
    p=json.loads((P/'protocol.json').read_text());ids=p[args.part]
    if args.case_ids_file:
        if not args.case_ids_sha256:raise ValueError('Pass the SHA256 of the alternate case manifest')
        verified['case_ids']=matches(args.case_ids_file,args.case_ids_sha256);ids=json.loads(args.case_ids_file.read_text());assert isinstance(ids,list) and len(ids)==len(set(ids)) and all(isinstance(k,str) and k.isdigit() for k in ids)
    if args.verify_only:print(json.dumps(dict(verified=verified,split=split,case_count=len(ids)),indent=2));return
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    from collect import torch,np,BoundaryExtractor,collect_case,S,R
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('This worker requires exactly one visible NVIDIA T4')
    modelname=torch.cuda.get_device_name(0)
    if not re.search(r'\bT4\b',modelname):raise RuntimeError(f'T4 required, detected {modelname}')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True,warn_only=True)
    model=BoundaryExtractor.from_pretrained(str(args.model_dir.resolve()),use_flashdeberta=False).float().to('cuda:0').eval();model.processor.change_mode(is_training=False);schema=model.create_schema().entities(S).build();args.out.mkdir(parents=True,exist_ok=True)
    import transformers,importlib.metadata
    run=dict(model_revision=MODEL_REV,model_params=sum(t.numel() for t in model.parameters()),gpu=modelname,capability=torch.cuda.get_device_capability(0),torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,gliner2=importlib.metadata.version('gliner2'),native_dtype='float32',extractor_sha256=digest(P/'collect.py'),worker_sha256=digest(Path(__file__)),split=split,case_ids=ids,verified=verified)
    previous=list(args.out.glob('t4-*-provenance.json'))
    if list(args.out.glob('*.npz')) and not previous:raise RuntimeError('Use a fresh T4 output directory or an earlier verified T4 run')
    for manifest in previous:
        old=json.loads(manifest.read_text())
        for field in ('model_revision','model_params','gpu','torch','cuda','transformers','gliner2','extractor_sha256','worker_sha256'):
            if old.get(field)!=run[field]:raise RuntimeError(f'Output provenance mismatch for {field}')
    rfile=args.out/f't4-{split}-{args.part}-provenance.json';text=json.dumps(run,indent=2)
    if rfile.exists() and json.loads(rfile.read_text())!=json.loads(text):raise RuntimeError('Output directory already contains differing provenance')
    rfile.write_text(text);target={k:[] for k in ids}
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            k=row['Index'];st=row['Text'].strip()
            if k in target and st:target[k].append((st,R.index(row['Label'])))
    for k in ids:
        out=args.out/f'{split}-{k}.npz'
        if out.exists():
            with np.load(out) as cached:
                if len(cached['y'])!=len(target[k]) or not np.array_equal(cached['y'],np.array([y for _,y in target[k]])):raise RuntimeError(f'Existing row alignment mismatch for {k}')
            continue
        if not target[k]:raise ValueError(f'Missing judgment {k} in {split}')
        start=time.time();data=collect_case(model,schema,target[k]);tmp=out.with_suffix('.part.npz');np.savez(tmp,**data);tmp.replace(out);print(k,len(data['y']),round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
