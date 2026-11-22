"""Test a closed-form correction from LegalSeg's frozen 109M-parameter BERT."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fcntl import LOCK_EX, flock
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from safetensors.numpy import save_file
from scipy.special import softmax

from confirm_sentence_graph_holdout import gather_validation
from sentence_document_graph_write import (
    DONOR, EDGE, OUT, ROOT, HierModel, aligned_embeddings, case_inputs, cm, document,
    file_hash, gold_cases, graph_features, load_file, paired, route_cases,
    row_summary, sample_whole, save_json, score, torch,
)
from sentence_local_readout_runtime import calibrated_labels
from weight_edit import ORIGINAL, offsets, tensor

TAG = "sentence-bert-referral"
FOLDER = ROOT/"results"
CACHE = ROOT/"cache"/TAG
BERT_ART = ROOT/"runtime/openvino-bert-fp32"
PARENT = OUT/"sentence-document-calibrated-portable.safetensors"
MODES = ("sentence", "previous two and target")
RATE = .10
CAP = 12
LIMIT = 96


def read(name):
    return json.loads((FOLDER/(name+".json")).read_text())


def select(current, posterior):
    row = np.arange(len(current))
    other = posterior.copy()
    other[row,current] = -np.inf
    margin = posterior[row,current]-other.max(1)
    rank = np.where(np.isin(current,(2,4)) | np.isin(other.argmax(1),(2,4)), margin-.065, margin)
    k = min(CAP, max(1, int(np.ceil(RATE*len(rank)))))
    # Avoid spending BERT queries on confident assignments or changing court reasoning.
    eligible = row[(current != 5) & (margin <= .45)]
    return eligible[np.argsort(rank[eligible],kind="stable")[:k]]


def case_view(d, h, model, cfg, vocab, positions=None):
    rate, gain, threshold = cfg["local_readout.settings"]
    c = case_inputs(d,h,model,cfg,vocab)
    g = graph_features(c)[-1]
    z = ((g-cfg["local_readout.center"])*cfg["local_readout.scale"])@cfg["local_readout.matrix"]+cfg["local_readout.bias"]
    v = positive(z)
    posterior = (1-gain)*c.prior+gain*v
    document_labels = score([(c,None)],[z],route_cases([(c,None)],rate),"posterior",gain,True,threshold)[0]
    current = calibrated_labels(document_labels,c.structure,c.prior,posterior,cfg)
    query = select(current,posterior) if positions is None else np.asarray(positions,dtype=np.int64)
    before = np.maximum(query-1,0)
    after = np.minimum(query+1,len(current)-1)
    x = np.column_stack((c.structure[query,:33],np.eye(7)[current[query]],posterior[query],c.prior[query],
                         np.eye(7)[current[before]],np.eye(7)[current[after]]))
    return c.key,c.gold,current,query,x,posterior[query],c.texts


def make_views(args):
    docs,split = args
    torch.set_num_threads(1)
    model = HierModel(DONOR)
    cfg = load_file(str(PARENT))
    vocab = json.loads((EDGE/"vocab.json").read_text())
    return [case_view(d,h,model,cfg,vocab) for d,h in zip(docs,aligned_embeddings(model,docs,split))]


def positive(x):
    v = np.maximum(x,0.)
    return v/np.maximum(v.sum(1,keepdims=True),1.e-8)


def docs_for_fit():
    previous = read("sentence-readout-fusion-frozen")["cases_for_fit"]
    seen = set(previous) | set(read("sentence-document-graph-frozen")["train_cases"])
    records = gather_validation(ROOT/"references/official/train.csv")
    order = sorted((k for k,v in records.items() if v[1] and k not in seen),
                   key=lambda k:sha256((TAG+"-1|"+k).encode()).digest())[:144]
    return [document(k,records[k][0]) for k in order]


def groups_for(split):
    if split == "fit":
        return docs_for_fit()
    if split in ("select","confirm"):
        return sample_whole()[split]
    if split == "holdout":
        path = ROOT/"references/official/val.csv"
        plan = read("sentence-graph-holdout-frozen")
        assert plan["val_csv_sha256"]==file_hash(path)
        records = gather_validation(path)
        return [document(k,records[k][0]) for k in plan["cases"]]
    assert split == "test"
    return [document(k,v) for k,v in gold_cases().items()]


def views(split):
    docs = groups_for(split)
    with ProcessPoolExecutor(4) as pool:
        batches = pool.map(make_views,[(docs[i::4],split) for i in range(4)])
        by_key = {r[0]:r for b in batches for r in b}
    return [by_key[d["id"]] for d in docs]


def pack_ids(tokenizer, rows, mode):
    result = []
    for _,_,_,q,_,_,texts in rows:
        if not len(q):
            continue
        used = sorted({int(j) for i in q for j in (range(max(0,i-2),i+1) if mode==MODES[1] else (i,))})
        ids = dict(zip(used,tokenizer([texts[j] for j in used],add_special_tokens=False,verbose=False)["input_ids"]))
        for i in q:
            t = ids[i]
            if mode == MODES[0]:
                content=t[-(LIMIT-2):]
            else:
                successors = t[-(LIMIT-4):]
                remaining = LIMIT-2-len(successors)
                chunks = []
                for j in (i-1,i-2):
                    if j>=0 and remaining>1:
                        previous = ids[j][-(remaining-1):]
                        take = min(len(previous),max(0,48 if j==i-1 else 28))
                        if take:
                            chunks[:0] = [*previous[-take:],tokenizer.sep_token_id]
                            remaining -= take+1
                content=[*chunks,*successors]
            result.append([tokenizer.cls_token_id,*content,tokenizer.sep_token_id])
    return result


def load_views(split):
    from transformers import BertTokenizerFast
    CACHE.mkdir(exist_ok=True)
    path = CACHE/(split+"-views-v1.json")
    if not path.exists():
        out = views(split)
        stage = path.with_suffix(".tmp")
        save_json(stage,[[k,y.tolist(),current.tolist(),q.tolist(),x.tolist(),p.tolist(),texts]
                         for k,y,current,q,x,p,texts in out])
        stage.replace(path)
    raw = json.loads(path.read_text())
    out = [(k,np.asarray(y,np.int64),np.asarray(current,np.int64),np.asarray(q,np.int64),
            np.asarray(x,np.float64).reshape(-1,68),np.asarray(p,np.float64).reshape(-1,7),texts)
           for k,y,current,q,x,p,texts in raw]
    print(split,len(out),sum(len(r[3]) for r in out),"referred sentences",flush=True)
    tokenizer = BertTokenizerFast.from_pretrained(ROOT/"references/tokenizer",local_files_only=True)
    return out,tokenizer


def encode(split, dev="GPU", modes=(0,1)):
    from openvino_worker import embed_ids, get_engine
    rows, tok = load_views(split)
    src = json.loads((BERT_ART/"source.json").read_text())
    assert file_hash(ORIGINAL)==src["source"]
    assert file_hash(BERT_ART/"encoder.bin")==src["encoder_bin_sha256"]
    assert file_hash(BERT_ART/"encoder.xml")==src["xml_sha256"]
    assert file_hash(PARENT)==read("sentence-document-calibrated-portable-qa")["weights_sha256"]
    engine,_ = get_engine(dev)
    for arm in modes:
        mode = MODES[arm]
        inputs = pack_ids(tok,rows,mode)
        fingerprint=sha256(json.dumps([src,mode,inputs]).encode()).hexdigest()
        batches = CACHE/(split+f"-{arm}-"+fingerprint[:18]+".npy")
        if batches.is_file():
            continue
        n = len(inputs)
        x = np.empty((n,768),dtype="f4")
        start = perf_counter()
        for j in range(0,n,512):
            stop = min(j+512,n)
            chunk = batches.with_name(f"{batches.stem}-rows-{j}-{stop}-{dev.lower()}.npy")
            if chunk.is_file():
                block = np.load(chunk,allow_pickle=False)
                assert block.dtype==np.float32 and block.shape==(stop-j,x.shape[1])
            else:
                block = embed_ids(engine,inputs[j:stop],max_length=LIMIT)
                stage = chunk.with_suffix(".tmp")
                with stage.open("wb") as stream:
                    np.save(stream,block,allow_pickle=False)
                stage.replace(chunk)
            x[j:stop] = block
            print(split,mode,stop,"/",n,round(perf_counter()-start,1),flush=True)
        stage = batches.with_suffix(".tmp")
        with stage.open("wb") as f:
            np.save(f,x,allow_pickle=False)
        stage.replace(batches)
        save_json(CACHE/(split+f"-{arm}.json"),
                  {"file":batches.name,"rows":n,"weights_sha256":src["source"],"features_sha256":file_hash(batches),
                   "input_sha256":fingerprint,"backend":dev,"wall_seconds":round(perf_counter()-start,2)})


ARMS = ("document control","BERT sentence","BERT previous two and target")


def encoded(split, arms):
    rows, tokenizer = load_views(split)
    source=json.loads((BERT_ART/"source.json").read_text())
    features = {}
    receipts = {}
    for a in arms:
        if a==ARMS[0]:
            continue
        index = ARMS.index(a)-1
        rec = json.loads((CACHE/(split+f"-{index}.json")).read_text())
        path = CACHE/rec["file"]
        assert file_hash(path)==rec["features_sha256"] and rec["weights_sha256"]==source["source"]
        identity=sha256(json.dumps([source,MODES[index],pack_ids(tokenizer,rows,MODES[index])]).encode()).hexdigest()
        assert identity==rec["input_sha256"]
        vec = np.load(path,allow_pickle=False)
        assert vec.shape==(sum(len(r[3]) for r in rows),768)==(rec["rows"],768)
        features[a] = vec
        receipts[a] = rec
    return rows,features,receipts


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1.e-8)


def prototypes(vec, roles):
    center = vec.mean(0,dtype=np.float64)
    scale = 1/np.maximum(vec.std(0,dtype=np.float64),.06)
    norm = unit((vec-center)*scale)
    centroid = np.stack([norm[roles==i].mean(0) if np.any(roles==i) else np.zeros(vec.shape[1])
                         for i in range(7)])
    return {"prototype.center":center,"prototype.scale":scale,"prototype.matrix":unit(centroid).T.copy()}


def design(rows, vec, variant, weights):
    x = np.concatenate([r[4] for r in rows])
    if vec is None:
        return x
    if variant=="direct head":
        return vec
    logit = vec@tensor(ORIGINAL,"classifier.weight").T+tensor(ORIGINAL,"classifier.bias")
    prob = softmax(logit,axis=1)
    columns = [x,prob,prob*x[:,40:47]]
    if variant == "hidden":
        columns.append(vec)
    if variant == "prototypes":
        z = unit((vec-weights["prototype.center"])*weights["prototype.scale"])
        columns.append(z @ weights["prototype.matrix"])
    return np.column_stack(columns)


def fit(x, roles, rarity):
    center = x.mean(0)
    inv = 1/np.maximum(x.std(0),.025)
    z = np.column_stack(((x-center)*inv,np.ones(len(x))))
    weights = np.maximum(np.bincount(roles,minlength=7)/len(roles),.013)**-rarity
    w = weights[roles]
    gram = z.T@(z*w[:,None])
    rhs = z.T@(np.eye(7)[roles]*w[:,None])
    damping = np.r_[np.ones(z.shape[1]-1),.02]*gram.trace()/len(gram)
    for value in (.025,.16,.7):
        coeff = np.linalg.solve(gram+np.diag(value*damping),rhs)
        yield value,{"center":center,"scale":inv,"matrix":coeff[:-1].copy(),"bias":coeff[-1].copy()}


def response(x, cfg):
    return positive(((x-cfg["center"])*cfg["scale"])@cfg["matrix"]+cfg["bias"])


def apply(rows, response, gain, margin):
    outputs = []
    at = 0
    for _,_,current,q,_,p,_ in rows:
        n = len(q)
        pred = current.copy()
        a = np.arange(n)
        scores = (1-gain)*(.5*p+.5*np.eye(7)[current[q]])+gain*response[at:at+n]
        choice = scores.argmax(1)
        change = scores[a,choice]-scores[a,current[q]] >= margin
        pred[q[change]] = choice[change]
        outputs.append(pred)
        at += n
    assert at == len(response)
    return outputs


def aggregate(rows, pred):
    return np.stack([cm(r[1],p) for r,p in zip(rows,pred)])


def current_metrics(rows):
    base = aggregate(rows,[r[2] for r in rows])
    return base,row_summary(base.sum(0))

def paired_priority_roles(a,b):
    def f1(m):
        cols = m.sum(0)[[2,4]]+m.sum(1)[[2,4]]
        return 2*m.diagonal()[[2,4]]/np.maximum(cols,1)

    seed = np.random.default_rng(3891)
    samples = [f1(a[i].sum(0))-f1(b[i].sum(0)) for i in seed.integers(len(a),size=(1800,len(a)))]
    intervals = np.quantile(samples,[.025,.975],axis=0).T.tolist()
    return dict(zip(("issue","respondent_argument"),intervals))


def selected_policy():
    fit_rows,training,train_receipts = encoded("fit",ARMS)
    dev,select,dev_receipts = encoded("select",ARMS)
    base,metric = current_metrics(dev)
    prior = read("sentence-readout-fusion")["splits"]["select"]["arms"]["document only"]["scores"]
    assert metric==prior, (metric["macro_f1"],prior["macro_f1"])
    y = np.concatenate([r[1][r[3]] for r in fit_rows])
    fitted = {}
    bench = {}
    optimal = {}
    for name in ARMS:
        bench[name] = []
        for variant in (("scores","hidden","prototypes","direct head") if name in training else ("document",)):
            anchors = prototypes(training[name],y) if variant=="prototypes" else {}
            x = design(fit_rows, training.get(name), variant, anchors)
            val = design(dev, select.get(name), variant, anchors)
            for rarity in (0.,.28,.55):
                for damping,params in fit(x,y,rarity):
                    yhat = response(val,params)
                    for gain,cut in product((0.,.45,.65,.85,1.),(.02,.08,.15)):
                        out = apply(dev,yhat,gain,cut)
                        stats = row_summary(aggregate(dev,out).sum(0))
                        args = (variant,rarity,damping,gain,cut)
                        bench[name].append([args,stats["macro_f1"],stats["accuracy"]])
                        criterion = (stats["macro_f1"],stats["accuracy"],args)
                        if stats["accuracy"]>=metric["accuracy"]-.001 and criterion>optimal.get(name,(-1.,)):
                            optimal[name] = criterion
                            fitted[name] = {**params,**anchors}
    assert optimal.keys()==set(ARMS)
    digest = {}
    for name in ARMS:
        path = OUT/(TAG+f"-{ARMS.index(name)}.safetensors")
        save_file(fitted[name],str(path))
        got = load_file(str(path))
        for k,v in fitted[name].items():
            np.testing.assert_array_equal(got[k],v)
        digest[name] = file_hash(path)
    direct_heads = {}
    for name in ARMS[1:]:
        if optimal[name][-1][0]=="direct head":
            w = fitted[name]
            coeff = w["scale"][:,None]*w["matrix"]
            bias = w["bias"]-w["center"]@coeff
            source = OUT/(TAG+f"-{ARMS.index(name)}-direct.safetensors")
            save_file({"classifier.weight":coeff.T.copy().astype("f4"),"classifier.bias":bias.astype("f4")},str(source))
            direct_heads[source.name] = file_hash(source)
    _,header = offsets(ORIGINAL)
    count = sum(int(np.prod(x["shape"])) for k,x in header.items()
                if k!="__metadata__" and not k.endswith(("position_ids","token_type_ids")))
    freeze = {"selection_cases":[r[0] for r in dev],"fit_cases":[r[0] for r in fit_rows],
              "training_referred_rows":len(y),"validation_referred_rows":sum(len(r[3]) for r in dev),
              "selection_baseline":metric,"frozen_modes":{k:v[-1] for k,v in optimal.items()},
              "features_count":{k:int(v["matrix"].shape[0]) for k,v in fitted.items()},
              "selection_grid":bench,"checkpoints":digest,"direct_classifier_tensors":direct_heads,"encoder_parameters":count,
              "train_feature_receipts":train_receipts,"selection_feature_receipts":dev_receipts,
              "accuracy_floor_points":0.10,"max_tokens":LIMIT,"max_referrals_per_judgment":CAP,
              "referral_fraction_cap":RATE,"backbone_weights_sha256":file_hash(ORIGINAL),
              "document_weights_sha256":file_hash(PARENT),"gradient_steps":0}
    save_json(FOLDER/(TAG+"-frozen.json"),freeze)
    print("Selected",optimal,"document baseline",metric["macro_f1"],metric["accuracy"],flush=True)


def evaluate(splits, which):
    f = FOLDER/(TAG+"-frozen.json")
    frozen = read(TAG+"-frozen")
    freeze_hash = file_hash(f)
    path = FOLDER/(TAG+".json")
    result = read(TAG) if path.is_file() else {"freeze_sha256":freeze_hash,"splits":{}}
    assert freeze_hash==result["freeze_sha256"]
    frozen_parent = read("sentence-readout-fusion")["splits"]
    archive_info = read("known-sentence-numerical-archives")
    archive_name = archive_info["best_by_macro_f1"]
    which = [ARMS[0],*which]
    for split in splits:
        rows,data,encoding = encoded(split,which)
        base,this_baseline = current_metrics(rows)
        assert this_baseline==frozen_parent[split]["arms"]["document only"]["scores"]
        gold = np.concatenate([r[1] for r in rows])
        referred = np.concatenate([r[1][r[3]] for r in rows])
        errors_in_referral = np.concatenate([r[1][r[3]][r[1][r[3]]!=r[2][r[3]]] for r in rows])
        predictions = [r[2].copy() for r in rows]
        for r,y in zip(rows,predictions):
            y[r[3]]=r[1][r[3]]
        oracle = row_summary(aggregate(rows,predictions).sum(0))
        q = result["splits"][split] = {"cases":len(rows),"base":this_baseline,"arms":{},
                  "oracle":oracle,"referred":len(referred),"referred_true_by_role":np.bincount(referred,minlength=7).tolist(),
                  "total_true_by_role":np.bincount(gold,minlength=7).tolist(),
                  "referral_errors_by_role":np.bincount(errors_in_referral,minlength=7).tolist(),
                  "encodings":encoding}
        routed = sum((cm(r[1][r[3]],r[2][r[3]]) for r in rows),np.zeros((7,7),np.int64))
        q["referred_parent_confusion"] = routed.tolist()
        q["priority_errors_predicted_reasoning"] = {
            name:int(base[:,i,5].sum()) for i,name in ((2,"issue"),(4,"respondent_argument"))}
        if split=="test":
            public = FOLDER/"known-sentence-numerical-archives.npz"
            assert file_hash(public)==archive_info["score_file_sha256"]
            with np.load(public,allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["cases"],[r[0] for r in rows])
                other = archive[archive_name]
            q["author_archive"] = {"name":archive_name,"scores":row_summary(other.sum(0))}
        matrices = {}
        for name in which:
            variant,weighting,damp,gain,margin = frozen["frozen_modes"][name]
            pat = OUT/(TAG+f"-{ARMS.index(name)}.safetensors")
            assert file_hash(pat)==frozen["checkpoints"][name]
            weights = load_file(str(pat))
            estimates = response(design(rows,data.get(name),variant,weights),weights)
            pred = apply(rows,estimates,gain,margin)
            sc = matrices[name] = aggregate(rows,pred)
            q["arms"][name]={"scores":row_summary(sc.sum(0)),"paired_to_parent":paired(sc,base),
                     "paired_to_same_referral_document_control":paired(sc,matrices[ARMS[0]]),
                     "paired_priority_roles_to_parent":paired_priority_roles(sc,base),
                     "changed":sum(int((r[2]!=y).sum()) for r,y in zip(rows,pred))}
            if split=="test":
                q["arms"][name]["paired_to_author_archive"] = paired(sc,other)
        with path.with_suffix(".lock").open("a") as lock:
            flock(lock,LOCK_EX)
            latest = read(TAG) if path.is_file() else {"freeze_sha256":freeze_hash,"splits":{}}
            assert latest["freeze_sha256"]==freeze_hash
            latest["splits"][split] = q
            stage = path.with_suffix(".tmp")
            save_json(stage,latest)
            stage.replace(path)
        print(split,"baseline",this_baseline["macro_f1"],"oracle",oracle["macro_f1"],
              {name:(v["scores"]["macro_f1"],v["scores"]["accuracy"],v["changed"]) for name,v in q["arms"].items()},
              flush=True)


def parity():
    from openvino_worker import embed_ids, get_engine

    arm = ARMS[2]
    rows, raw, receipt = encoded("select",(arm,))
    _,tok = load_views("select")
    ids = pack_ids(tok,rows,MODES[1])
    sample = np.arange(0,len(ids),7)[:96]
    old = raw[arm]
    assert receipt[arm]["backend"]=="GPU"
    engine,_ = get_engine("CPU")
    out = embed_ids(engine,[ids[i] for i in sample],max_length=LIMIT)
    alternate = old.copy()
    alternate[sample] = out
    frozen = read(TAG+"-frozen")
    variant,*_,gain,margin = frozen["frozen_modes"][arm]
    artifact = OUT/(TAG+"-2.safetensors")
    assert file_hash(artifact)==frozen["checkpoints"][arm]
    w = load_file(str(artifact))
    score = [response(design(rows,v,variant,w),w) for v in (old,alternate)]
    repeat = [apply(rows,v,gain,margin) for v in score]
    change = sum(np.count_nonzero(a!=b) for a,b in zip(*repeat))
    detail = {"vectors":len(sample),"max_feature_error":float(np.max(np.abs(out-old[sample]))),
              "max_readout_error":float(np.max(np.abs(score[0]-score[1]))),"changed_labels":int(change),
              "select_feature_sha256":receipt[arm]["features_sha256"],"weights_sha256":file_hash(artifact),
              "frozen_sha256":file_hash(FOLDER/(TAG+"-frozen.json"))}
    save_json(FOLDER/(TAG+"-backend-parity.json"),detail)
    print("CPU and GPU",detail,flush=True)


def predict(sentences, device):
    from contextlib import redirect_stdout
    from hier_model import LABELS
    from openvino_worker import embed_ids, get_engine
    import sys

    if not isinstance(sentences,list) or not sentences or any(not isinstance(text,str) or not text.strip() for text in sentences):
        raise ValueError("Supply a nonempty JSON array of nonblank sentence strings")
    torch.set_num_threads(2)
    item = document("input",[(i,text,0) for i,text in enumerate(sentences)])
    model = HierModel(DONOR)
    cfg = load_file(str(PARENT))
    vocab = json.loads((EDGE/"vocab.json").read_text())
    state = model.contextualize(model.words([item["text"][a:b] for a,b in item["reference_units"]]))
    row = case_view(item,state,model,cfg,vocab)
    pred = row[2]
    if len(row[3]):
        arm = ARMS[2]
        frozen = read(TAG+"-frozen")
        artifact = OUT/(TAG+"-2.safetensors")
        assert file_hash(artifact)==frozen["checkpoints"][arm]
        assert file_hash(PARENT)==frozen["document_weights_sha256"]
        assert file_hash(ORIGINAL)==frozen["backbone_weights_sha256"]
        with redirect_stdout(sys.stderr):
            engine,tk = get_engine(device)
        vec = embed_ids(engine,pack_ids(tk,[row],MODES[1]),max_length=LIMIT)
        variant,*_,gain,margin = frozen["frozen_modes"][arm]
        weights = load_file(str(artifact))
        pred = apply([row],response(design([row],vec,variant,weights),weights),gain,margin)[0]
    return {"labels":[LABELS[i] for i in pred],"referred_sentences":len(row[3])}


if __name__=="__main__":
    import sys
    ap=argparse.ArgumentParser()
    ap.add_argument("operation",choices=("encode","prepare","choose","evaluate","parity","predict"))
    ap.add_argument("splits",nargs="*",metavar="SPLIT")
    ap.add_argument("--device",default="GPU",choices=("CPU","GPU"))
    ap.add_argument("--input",type=Path,help="JSON array for experimental predict")
    ap.add_argument("--mode",type=int,nargs="+",default=[0,1],choices=(0,1))
    # Support concurrent feature preparation while the Intel encoder processes another cohort.
    if len(sys.argv)>1 and sys.argv[1] in ("fit","select","confirm","holdout","test"):
        sys.argv.insert(1,"encode")
    args=ap.parse_args()
    if any(split not in ("fit","select","confirm","holdout","test") for split in args.splits):
        ap.error("Split must be fit, select, confirm, holdout or test")
    if args.operation=="choose":
        selected_policy()
    elif args.operation=="predict":
        if args.input is None:
            ap.error("predict requires --input")
        print(json.dumps(predict(json.loads(args.input.read_text()),args.device),ensure_ascii=False))
    elif args.operation=="parity":
        parity()
    elif args.operation=="evaluate":
        evaluate(args.splits,[ARMS[i+1] for i in args.mode])
    else:
        for split in args.splits:
            if args.operation=="encode":
                encode(split,args.device,args.mode)
            else:
                load_views(split)
