"""Edit all-position, multiscale sentence readouts with closed-form ridge."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from confirm_sentence_graph_holdout import gather_validation, previous_ids
from role_cues import cue_features
from sentence_bert_referral_write import PARENT, calibrated_labels, groups_for, paired_priority_roles, positive
from sentence_document_graph_write import (
    DONOR, EDGE, OUT, ROOT, HierModel, aligned_embeddings, case_inputs, cm, document,
    file_hash, graph_features, load_file, measure, paired, route_cases, row_summary, save_json, score, torch,
)
from sentence_ngram_residual_write import STEM as NGRAM, VECTORIZER

TAG = "sentence-dense-readout"
FOLDER = ROOT/"results"
CACHE = ROOT/"cache"/TAG
LEXICAL = OUT/(NGRAM+"-memory.safetensors")
FIT_CASES = 640
ARMS = ("current scores", "multiscale", "nonlinear")


def read(name):
    return json.loads((FOLDER/(name+".json")).read_text())


def fit_documents():
    # These judgments were not used to fit the parent calibration, graph, or lexical memory.
    excluded = set(read(NGRAM+"-frozen")["memory_training_cases"])
    excluded.update(read("sentence-document-graph-frozen")["train_cases"])
    excluded.update(read("sentence-readout-fusion-frozen")["cases_for_fit"])
    excluded.update(read("sentence-bert-referral-frozen")["fit_cases"])
    source = ROOT/"references/official/train.csv"
    data = gather_validation(source)
    keys = sorted((k for k,(_,size,_) in data.items() if size and k not in excluded),
                  key=lambda k:sha256((TAG+"-v1|"+k).encode()).digest())[:FIT_CASES]
    assert len(keys)==FIT_CASES
    return [document(k,data[k][0]) for k in keys]


def prospective_documents(split="prospective"):
    source = ROOT/"references/official/val.csv"
    path = FOLDER/(TAG+"-"+split+".json")
    records = gather_validation(source)
    if path.is_file():
        cohort = json.loads(path.read_text())
        assert cohort["csv_sha256"]==file_hash(source)
    else:
        seen = set()
        for directory in ("known-sentence-pair-edit","hier-emitter-edits-v1","whole-joint-states"):
            seen.update(p.stem.rsplit("-",1)[-1] for p in (ROOT/"cache"/directory).glob("*.npy"))
        for receipt in sorted(set(FOLDER.glob("*.json"))|set(EDGE.glob("*.json"))):
            seen.update(previous_ids(json.loads(receipt.read_text()),records))
        seen.intersection_update(records)
        assert set(read("sentence-graph-holdout-frozen")["cases"]).issubset(seen)
        text_counts = {}
        for _,_,digest in records.values():
            text_counts[digest] = text_counts.get(digest,0)+1
        eligible = sorted((n,k) for k,(_,n,h) in records.items() if k not in seen and n and text_counts[h]==1)
        strata = np.array_split(np.asarray([k for _,k in eligible]),4)
        size = min(49,min(len(stratum) for stratum in strata))
        assert size>=20,(len(seen),len(eligible))
        cases = [str(k) for stratum in strata for k in sorted(stratum,
                 key=lambda k:sha256((TAG+"-"+split+"-v1|"+k).encode()).digest())[:size]]
        cohort = {"csv_sha256":file_hash(source),"cases":cases,"cohort_size":len(cases),
                  "eligible":len(eligible),"previously_seen":sorted(seen),
                  "policy":"Equal case counts from four judgment-length quartiles. Exclude text duplicates within validation and IDs found in prior results or hierarchy feature caches."}
        save_json(path,cohort)
    assert len(set(cohort["cases"]))==len(cohort["cases"])
    return [document(k,records[k][0]) for k in cohort["cases"]]


def windows(values):
    values = values.astype(np.float32)
    n = len(values)
    i = np.arange(n)
    sum_to = np.vstack((np.zeros((1,values.shape[1])),np.cumsum(values,axis=0,dtype=np.float64)))
    for radius in (2,8,24,64):
        left = np.maximum(i-radius,0)
        right = np.minimum(i+1+radius,n)
        yield ((sum_to[i]-sum_to[left])/np.maximum(i-left,1)[:,None]).astype(np.float32)
        yield ((sum_to[right]-sum_to[i+1])/np.maximum(right-i-1,1)[:,None]).astype(np.float32)


def dense_features(doc, h, model, cfg, vocab, lexical):
    case = case_inputs(doc,h,model,cfg,vocab)
    graph = graph_features(case)[-1]
    rate,gain,min_margin = cfg["local_readout.settings"]
    z = ((graph-cfg["local_readout.center"])*cfg["local_readout.scale"])@cfg["local_readout.matrix"]+cfg["local_readout.bias"]
    posterior = (1-gain)*case.prior+gain*positive(z)
    document_labels = score([(case,None)],[z],route_cases([(case,None)],rate),"posterior",gain,True,min_margin)[0]
    current = calibrated_labels(document_labels,case.structure,case.prior,posterior,cfg)
    one = np.eye(7,dtype=np.float32)[current]
    core = np.column_stack((case.structure[:,:33],one,posterior,case.prior*posterior)).astype(np.float32)
    x = VECTORIZER.transform(case.texts)
    strength = (x@lexical)/np.sqrt(np.maximum(np.diff(x.indptr),1))[:,None]
    cue = cue_features(case.texts)
    anchor = (.62*posterior+.12*case.prior+.26*one).astype(np.float32)
    features = np.column_stack((core,graph[:,33:],strength,cue,*windows(posterior),
                               np.full(len(current),np.log1p(len(current))))).astype(np.float32)
    return case.key,case.gold,current,anchor,features


def block(spec):
    docs,split = spec
    torch.set_num_threads(1)
    model = HierModel(DONOR)
    cfg = load_file(str(PARENT))
    vocab = json.loads((EDGE/"vocab.json").read_text())
    lexical = load_file(str(LEXICAL))["lexical"][0]
    rows = [dense_features(d,h,model,cfg,vocab,lexical) for d,h in zip(docs,aligned_embeddings(model,docs,split))]
    return rows


def prepare(split):
    path = CACHE/(split+".npz")
    checks = {"parent":file_hash(PARENT),"lexical":file_hash(LEXICAL),
              "data":file_hash(ROOT/"references/official"/(("train" if split=="fit" else "test" if split=="test" else "val")+".csv"))}
    if path.exists():
        with np.load(path,allow_pickle=False) as store:
            out = tuple(store[k] for k in ("cases","offsets","gold","current","anchor","features"))
            assert json.loads(store["source"].item())==checks
    else:
        docs = fit_documents() if split=="fit" else prospective_documents(split) if split in ("prospective","role-guard","phrase-prospect") else groups_for(split)
        with ProcessPoolExecutor(4) as pool:
            batches = pool.map(block,[(docs[i::4],split) for i in range(4)])
            by_key = {row[0]:row for rows in batches for row in rows}
        items = [by_key[d["id"]] for d in docs]
        keys = np.asarray([x[0] for x in items])
        ends = np.r_[0,np.cumsum([len(x[1]) for x in items])]
        out = (keys,ends,*[np.concatenate([x[i] for x in items]) for i in range(1,5)])
        CACHE.mkdir(exist_ok=True)
        stage = path.with_suffix(".tmp")
        with stage.open("wb") as stream:
            np.savez(stream,source=json.dumps(checks),**dict(zip(("cases","offsets","gold","current","anchor","features"),out)))
        stage.replace(path)
    print(split,len(out[0]),len(out[2]),out[-1].shape[1],"features",flush=True)
    return out


def infer_basis(x, arm, saved):
    if arm=="current scores":
        return x[:,:54].copy()
    if arm=="multiscale":
        return x.copy()
    z = np.clip((x-saved["basis.center"])*saved["basis.scale"],-4.,4.)
    nonlinear = np.tanh(z@saved["basis.projection"]+saved["basis.offset"])
    contextual = np.column_stack((x[:,40:47],x[:,-57:-50]))
    interaction = (x[:,33:40,None]*contextual[:,None,:]).reshape(len(x),-1)
    return np.column_stack((x,nonlinear,interaction))


def learn_basis(x, arm):
    if arm!="nonlinear":
        return {}
    rng = np.random.default_rng(530891)
    width = x.shape[1]
    return {"basis.center":x.mean(0,dtype=np.float64).astype(np.float32),
            "basis.scale":(1./np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32),
            "basis.projection":(rng.standard_normal((width,160))/np.sqrt(width)).astype(np.float32),
            "basis.offset":rng.uniform(-.6,.6,160).astype(np.float32)}


def fit_moments(x, labels, anchor, w):
    gram = np.zeros((x.shape[1],x.shape[1]),np.float64)
    links = np.zeros((x.shape[1],14),np.float64)
    for lo in range(0,len(labels),8192):
        hi = min(lo+8192,len(labels))
        batch = x[lo:hi]
        hot = np.eye(7,dtype=np.float32)[labels[lo:hi]]
        weighted = batch*w[lo:hi,None]
        gram += batch.T@weighted
        links += weighted.T@np.column_stack((hot,hot-anchor[lo:hi]))
    return gram,links


def output(x, weight):
    return ((x-weight["center"])*weight["scale"])@weight["weight"]+weight["bias"]


def replace(current, anchor, score, mode, gain, margin, guard):
    if mode=="unchanged":
        return current.copy()
    if mode=="residual":
        mixed = anchor+gain*np.clip(score[:,7:],-.55,.55)
    else:
        mixed = (1-gain)*anchor+gain*positive(score[:,:7])
    new = mixed.argmax(1)
    r = np.arange(len(current))
    allows = mixed[r,new]-mixed[r,current]>=margin
    if guard:
        allows &= current!=5
    return np.where(allows,new,current)


def paired_confusion(bank, y):
    _, offsets, gold, *_ = bank
    return np.stack([cm(gold[a:b],y[a:b]) for a,b in zip(offsets,offsets[1:])])


def baseline(bank, split):
    z = row_summary(cm(bank[2],bank[3]))
    if split not in ("prospective","role-guard","phrase-prospect"):
        assert z==read("sentence-readout-fusion")["splits"][split]["arms"]["document only"]["scores"]
    return z


def learn():
    train = prepare("fit")
    val = prepare("select")
    original = baseline(val,"select")
    labels = train[2]
    freq = np.maximum(np.bincount(labels,minlength=7)/len(labels),.013)
    frozen = {"prospective_sha256":file_hash(FOLDER/(TAG+"-prospective.json")),
              "fit_cases":train[0].tolist(),"fit_rows":len(labels),"validation_cases":val[0].tolist(),
              "accuracy_floor_points":.10,"gradient_steps":0,"validation_parent":original,"arms":{},
              "source_weights_sha256":{"parent":file_hash(PARENT),"lexical":file_hash(LEXICAL)},
              "source_csv_sha256":{s:file_hash(ROOT/"references/official"/(s+".csv")) for s in ("train","val","test")}}
    for arm in ARMS:
        basis = learn_basis(train[-1],arm)
        x = infer_basis(train[-1],arm,basis)
        v = infer_basis(val[-1],arm,basis)
        center = x.mean(0,dtype=np.float64).astype(np.float32)
        scale = (1./np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32)
        x -= center
        x *= scale
        features = np.column_stack((x,np.ones(len(x),np.float32)))
        del x
        records = []
        fallback = (0.,0.,0.,"unchanged",0.,0.,False)
        peak = (original["macro_f1"],original["accuracy"],fallback)
        result = {**basis,"center":center,"scale":scale,
                  "weight":np.zeros((len(center),14),np.float32),"bias":np.zeros(14,np.float32)}
        difficulty = (1.-train[4][np.arange(len(labels)),labels])**2
        for rarity,focus in product((0.,.18,.35),(0.,1.5)):
            w = (freq[labels]**-rarity*(1+focus*difficulty)).astype(np.float32)
            gram, links = fit_moments(features,labels,train[4],w)
            reg = np.r_[np.ones(len(gram)-1),.02]*gram.trace()/len(gram)
            for damping in (.009,.055,.20):
                beta = np.linalg.solve(gram+np.diag(damping*reg),links).astype(np.float32)
                params = {**basis,"center":center,"scale":scale,"weight":beta[:-1].copy(),"bias":beta[-1].copy()}
                scores = output(v,params)
                for target,gain,margin,guard in product(("posterior","residual"),(.20,.5,.8,1.),(.015,.075,.14),(False,True)):
                    pred = replace(val[3],val[4],scores,target,gain,margin,guard)
                    macro,accuracy,reasoning,_ = measure(cm(val[2],pred))
                    setting = (rarity,focus,damping,target,gain,margin,guard)
                    records.append([setting,float(macro),float(accuracy),float(reasoning)])
                    rank = (macro,accuracy,setting)
                    if rank > peak and accuracy>=original["accuracy"]-.001:
                        peak, result = rank, params
        path = OUT/(TAG+"-"+str(ARMS.index(arm))+".safetensors")
        save_file(result,str(path))
        checked = load_file(str(path))
        for k,v in result.items():
            np.testing.assert_array_equal(checked[k],v)
        frozen["arms"][arm] = {"selection":peak[-1],"features":int(len(center)),
                               "selection_macro_f1":float(peak[0]),"selection_accuracy":float(peak[1]),
                               "sha256":file_hash(path),"search":records}
        print(arm,peak,flush=True)
    save_json(FOLDER/(TAG+"-frozen.json"),frozen)


def evaluate(splits):
    ref = read("known-sentence-numerical-archives")
    f = FOLDER/(TAG+"-frozen.json")
    plan = read(TAG+"-frozen")
    out = {"freeze_sha256":file_hash(f),"splits":{}}
    assert plan["prospective_sha256"]==file_hash(FOLDER/(TAG+"-prospective.json"))
    for split in splits:
        bank = prepare(split)
        base = baseline(bank,split)
        parent = paired_confusion(bank,bank[3])
        q = out["splits"][split] = {"cases":len(bank[0]),"parent":base,"arms":{}}
        if split=="test":
            src = FOLDER/"known-sentence-numerical-archives.npz"
            assert file_hash(src)==ref["score_file_sha256"]
            with np.load(src,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["cases"],bank[0])
                ref_name = ref["best_by_macro_f1"]
                author = saved[ref_name]
            q["author"] = {"name":ref_name,"scores":row_summary(author.sum(0))}
        for arm,spec in plan["arms"].items():
            path = OUT/(TAG+"-"+str(ARMS.index(arm))+".safetensors")
            assert file_hash(path)==spec["sha256"]
            w = load_file(str(path))
            z = output(infer_basis(bank[-1],arm,w),w)
            _,_,_,target,gain,margin,guard = spec["selection"]
            y = replace(bank[3],bank[4],z,target,gain,margin,guard)
            counts = paired_confusion(bank,y)
            r = q["arms"][arm] = {"scores":row_summary(counts.sum(0)),"paired_parent":paired(counts,parent),
                                  "changed":int(np.count_nonzero(y!=bank[3])),
                                  "fixed":int(np.count_nonzero((y==bank[2]) & (bank[3]!=bank[2]))),
                                  "regressed":int(np.count_nonzero((y!=bank[2]) & (bank[3]==bank[2]))),
                                  "paired_priority_roles":paired_priority_roles(counts,parent)}
            if split=="test":
                r["paired_author"] = paired(counts,author)
            if split=="select":
                assert r["scores"]["macro_f1"]==spec["selection_macro_f1"]
            print(split,arm,r["scores"]["macro_f1"],r["scores"]["accuracy"],r["changed"],flush=True)
        save_json(FOLDER/(TAG+".json"),out)


def preserve_issues(parent, proposal):
    return np.where((parent==2)|(proposal==2),parent,proposal)


def load_predictor(arm):
    plan = read(TAG+"-frozen")
    spec = plan["arms"][arm]
    path = OUT/(TAG+"-"+str(ARMS.index(arm))+".safetensors")
    assert file_hash(path)==spec["sha256"]
    assert plan["source_weights_sha256"]=={"parent":file_hash(PARENT),"lexical":file_hash(LEXICAL)}
    return (HierModel(DONOR),load_file(str(PARENT)),json.loads((EDGE/"vocab.json").read_text()),
            load_file(str(LEXICAL))["lexical"][0], load_file(str(path)),spec)


def predict(sentences, arm, runtime, keep_issue=False):
    from hier_model import LABELS
    if not isinstance(sentences,list) or not sentences or any(not isinstance(t,str) or not t.strip() for t in sentences):
        raise ValueError("Supply a nonempty JSON array of nonblank sentence strings")
    model, parent, vocab, lexical, weights, spec = runtime
    item = document("input",[(i,text,0) for i,text in enumerate(sentences)])
    state = model.contextualize(model.words([item["text"][a:b] for a,b in item["reference_units"]]))
    _,_,current,anchor,x = dense_features(item,state,model,parent,vocab,lexical)
    z = output(infer_basis(x,arm,weights),weights)
    _,_,_,*action = spec["selection"]
    labels = replace(current,anchor,z,*action)
    if keep_issue:
        labels = preserve_issues(current,labels)
    return {"labels":[LABELS[i] for i in labels]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("operation",choices=("prepare","learn","evaluate","cohort","predict"))
    ap.add_argument("splits",nargs="*")
    ap.add_argument("--input",type=Path)
    ap.add_argument("--arm",choices=ARMS,default="nonlinear")
    ap.add_argument("--preserve-issues",action="store_true")
    args = ap.parse_args()
    if args.operation=="predict":
        if args.input is None:
            ap.error("predict requires --input")
        torch.set_num_threads(2)
        print(json.dumps(predict(json.loads(args.input.read_text()),args.arm,load_predictor(args.arm),args.preserve_issues),ensure_ascii=False))
    elif args.operation=="cohort":
        print("Reserved prospectively",len(prospective_documents()),"judgments",flush=True)
    elif args.operation=="learn":
        learn()
    elif args.operation=="evaluate":
        evaluate(args.splits)
    else:
        for split in args.splits:
            prepare(split)
