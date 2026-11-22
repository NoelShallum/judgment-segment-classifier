"""Solve a one-step inverse for native document LSTM candidate-gate weights."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file
from scipy.special import expit

import sentence_dense_readout_write as dense
from sentence_document_graph_write import DONOR, EDGE, OUT, ROOT, HierModel, cm, file_hash, load_file, paired, row_summary, save_json, torch

TAG = "sentence-recurrent-inverse"
STORE = ROOT/"cache"/TAG
RESULT = ROOT/"results"
SIZE = 240
LAYERS = ("", "_reverse")
ARMS = ("readout-only", "input gate", "input and recurrent gate")
BASE = DONOR/"hier.safetensors"


def read(name):
    return json.loads((RESULT/(name+".json")).read_text())


def split_docs(split):
    if split=="fit":
        docs = dense.fit_documents()
        return sorted(docs,key=lambda d:sha256((TAG+"|"+d["id"]).encode()).digest())[:SIZE]
    if split in ("prospective","role-guard"):
        return dense.prospective_documents(split)
    return dense.groups_for(split)


def encode(shard):
    torch.set_num_threads(1)
    model = HierModel(DONOR)
    return [(d["id"], model.words([d["text"][a:b] for a,b in d["reference_units"]]).numpy())
            for d in shard]


def prepare(split):
    STORE.mkdir(exist_ok=True)
    path = STORE/(split+".npz")
    docs = split_docs(split)
    if not path.exists():
        with ProcessPoolExecutor(4) as pool:
            batches = pool.map(encode,[docs[i::4] for i in range(4)])
            lookup = dict(item for batch in batches for item in batch)
        words = [lookup[d["id"]] for d in docs]
        stage = path.with_suffix(".tmp")
        with stage.open("wb") as out:
            np.savez(out,keys=[d["id"] for d in docs],ends=np.r_[0,np.cumsum([len(v) for v in words])],
                     words=np.concatenate(words),backbone=file_hash(BASE))
        stage.replace(path)
        print("Cached",split,len(docs),sum(len(v) for v in words),flush=True)
    with np.load(path,allow_pickle=False) as f:
        np.testing.assert_array_equal(f["keys"],[d["id"] for d in docs])
        assert f["backbone"].item()==file_hash(BASE)
        x = f["words"]
        index = f["ends"]
    return docs,[x[a:b] for a,b in zip(index,index[1:])]


def raw_parameters(model):
    params = {"emitter.hidden2tag.weight":model.weight.copy(),"emitter.hidden2tag.bias":model.bias.copy()}
    for suffix in LAYERS:
        for item in ("weight_ih_l0","weight_hh_l0","bias_ih_l0"):
            name = item+suffix
            params["emitter.lstm."+name] = getattr(model.doc,name).detach().numpy().copy()
    return params


def inverse_targets(words, states, desired, model, direction):
    reverse = direction==1
    suffix = LAYERS[direction]
    take = slice(None,None,-1) if reverse else slice(None)
    width = model.doc.hidden_size
    section = slice(direction*width,(direction+1)*width)
    ordered, hidden, delta = words[take], states[take,section], desired[take,section]
    previous = np.vstack((np.zeros((1,width),np.float32),hidden[:-1]))
    def tensor(k):
        return getattr(model.doc,k+suffix).detach().numpy()
    pre = ordered@tensor("weight_ih_l0").T+previous@tensor("weight_hh_l0").T
    pre += tensor("bias_ih_l0")+tensor("bias_hh_l0")
    i,f,g,o = np.split(pre,4,axis=1)
    write, forget, content, gate = expit(i),expit(f),np.tanh(g),expit(o)
    memory = np.empty_like(hidden)
    cell = np.zeros(width,np.float32)
    for t in range(len(hidden)):
        cell = forget[t]*cell+write[t]*content[t]
        memory[t] = cell
    decoded = gate*np.tanh(memory)
    worst = float(np.max(np.abs(decoded-hidden)))
    assert worst < 6e-5,worst
    next_hidden = np.clip(decoded+delta,-.997*gate,.997*gate)
    shifted_cell = np.arctanh(next_hidden/np.maximum(gate,1e-8))
    step = shifted_cell-np.arctanh(np.clip(np.tanh(memory),-.997,.997))
    candidate = np.clip(content+step/np.maximum(write,.055),-.9999,.9999)
    target = np.clip(np.arctanh(candidate)-g,-.7,.7)
    target[delta==0] = 0.
    return np.column_stack((ordered,previous)),target,worst


def teacher():
    with np.load(STORE/"fit.npz",allow_pickle=False) as f:
        cases, bounds, x = f["keys"],f["ends"],f["words"]
        assert f["backbone"].item()==file_hash(BASE)
    words = [x[a:b] for a,b in zip(bounds,bounds[1:])]
    model = HierModel(DONOR)
    file = dense.CACHE/"fit.npz"
    assert file.is_file()
    with np.load(file,allow_pickle=False) as previous:
        keys,bounds,cur,gold = (previous[k] for k in ("cases","offsets","current","gold"))
    incumbent = {k:(cur[a:b],gold[a:b]) for k,a,b in zip(keys,bounds,bounds[1:])}
    E = model.weight[3:].astype(np.float64)
    E -= E.mean(0)
    covariance = E@E.T
    projection = np.linalg.solve(covariance+np.eye(7)*(.17*covariance.trace()/7.),E).astype(np.float32)
    h, teacher_targets = [],[]
    chips = [[],[]]
    targets = [[],[]]
    mistakes = 0
    largest = 0.
    for d,x in zip(cases,words):
        hidden = model.contextualize(torch.from_numpy(x))
        old,y = incumbent[d]
        assert len(y)==len(old)==len(x)
        logits = model.logits(hidden)
        row = np.arange(len(y))
        strength = np.clip(.55+.22*(logits[row,old]-logits[row,y]),.25,1.5)
        diff = ((np.eye(7)[y]-np.eye(7)[old])*strength[:,None]).astype(np.float32)
        desired = np.clip(diff@projection,-.14,.14)
        mistakes += int(np.count_nonzero(y!=old))
        teacher_targets.append(diff)
        h.append(hidden)
        for j in range(2):
            k,v,err = inverse_targets(x,hidden,desired,model,j)
            chips[j].append(k)
            targets[j].append(v)
            largest = max(largest,err)
    y = np.concatenate(teacher_targets)
    fit = {"readout-only":(np.concatenate(h),y),
            "gate":[(np.concatenate(x),np.concatenate(t)) for x,t in zip(chips,targets)]}
    details = {"rows":len(y),"cases":cases.tolist(),"incumbent_errors":mistakes,
              "max_baseline_lstm_reconstruction_error":largest}
    print("Constructed inverse targets",details["rows"],details["incumbent_errors"],largest,flush=True)
    return model,fit,details


def ridge(x,y,damping):
    n,d = x.shape
    center = x.mean(0,dtype=np.float64)
    scale = 1./np.maximum(x.std(0,dtype=np.float64),.035)
    z = ((x-center)*scale).astype(np.float32)
    gram = z.T@z / n
    target = z.T@(y-y.mean(0)) / n
    w = np.linalg.solve(gram.astype(np.float64)+np.eye(d)*damping*gram.trace()/d,target)
    affine = (w*scale[:,None]).T.astype(np.float32)
    return affine,(y.mean(0)-affine@center).astype(np.float32)


def edits_for(arm, original, fit, damping):
    out = {name:np.zeros_like(v) for name,v in original.items()}
    if arm=="readout-only":
        a,b = ridge(*fit[arm],damping)
        out["emitter.hidden2tag.weight"][3:] = a
        out["emitter.hidden2tag.bias"][3:] = b
    else:
        for suffix,(x,y) in zip(LAYERS,fit["gate"]):
            hidden = y.shape[1]
            input_key = "emitter.lstm.weight_ih_l0"+suffix
            rec_key = "emitter.lstm.weight_hh_l0"+suffix
            width = original[input_key].shape[1]
            if arm=="input gate":
                x = x[:,:width]
            a,b = ridge(x,y,damping)
            g = slice(2*hidden,3*hidden)
            out[input_key][g] = a[:,:width]
            if arm=="input and recurrent gate":
                out[rec_key][g] = a[:,width:]
            out["emitter.lstm.bias_ih_l0"+suffix][g] = b
    return {k:v for k,v in out.items() if np.any(v)}


def updated(original, delta, strength):
    return {k:(original[k]+strength*v).astype(np.float32) for k,v in delta.items()}


def install(model, state):
    with torch.no_grad():
        for k,v in state.items():
            if k=="emitter.hidden2tag.weight":
                model.weight[:] = v
            elif k=="emitter.hidden2tag.bias":
                model.bias[:] = v
            else:
                getattr(model.doc,k.removeprefix("emitter.lstm.")).copy_(torch.from_numpy(v))


def predict_cases(model, docs, words, cfg, vocab, side=None):
    from sentence_bert_referral_write import case_view
    predictions = []
    for d,x in zip(docs,words):
        h = model.contextualize(torch.from_numpy(x))
        if side is None:
            predictions.append(case_view(d,h,model,cfg,vocab)[2])
        else:
            lexical,w,selection = side
            _,_,base,anchor,feat = dense.dense_features(d,h,model,cfg,vocab,lexical)
            z = dense.output(dense.infer_basis(feat,"nonlinear",w),w)
            predictions.append(dense.replace(base,anchor,z,*selection))
    return predictions


def statistics(docs, ys):
    return np.stack([cm(np.asarray(d["reference_roles"]),p) for d,p in zip(docs,ys)])


def worker(task):
    arm,key,patch = task
    docs,words = VALIDATION
    torch.set_num_threads(1)
    model = HierModel(DONOR)
    cfg = load_file(str(dense.PARENT))
    vocab = json.loads((EDGE/"vocab.json").read_text())
    install(model,patch)
    result = row_summary(statistics(docs,predict_cases(model,docs,words,cfg,vocab)).sum(0))
    return arm,key,result["macro_f1"],result["accuracy"],result["per_role_f1"]["issue"]


def learn():
    torch.set_num_threads(1)
    global VALIDATION
    VALIDATION = prepare("select")
    model,fit,details=teacher()
    original = raw_parameters(model)
    combinations = {(arm,d):edits_for(arm,original,fit,d) for arm,d in product(ARMS,(.08,.5))}
    del fit,model
    trial = [("parent",(0.,0.),{})]
    for (arm,d),delta in combinations.items():
        for strength in (.5,1.,2.):
            trial.append((arm,(d,strength),updated(original,delta,strength)))
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(worker,trial))
    _,_,base_macro,base_accuracy,_ = results[0]
    plan = {"source":details,"source_sha256":{"backbone":file_hash(BASE),"decoder":file_hash(dense.PARENT),
                 "fit_cache":file_hash(STORE/"fit.npz")},
            "selection_cases":[d["id"] for d in VALIDATION[0]],"gradient_steps":0,"grid":results,"arms":{}}
    for arm in ARMS:
        eligible = [(m,a,k) for name,k,m,a,_ in results if name==arm and a>=base_accuracy-.001]
        options = eligible or [(m,a,k) for name,k,m,a,_ in results if name==arm]
        m,a,selection = max(options)
        guess = updated(original,combinations[arm,selection[0]],selection[1])
        dest = OUT/(TAG+"-"+str(ARMS.index(arm))+".safetensors")
        save_file(guess,str(dest))
        recovered = load_file(str(dest))
        assert all(np.array_equal(v,recovered[k]) for k,v in guess.items())
        plan["arms"][arm]={"choice":selection,"sha256":file_hash(dest),"select_macro_f1":m,"select_accuracy":a,
              "within_accuracy_floor":bool(eligible),"values_replaced":int(sum(np.count_nonzero(v!=original[k]) for k,v in guess.items())),
              "tensors":{k:list(v.shape) for k,v in guess.items()}}
        print("Selected",arm,selection,m,a,"baseline",base_macro,base_accuracy,flush=True)
    save_json(RESULT/(TAG+"-frozen.json"),plan)

def evaluate_chunk(task):
    docs,words,selected,gate = task
    torch.set_num_threads(1)
    model = HierModel(DONOR)
    original = raw_parameters(model)
    cfg = load_file(str(dense.PARENT))
    vocab = json.loads((EDGE/"vocab.json").read_text())
    spec = dense.read(dense.TAG+"-frozen")["arms"]["nonlinear"]
    path = OUT/(dense.TAG+"-2.safetensors")
    assert file_hash(path)==spec["sha256"]
    side = (load_file(str(dense.LEXICAL))["lexical"][0],load_file(str(path)),spec["selection"][3:])
    labels = {}
    for arm in ("parent",*ARMS,"dense parent","dense with gate"):
        install(model,original)
        patch = selected.get(gate if arm=="dense with gate" else arm,{})
        install(model,patch)
        labels[arm] = predict_cases(model,docs,words,cfg,vocab,side if arm.startswith("dense ") else None)
    matrices = {arm:statistics(docs,v) for arm,v in labels.items()}
    return [d["id"] for d in docs],matrices,{arm:int(sum(np.count_nonzero(a!=b) for a,b in zip(v,labels["dense parent" if arm.startswith("dense ") else "parent"]))) for arm,v in labels.items()}


def evaluate(splits):
    frozen = RESULT/(TAG+"-frozen.json")
    plan = read(TAG+"-frozen")
    selected = {}
    for arm,spec in plan["arms"].items():
        p = OUT/(TAG+"-"+str(ARMS.index(arm))+".safetensors")
        assert file_hash(p)==spec["sha256"]
        selected[arm] = load_file(str(p))
    gate = max(ARMS[1:],key=lambda a:(plan["arms"][a]["within_accuracy_floor"],plan["arms"][a]["select_macro_f1"]))
    stored = dense.read(dense.TAG)["splits"]
    ordered = dense.read("known-sentence-numerical-archives")
    reference = RESULT/"known-sentence-numerical-archives.npz"
    assert file_hash(reference)==ordered["score_file_sha256"]
    output = {"freeze_sha256":file_hash(frozen),"gate_with_dense":gate,"splits":{}}
    for split in splits:
        docs,words = prepare(split)
        with ProcessPoolExecutor(4) as pool:
            locations = {d["id"]:i for i,d in enumerate(docs)}
            tasks = [(docs[i::4],words[i::4],selected,gate) for i in range(4)]
            options = list(pool.map(evaluate_chunk,tasks))
        shape = (len(docs),7,7)
        alleys = {arm:np.empty(shape,np.int64) for arm in options[0][1]}
        changed = {k:0 for k in alleys}
        for keys,rows,ns in options:
            sub = [locations[k] for k in keys]
            for k,v in rows.items():
                alleys[k][sub] = v
                changed[k] += ns[k]
        old = row_summary(alleys["parent"].sum(0))
        flat = row_summary(alleys["dense parent"].sum(0))
        assert old==stored[split]["parent"],(split,old["macro_f1"],stored[split]["parent"]["macro_f1"])
        assert flat==stored[split]["arms"]["nonlinear"]["scores"]
        item = {"cases":len(docs),"parent":old,"dense_parent":flat,"arms":{}}
        if split=="test":
            name = ordered["best_by_macro_f1"]
            with np.load(reference,allow_pickle=False) as bank:
                np.testing.assert_array_equal(bank["cases"],[d["id"] for d in docs])
                author = bank[name]
            item["author"]={"name":name,"scores":row_summary(author.sum(0))}
        for arm,mat in alleys.items():
            if arm in ("parent","dense parent"):
                continue
            ref = "dense parent" if arm=="dense with gate" else "parent"
            item["arms"][arm]={"scores":row_summary(mat.sum(0)),"base":ref,"paired":paired(mat,alleys[ref]),
                                "changed":changed[arm]}
            if arm in ARMS:
                item["arms"][arm]["paired_to_readout"]=paired(mat,alleys["readout-only"])
            if split=="test":
                item["arms"][arm]["paired_author"]=paired(mat,author)
            if split=="select" and arm in ARMS:
                assert item["arms"][arm]["scores"]["macro_f1"]==plan["arms"][arm]["select_macro_f1"]
        output["splits"][split]=item
        save_json(RESULT/(TAG+".json"),output)
        print(split,"baseline",old["macro_f1"],
              {a:(round(v["scores"]["macro_f1"],7),round(v["scores"]["accuracy"],7),v["changed"])
               for a,v in item["arms"].items()},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("operation",choices=("prepare","learn","evaluate"))
    ap.add_argument("splits",nargs="*")
    args=ap.parse_args()
    if args.operation=="learn":
        learn()
    elif args.operation=="evaluate":
        evaluate(args.splits)
    else:
        for s in args.splits:
            prepare(s)
