"""Solve and verify the seven native output rows, retaining the other projection rows."""
from pathlib import Path
import hashlib
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from safetensors.numpy import save_file
import scipy.linalg as sla

HERE=Path(__file__).resolve().parent
RIDGES=(.03,.15,.6)
RARITY=(0.,.4,.75)


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(name):
    return json.loads((HERE/name).read_text())


def info_by_case(y,p,lengths):
    counts=np.bincount(y*7+p,minlength=49)
    if len(lengths)==1:
        return counts.reshape(1,7,7)
    offset=np.cumsum([0,*lengths])
    return np.stack([np.bincount(y[a:b]*7+p[a:b],minlength=49).reshape(7,7) for a,b in zip(offset,offset[1:])])


def scores(a):
    tp=np.diagonal(a,axis1=-2,axis2=-1)
    f=2*tp/np.maximum(a.sum(-1)+a.sum(-2),1)
    return f.mean(-1),tp.sum(-1)/a.sum(axis=(-1,-2)),f


def ridge(features, y, x, label_map):
    z=features
    best=(-1.,-1.)
    artifact=None
    log=[]
    for rarity in RARITY:
        w=np.maximum(np.bincount(y,minlength=7),1)**-rarity
        weight=w[y]/w[y].mean()
        total=z.T @ (z*weight[:,None])
        rhs=z.T @ (np.eye(7)[y]*weight[:,None])
        diagonal=total.diagonal().mean()
        for reg in RIDGES:
            solve=sla.solve(total+diagonal*reg*np.eye(total.shape[0]),rhs,assume_a="pos")
            prediction=(x@solve).argmax(-1)
            f1,acc,_=scores(info_by_case(label_map,prediction,[len(label_map)]).sum(0))
            log.append([rarity,reg,float(f1),float(acc)])
            if (f1,acc)>best:
                best=(f1,acc)
                artifact=(rarity,reg,solve)
    return artifact,log


def concatenate(ids,mode):
    x,y,rows,lengths=[],[],[],[]
    for cid in ids:
        f=HERE/"features"/(cid+".npz")
        assert file_hash(f)==load(f"features/{cid}.json")["sha256"]
        with np.load(f,allow_pickle=False) as data:
            x.append(data[mode]);y.append(data["gold"]);rows.append(data["row"])
            lengths.append(len(data["gold"]))
    return np.concatenate(x).astype(np.float64),np.concatenate(y).astype(np.int64),np.concatenate(rows),lengths


def published_features(x,proj):
    p=np.round(127*np.tanh(x@proj.T))
    p/=np.maximum(np.linalg.norm(p,axis=-1,keepdims=True),1e-8)
    return np.column_stack((p,np.ones(len(p))))


def main():
    plan=load("protocol.json")
    baseline=np.concatenate([load("incumbent.json")[i] for i in plan["shared"]])
    proj=np.load(HERE/"projection.npy",allow_pickle=False)
    assert proj.shape==(1024,1024) and proj.dtype==np.float32
    where=HERE/"scored"
    where.mkdir(exist_ok=True)
    receipt={"protocol_sha256":file_hash(HERE/"protocol.json"),"t4_parity":load("native-parity.json"),"models":{},
             "rows":int(len(baseline)),"cases":len(plan["shared"])}
    preds={"incumbent":baseline}
    for mode in ("contextual","isolated"):
        tr,gt,_,_=concatenate(plan["fit"],mode)
        de,gd,_,_=concatenate(plan["select"],mode)
        ev,ge,rows,sizes=concatenate(plan["shared"],mode)
        assert len(ge)==len(baseline)
        (power,lam,matrix),search=ridge(tr,gt,de,gd)
        targeted=proj.copy()
        targeted[:7]=matrix.T.astype(np.float32)
        assert np.array_equal(targeted[7:],proj[7:])

        # Shuffling class targets keeps the same fit representation, regularizer and feature weights.
        gy=gt[np.random.default_rng(222023).permutation(len(gt))]
        w=np.maximum(np.bincount(gy,minlength=7),1)**-power
        vw=w[gy]/w[gy].mean()
        gram=tr.T@(vw[:,None]*tr)
        bad=sla.solve(gram+lam*gram.diagonal().mean()*np.eye(len(gram)),tr.T@(vw[:,None]*np.eye(7)[gy]),assume_a="pos").T
        delta=bad-proj[:7]
        shuffled=proj.copy()
        shuffled[:7]=(proj[:7]+delta*(np.linalg.norm(targeted[:7]-proj[:7])/np.linalg.norm(delta))).astype(np.float32)
        ordinary,head_grid=ridge(published_features(tr,proj),gt,published_features(de,proj),gd)

        save_file({"contextual_projection.weight":targeted},str(where/(mode+"-native-projection.safetensors")))
        save_file({"contextual_projection.weight":shuffled},str(where/(mode+"-shuffled-projection.safetensors")))
        save_file({"classifier.weight":ordinary[2][:-1].T.copy(),"classifier.bias":ordinary[2][-1].copy()},str(where/(mode+"-frozen-embed-head.safetensors")))
        acts={"native projection write":(ev@targeted[:7].T).argmax(1),
              "shuffled target write":(ev@shuffled[:7].T).argmax(1),
              "refit after frozen projected embeddings":(published_features(ev,proj)@ordinary[2]).argmax(1)}
        counts={name:info_by_case(ge,p,sizes) for name,p in acts.items()}
        inc=info_by_case(ge,baseline,sizes)
        draws=np.random.default_rng(1067).integers(len(sizes),size=(2000,len(sizes)))
        pert={}
        for name,cm in counts.items():
            f,a,r=scores(cm.sum(0))
            d=scores(cm[draws].sum(1))[0]-scores(inc[draws].sum(1))[0]
            pert[name]={"macro_f1":float(f),"accuracy":float(a),"role_f1":r.tolist(),"target_minus_incumbent_ci95":np.quantile(d,[.025,.975]).tolist()}
        preds.update({mode+" "+name:v for name,v in acts.items()})
        receipt["models"][mode]={"fit_rows":len(gt),"select_rows":len(gd),"role_support":np.bincount(ge,minlength=7).tolist(),
            "native_projection":{"tensor":"contextual_projection.weight","changed_values":int(np.count_nonzero(targeted-proj)),
                                 "total":int(proj.size),"first_seven_rows":True,"regularization":lam,"rarity_power":power,
                                 "replacement_sha256":file_hash(where/(mode+"-native-projection.safetensors")),
                                 "relative_frobenius":float(np.linalg.norm(targeted-proj)/np.linalg.norm(proj))},
            "fit_grid":search,"head_only_grid":head_grid,"scores":pert}
        print(mode, {k:round(v["macro_f1"],5) for k,v in pert.items()},flush=True)
    f,a,r=scores(info_by_case(ge,baseline,sizes).sum(0))
    receipt["incumbent"]={"macro_f1":float(f),"accuracy":float(a),"role_f1":r.tolist()}
    receipt["native_note"]="Logits directly expose the first seven physically written rows prior to the original retrieval quantization. The other projection rows and all backbone tensors retain their original values."
    (where/"report.json").write_text(json.dumps(receipt,indent=2)+"\n")
    np.savez_compressed(where/"shared-predictions.npz",row=rows,gold=ge,cases=np.repeat(plan["shared"],sizes),**preds)
    with ZipFile(HERE/"results.zip","w",compression=ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(where.iterdir()):z.write(p,p.name)
        for name in ("protocol.json","native-parity.json"):z.write(HERE/name,name)
        for f in sorted((HERE/"features").glob("*.json")):z.write(f,"features/"+f.name)
    print("archived",file_hash(HERE/"results.zip"),flush=True)


if __name__=="__main__":
    main()
