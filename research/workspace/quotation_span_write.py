"""Write explicit evidence for quoted authorities and statutory material under LegalSeg's None role."""
import itertools
import re
import numpy as np
from safetensors.numpy import load_file,save_file
import torch

from discourse_span_edit import text_features
from hier_full import CACHE,gold_cases
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from preserved_emitter_edit import embeddings
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash

RULES=(
    ("statute_quotation_lead",r"\b(?:section|clause|article|rule|provision|sub.section)\s+.{0,70}?\b(?:reads?\s+(?:as follows|thus|as under)|is reproduced\s+(?:below|hereunder)|(?:which\s+)?provides\s+as follows)\b"),
    ("extract_lead",r"\b(?:reads? as follows|reads? as under|reads? thus|reproduced (?:below|hereunder)|the following (?:passage|observations?|extract))\b"),
    ("reporter_citation_only",r"^\s*(?:\[?\(?\d{1,3}\)?[.)\]]\s+)?(?:\(?\d{4}\)?\s*[\[(]?\d{0,3}[\])]?\s+)?(?:SCC|AIR|SCR|Cri\s*LJ|SCC\s*OnLine|Supp\s*SCR|A\.?I\.?R\.?)\b.{0,35}$"),
    ("contents_heading",r"^\s*(?:section|chapter|part)\s+(?:\d+|[IVXLCDM]+)\s*(?:[.:-]\s*|\s+)(?:definitions?|short title|interpretation|commencement|miscellaneous|application)\b"),
    ("definition",r"^\s*(?:\(?\w\)[.,]?\s*)?[\"“‘'][-\w ]{2,54}[\"”’']\s+(?:means|includes|has the same meaning)\b"),
    ("quoted_text",r"^\s*(?:\d{1,3}[.)]\s*)?[\"“‘'].{16,}"),
    ("proviso",r"^\s*(?:provided (?:further )?that|explanation\s*(?:\d+|[IVXLCDM]+)?\s*(?:[.:(–—-]|for the purposes))\b"),
)
PATTERNS=tuple(re.compile(v,re.I) for _,v in RULES)
RETURN_TO_OPINION=re.compile(r"^\s*(?:(?:having|after)\s+considered|we\s+(?:consider|find|are of|hold|conclude|do not)|in\s+(?:our|my)\s+(?:view|opinion)|therefore)",re.I)

def indicators(d):
    snippets=[d["text"][a:b].strip() for a,b in d["units"]]
    keys=np.asarray([[p.search(t[:300]) is not None for p in PATTERNS] for t in snippets],dtype=np.float32)
    watch=np.zeros(len(snippets),dtype=np.float32)
    for i,x in enumerate(keys):
        if not (x[0] or x[1]):continue
        for j in range(i+1,min(i+4,len(keys))):
            candidate=snippets[j]
            if RETURN_TO_OPINION.search(candidate):break
            if j==i+1 or keys[j,3:].any() or snippets[j-1][-1:] in {":",";","“",'"'}:
                watch[j]=1.
            else:
                break
    return np.column_stack((keys,watch))

SIGNALS={
    "formal":np.array([1.5,.7,1.5,1.2,1.,0.,.9,0.],np.float32),
    "scope":np.array([1.5,.7,1.5,1.2,1.,.3,.9,.6],np.float32),
    "statute_only":np.array([1.5,.2,0.,1.2,1.,0.,.75,.3],np.float32),
}
DIRECTION=np.array([1.,0.,0.,0.,0.,-.8,-.2],np.float32)

def confidence(a,b):
    rng=np.random.default_rng(77922);xs=[]
    for ii in rng.integers(len(a),size=(1600,len(a))):
        x=summarize(a[ii].sum(0));z=summarize(b[ii].sum(0))
        xs.append([x["exact_labeled_spans"]["f1"]-z["exact_labeled_spans"]["f1"],x["token_accuracy"]-z["token_accuracy"]])
    return np.quantile(xs,[.025,.975],axis=0).T.tolist()

def main():
    torch.set_num_threads(2)
    hierarchy=HierModel(DONOR)
    cohorts=sample_whole()
    source={k:load_file(str(OUT/f"whole-discourse-{k}.safetensors")) for k in ("exact","balanced")}
    def stage(d,raw):
        clue,pos=text_features(d)
        old={n:raw+clue@v["cue_head"]+pos@v["position_head"] for n,v in source.items()}
        return d,old,indicators(d)
    banks={}
    for name,docs in cohorts.items():
        data=embeddings(hierarchy,docs,"raw-boundary-write-"+name)
        banks[name]=[stage(d,hierarchy.logits(z)) for d,z in zip(docs,data)]
    def evaluate(groups,n,variant,strength):
        layer=SIGNALS[variant][:,None]*DIRECTION[None,:]*strength
        return np.stack([counts(d,d["units"],path_labels(r[n]+h@layer,source[n][CRF])) for d,r,h in groups])
    chosen={};report={}
    for name in source:
        parent=summarize(evaluate(banks["select"],name,"formal",0).sum(0))
        best=None;grid=[]
        for m,g in itertools.product(SIGNALS,(0.,.3,.7,1.2,2.)):
            q=summarize(evaluate(banks["select"],name,m,g).sum(0))
            f=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
            grid.append([m,g,*f])
            if name=="balanced" and f[1]<parent["token_accuracy"]-.01:continue
            if best is None or f>best[0]:best=(f,m,g)
        _,m,g=best;chosen[name]=[m,g]
        p=OUT/f"whole-quote-cues-{name}.safetensors"
        save_file({**source[name],"quotation.weight":(SIGNALS[m][:,None]*DIRECTION[None,:]*g).astype("f4")},str(p))
        report[name]={"choice":[m,g],"weights_sha256":file_hash(p),"grid":grid,"scores":{}}
    freeze=ROOT/"results/quotation-span-freeze.json"
    save_json(freeze,{"profiles":report,"validation_cases":{k:[d["id"] for d in e] for k,e in cohorts.items()}})
    print("Frozen quotation edits",chosen,flush=True)
    data=[]
    for i,(k,rows) in enumerate(gold_cases().items(),1):
        doc=document(k,rows)
        with np.load(CACHE/(k+".npz")) as pack:
            data.append(stage(doc,pack["raw"]))
        if i%144==0:print("Raw test inputs",i,"/",712,flush=True)
    banks["test"]=data
    banks["test_long"]=[z for z in data if len(z[0]["text"].split())>=8000]
    for name,x in report.items():
        m,g=chosen[name]
        for k,rs in banks.items():
            a=evaluate(rs,name,m,g);b=evaluate(rs,name,m,0)
            x["scores"][k]={"cases":len(a),"candidate":summarize(a.sum(0)),"incumbent":summarize(b.sum(0)),
                            "span_and_token_gain_ci95":confidence(a,b)}
        z=x["scores"]["test"]["candidate"]
        print(name,z["exact_labeled_spans"]["f1"],z["token_accuracy"],flush=True)
    save_json(ROOT/"results/quotation-span-edit.json",{"premise":"LegalSeg Table 4 uses clause and quotation introductions as examples of None. These cue and governance weights have no learned regex parameters. Only a fixed gain and cue policy are selected on development. Confirmation and full test are case-disjoint raw inputs.",
      "rule_names":[k for k,_ in RULES]+["bounded_continuation"],"freeze_sha256":file_hash(freeze),"profiles":report})
if __name__=="__main__":
    main()
