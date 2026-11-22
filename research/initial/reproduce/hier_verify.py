"""Check the rebuilt deterministic hierarchy against the authors' architecture and deployed interfaces."""
import hashlib,importlib.util,json,sys,types,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from hier_model import HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import save_json
from seven_paths import LABELS,ROOT,folder_for
from span_eval import sets
from span_model import merge

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def author_parity():
    torch.set_num_threads(1);sys.modules['model']=types.ModuleType('model')
    module('model.submodels',DONOR/'submodels.py');up=module('model.Hier_BiLSTM_CRF',DONOR/'Hier_BiLSTM_CRF.py')
    local=HierModel(DONOR)
    author=up.Hier_LSTM_CRF_Classifier(10,200,1,2,0,len(local.vocab),100,device='cpu')
    author.load_state_dict(load_file(str(DONOR/'hier.safetensors')));author.eval()
    for r in (author.sent_encoder,author.emitter):r.device='cpu'
    docs=sets()['confirm'][:4];diff=rows=0;seeded_agreement=0;highest=0.
    with torch.inference_mode():
        for d in docs:
            texts=[d['text'][a:b].strip() for a,b in d['reference_units']];tokens=[local.tokenize(t) for t in texts]
            for r in (author.sent_encoder,author.emitter):r.init_hidden=type(r).init_hidden.__get__(r,type(r))
            torch.manual_seed(42);random_reference=np.asarray(author([tokens])[0])-3
            for r in (author.sent_encoder,author.emitter):
                def zero(n,h=r.hidden_dim//2):return torch.zeros(2,n,h),torch.zeros(2,n,h)
                r.init_hidden=zero
            original=np.asarray(author([tokens])[0])-3
            x=local.contextualize(local.words(texts));emission=local.logits(x);ours=path_labels(emission,local.transitions)
            diff+=np.count_nonzero(original!=ours);rows+=len(ours);seeded_agreement+=np.sum(ours==random_reference)
            highest=max(highest,float(np.max(abs(author.emissions[0,:,3:].numpy()-emission))))
    assert diff==0 and highest<.002
    return {'cases':len(docs),'rows':rows,'zero_state_different_labels':int(diff),'max_emission_difference':highest,'native_stochastic_seed_42_same_labels':int(seeded_agreement)}
def verify():
    torch.set_num_threads(1);rec=author_parity();short=HierModel(DONOR,OUT/'excerpt.safetensors');doc=HierModel(DONOR,OUT/'document.safetensors');base=HierModel(DONOR)
    warm=[r['text'] for r in json.loads((folder_for('pilot')/'approved.json').read_text())[:20]]
    short.classify(warm[0]);elapsed=[]
    for t in warm:
        t0=time.perf_counter();short.classify(t);elapsed.append(1000*(time.perf_counter()-t0))
    evidence=[]
    groups=sets()['cloud']
    for label,which in [('hier',doc),('fragment',base)]:
        f=ROOT/'seven-spans'/(label+'-cloud.json')
        if not f.is_file():continue
        expected={r['doc']:r for r in json.loads(f.read_text())}
        same=0
        for d in groups:
            spans=(which.segment(d['text']) if label=='hier' else which.segment_fragment(d['text'],OUT/'fragment.safetensors'))['spans']
            same+=int(spans==expected[d['id']]['spans'])
        assert same==len(groups);evidence.append({'profile':label,'exactly_replayed':same})
    sx=[]
    for kind in ('pilot','confirmation','paragraphs','homogeneous'):
        f=folder_for(kind);rows=json.loads((f/'approved.json').read_text());record=json.loads((f/'hier.json').read_text())['hier_edited']['predictions']
        correct=sum(short.classify(r['text'])['label']==record[r['id']] for r in rows);assert correct==len(rows)
        sx.append({'cohort':kind,'matching_exported_labels':correct})
    multi='  The plaintiff’s notice was served.\n\nThe court dismissed the appeal.  '
    spans=doc.segment(multi)['spans'];assert ''.join(multi[s['start']:s['end']] for s in spans)==multi
    assert all(s['label'] in LABELS for s in spans)
    res={'upstream_parity':rec,'span_forward_replay':evidence,'excerpt_forward_replay':sx,'unicode_coverage':True,'laptop':'Intel Core i5-8250U, CPU FP32, one PyTorch thread. Concurrent benchmark jobs were running. Warm overhead includes tokenization and inference but excludes loading.','median_warm_ms_20':float(np.median(elapsed)),'p95_warm_ms_20':float(np.quantile(elapsed,.95))}
    save_json(ROOT/'results/hierarchical-implementation-qa.json',res);print(res,flush=True)
if __name__=='__main__':verify()
