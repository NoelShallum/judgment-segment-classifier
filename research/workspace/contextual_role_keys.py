"""Program an alternative readout from role keys encoded within mixed-role passages."""
import argparse
import itertools
import json
import numpy as np
import torch
import encoder_span_edit as study
from contrastive_hierarchy_edit import TOPICS,FRAMES
from program_trial import save_json
from seven_paths import ROOT

ROLES=np.arange(7)
def fictional_blocks():
    # A change of role is defined by the frame around a shared proposition.
    # Rotating each document removes a deterministic link between role and position.
    result={};labels=[]
    for topic,style in itertools.product(range(len(TOPICS)),(0,1)):
        order=np.roll(ROLES,(topic*3+style*2)%7)
        stimuli=[FRAMES[j][style].format(topic=TOPICS[topic]) for j in order]
        result[f"fiction-{topic}-{style}"]=stimuli
        labels.extend(order)
    return result,np.asarray(labels)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("encoder",choices=study.MODELS)
    args=ap.parse_args()
    torch.set_num_threads(2)
    name=args.encoder
    result=ROOT/"results"/f"encoder-span-{name}.json"
    if not result.exists():result=ROOT/"results"/f"encoder-span-{name}-accelerated.json"
    info=json.loads(result.read_text())
    study.CACHE=study.CACHE/(name if info.get("backend","torch")=="torch" else name+"-gpu")
    enc=study.Encoder(name)
    groups,bank=study.make_documents()
    rows,y=fictional_blocks()
    # New keys are encoded with identical pooling and window policy to each inference mode.
    h={};vectors={}
    neutral={"unlabeled":study.neutral_texts()}
    for mode in study.MODES:
        controls=enc.cache({**rows,**neutral},mode)
        features=np.concatenate([controls[k] for k in rows])
        h[mode]=list(study.candidate_heads(features,y,controls["unlabeled"]))
        vectors[mode]=enc.cache(bank,mode)
    trial=study.evaluate(name+"-contextual-keys",groups,vectors,h)
    save_json(ROOT/"results"/("encoder-contextual-keys-"+name+".json"),{"model":name,"author_key_count":len(y),
        "method":"Readout weights are derived from seven contrastive rhetorical role frames inside mixed-role sequences. Order varies by design. Readout keys and benchmark queries use the same bidirectional or isolated representation scheme. Validation chooses profiles. No gold training labels or gradients.",
        "tasks":trial})
    for task,v in trial.items():
        print(task,v["best"]["mode"],v["best"]["scores"]["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)
if __name__=="__main__":
    main()
