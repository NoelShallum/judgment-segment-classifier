import json
import numpy as np
from lex_counts import HERE,dump
from embedding_write import confusion,metric
def main():
    arms={}
    for path in HERE.glob('replay-*-shared-16.npz'):
        f=np.load(path);keys=f['keys'];g=f['gold'];h=f['pred']
        strong=np.concatenate([json.loads((HERE.parent/'pooler/shared-cases'/(str(k)+'.json')).read_text())['predictions']['unchanged'] for k in keys])
        agrees=(strong==h)
        differences={str(k):dict(support=int((g==k).sum()),weak_only_correct=int(((g==k)&(h==g)&~agrees).sum()),incumbent_only_correct=int(((g==k)&(strong==g)&~agrees).sum())) for k in range(7)}
        arms[path.stem]=dict(incumbent=metric(confusion(g,strong)),candidate=metric(confusion(g,h)),disagreements=int((~agrees).sum()),candidate_only_correct=int(((h==g)&~agrees).sum()),incumbent_only_correct=int(((strong==g)&~agrees).sum()),both_wrong=int(((strong!=g)&(h!=g)).sum()),by_human_role=differences)
    dump(HERE/'shared-error-overlap.json',dict(human_oracle_used_only_for_descriptive_disagreement_counts=True,arms=arms))
if __name__=='__main__':main()
