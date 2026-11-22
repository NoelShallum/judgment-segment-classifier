"""Count paired errors to estimate the unreachable oracle accuracy ceiling on an ensemble."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;o=p/'results';stage=sys.argv[1];sample=json.loads((o/(stage+'-predictions.json')).read_text());res={}
for k,pp in sample['incumbent'].items():
    with np.load(p/'features'/('val-'+k+'.npz')) as a:y=a['y']
    i=np.asarray(pp);e=np.asarray(sample['edit'][k]);correct_i=i==y;correct_e=e==y
    keys={'both_correct':correct_i&correct_e,'incumbent_only':correct_i&~correct_e,'edit_only':~correct_i&correct_e,'both_wrong_same_prediction':~correct_i&~correct_e&(i==e),'both_wrong_different_predictions':~correct_i&~correct_e&(i!=e)}
    res[k]={name:np.bincount(y[ix],minlength=7).tolist() for name,ix in keys.items()}
by_role={t:np.sum([x[t] for x in res.values()],axis=0).tolist() for t in next(iter(res.values()))};tot={t:sum(a) for t,a in by_role.items()};rows=sum(tot.values());inc_errors=tot['edit_only']+tot['both_wrong_same_prediction']+tot['both_wrong_different_predictions'];out=dict(definition='Gold-label oracle picks the edit when it is correct and the incumbent is wrong. The resulting accuracy bounds any selector choosing one of these two existing labels at each row. It is not an implementable gate.',rows=rows,total=tot,by_gold_role=by_role,by_judgment=res,oracle_pair_accuracy=(tot['both_correct']+tot['incumbent_only']+tot['edit_only'])/rows,fraction_of_incumbent_errors_already_corrected=tot['edit_only']/inc_errors,oracle_accuracy_gain=tot['edit_only']/rows);(o/(stage+'-complementarity.json')).write_text(json.dumps(out,indent=2));print(stage,tot,round(out['oracle_pair_accuracy'],5),round(out['fraction_of_incumbent_errors_already_corrected'],5))
