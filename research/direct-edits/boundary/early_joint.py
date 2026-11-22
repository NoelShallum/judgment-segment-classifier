"""Small fixed-regularization test of the second closed-form solver on fit-pool cases."""
from score import PR,readcases,native,get_pred,evaluate,NAMES,KEYS,OUT,hsh,np,json
from joint_solver import joint
trainids=PR['fit'][:3]; checkids=PR['fit'][3:5]; train=readcases('train',trainids);check=readcases('train',checkids)
r=0.3;dr,diag=joint(train,(r,));cr,cdiag=joint(train,(r,),True);d=dr[r];s=[c*(np.linalg.norm(a)/np.linalg.norm(c)) for a,c in zip(d,cr[r])];pred={'native':{k:native(a).argmax(1) for k,a in check.items()},'joint_edit':get_pred(check,d),'shuffled':get_pred(check,s)}
report=dict(purpose='Pipeline diagnostic using five preassigned fit cases; no official validation or test.',fit=trainids,check=checkids,fit_rows=sum(len(x['y']) for x in train.values()),results={name:evaluate(check,p)[0] for name,p in pred.items()},solver=diag,shuffled_solver=cdiag,norms={k:[float(np.linalg.norm(a)),float(np.linalg.norm(b))] for k,a,b in zip(KEYS,d,s)});(OUT/'early-joint.json').write_text(json.dumps(report,indent=2));print({k:{z:round(v[z],5) for z in ('macro_f1','accuracy','exact_span_micro_f1','rows')} for k,v in report['results'].items()});print(diag,cdiag)
