"""Read layer 12 and write the actual last MLP to test whether the masked-LM output hides role signal."""
import json,os
import numpy as np
import scipy.linalg as la
import torch
from safetensors.numpy import save_file
from extract import groups,Encoder
from study import data,load_source,ridge,cw,metrics,compare,summary,h,SOURCE,KEY_PERM,select_head
from joint import scale,join
from prepare import ROOT,LABELS,save
from uncertainty import interval

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    bank=groups('screen');fit=join(data(bank['fit'],384,'context'),data(bank['edit'],384,'context'))
    sel=data(bank['select'],384,'context');sets={'select':sel,'confirm':data(bank['confirm'],384,'context')}
    y=np.eye(7)[fit['y']];float_scale={k:scale(fit[k]) for k in ['e','m','k']};trials=[];cands={}
    midw,midb,midselect=select_head(fit,sel,'m')
    for mg in [.5,1.5,4.]:
        for kg in [0.,.5,1.5]:
            X=np.column_stack([fit['e']/float_scale['e'],mg*fit['m']/float_scale['m'],kg*fit['k']/float_scale['k']])
            for balance in [0.,.65]:
                for reg in [.03,.3,1.5]:
                    W,b=ridge(X,y,cw(fit['y'],balance),reg,True)
                    we=W[:768]/float_scale['e'];wm=mg*W[768:1536]/float_scale['m'];wk=kg*W[1536:]/float_scale['k']
                    S=sel['e']@we+sel['m']@wm+sel['k']@wk+b;score=metrics(sel['y'],S.argmax(1))['macro_f1_7']
                    cfg=(mg,kg,balance,reg);trials.append([score,*cfg]);cands[cfg]=(we,wm,wk,b)
    edit=max([v for v in trials if v[2]>0],key=lambda c:c[0]);z=max([v for v in trials if v[2]==0],key=lambda c:c[0])
    we,wm,wk,b=cands[tuple(edit[1:])];ze,zm,_,zb=cands[tuple(z[1:])]
    lift=la.pinv(we,rtol=1e-6,check_finite=False);delta=(wk@lift).T.astype(np.float32)
    src=load_source();new=(src+delta).astype(np.float32);real=new-src;keyw=real.astype(float).T@we
    out=ROOT/'middle-screen';out.mkdir(exist_ok=True)
    save_file({SOURCE:new,'late.weight':np.ascontiguousarray(we.T),'middle-12.weight':np.ascontiguousarray(wm.T),'bias':b,
               'zero.late.weight':np.ascontiguousarray(ze.T),'zero.middle-12.weight':np.ascontiguousarray(zm.T),'zero.bias':zb},str(out/'edit.safetensors'))
    fr={'protocol':'3-view centered ridge fitted on the same source fit + edit cases. Tune only on original train selector. The external linear readout samples both layer 12 and layer 24. The last native MLP is physically replaced to supply extra evidence from its keys.',
        'scope':'supplemental, devised after earlier shared16 occurred; the shared labels do not enter training or tuning.',
        'tensor':{'source':SOURCE,'original':summary(src),'updated':summary(new),'delta':summary(real),'changed_entries':int(np.count_nonzero(real)),
                  'relative_delta':float(np.linalg.norm(real)/np.linalg.norm(src))},
        'selected':edit,'selected_reoptimized_zero':z,'cfg_columns':['selector macro F1','mid strength','key strength','class balance','ridge'],
        'selection_grid':trials,'middle_only_head':midselect,'fit_cases':len(np.unique(fit['case'])),'fit_rows':len(y)}
    save(out/'freeze.json',fr)
    sets['shared16']=data(groups('common')['common'],384,'context')
    with np.load(ROOT/'common16-predictions.npz') as q:old=q['incumbent'];np.testing.assert_array_equal(sets['shared16']['row'],q['source_csv_row'])
    rep={'role_order':LABELS,'results':{}}
    for name,d in sets.items():
        base=d['e']@we+d['m']@wm+b
        middle=(d['m']/np.sqrt(np.mean(d['m']**2,axis=1,keepdims=True)+1e-8)@midw+midb).argmax(1)
        preds={'edited':(base+d['k']@keyw).argmax(1),'matched_zero':base.argmax(1),'reoptimized_zero':(d['e']@ze+d['m']@zm+zb).argmax(1),
               'pure_middle_head':middle,'equal_norm_shuffle':(base+d['k']@keyw[KEY_PERM]).argmax(1)}
        if name=='shared16':preds['incumbent']=old
        r={k:metrics(d['y'],v) for k,v in preds.items()};r['delta_vs_reoptimized_zero']=interval(d['case'],d['y'],preds['reoptimized_zero'],preds['edited'])
        rep['results'][name]=r;np.savez_compressed(out/(name+'-predictions.npz'),case=d['case'],row=d['row'],start=d['start'],end=d['end'],ix=d['ix'],gold=d['y'],**preds)
        print(name,{k:round(v['macro_f1_7'],5) for k,v in r.items() if 'macro_f1_7' in v},flush=True)
    enc=Encoder()
    with torch.no_grad():enc.model.layers[23].mlp.down_proj.weight.copy_(torch.from_numpy(new))
    item=bank['fit'][0];d=data([item],384,'context');actual=enc.run(item,384,'context')
    predicted=d['e']@we+d['m']@wm+b+d['k']@keyw;observed=actual['e']@we+actual['m']@wm+b
    rep['physical_verification']={'case':item['id'],'max_logit_difference':float(np.max(abs(predicted-observed))),
             'all_argmax_match':bool(np.array_equal(predicted.argmax(1),observed.argmax(1))),
             'actual_tensor_sha256':h(enc.model.layers[23].mlp.down_proj.weight.detach().numpy())}
    save(out/'report.json',rep)
if __name__=='__main__':main()
