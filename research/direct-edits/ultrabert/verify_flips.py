"""Fresh before/after native PyTorch forwards on a training snippet with a nontrivial effect."""
import json,os
import numpy as np
import torch
from safetensors.numpy import load_file
from extract import Encoder,groups
from study import SOURCE,data,h,load_source,norm
from joint import position,decode
from prepare import ROOT,save,LABELS
def main():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')));torch.set_num_interop_threads(1)
    gs=groups('screen')['fit'];ref=load_source();enc=Encoder()
    native=enc.model.layers[23].mlp.down_proj.weight
    output={'rule':'First existing fit snippet (predefined training order) with one or more changed decoded labels. Fit example displayed only to verify the tensor effect.',
            'source_tensor':SOURCE,'source_unmutated_sha256':h(ref),'role_order':LABELS,'methods':{}}
    for kind,path in [('normalized_residual','write-screen-context-384/edit.safetensors'),('joint_text','joint-screen-context-384/text.safetensors'),('joint_position','joint-screen-context-384/position.safetensors')]:
        ck=load_file(str(ROOT/path));W=ck['classifier.weight'].T;bias=ck['classifier.bias'];changed=ck[SOURCE]
        def predict(d,e):
            if kind=='normalized_residual':return (norm(e)@W+bias).argmax(1)
            return decode(d,e@W+position(d)@ck['position.weight'].T+bias,ck['transition.log_probabilities'],float(ck['transition.gain'][0]))
        delta=(changed-ref).astype(float)
        for item in gs:
            d=data([item],384,'context')
            p0=predict(d,d['e']);p1=predict(d,d['e']+d['k']@delta.T)
            if np.any(p0!=p1):break
        else:raise ValueError(kind+' never changed a fit snippet')
        with torch.no_grad():native.copy_(torch.from_numpy(ref))
        b=enc.run(item,384,'context')
        with torch.no_grad():native.copy_(torch.from_numpy(changed))
        a=enc.run(item,384,'context')
        actual0=predict(d,b['e']);actual1=predict(d,a['e'])
        assert np.array_equal(actual0,p0) and np.array_equal(actual1,p1)
        idx=np.flatnonzero(p0!=p1)
        output['methods'][kind]={'id':item['id'],'source_rows':d['row'].tolist(),'tensor_after_runtime_copy_sha256':h(native.detach().numpy()),
                                'actual_frozen_base_predictions':actual0.tolist(),'actual_edited_predictions':actual1.tolist(),
                                'original_csv_labels':d['y'].tolist(),'flipped_rows':d['row'][idx].tolist(),
                                'case_char_offsets_for_flips':list(zip(d['start'][idx].tolist(),d['end'][idx].tolist())),
                                'exact_argmax_match_with_physical_forwards':True,'checkpoint':path}
        save(ROOT/'native-before-after-flips.json',output)
        print(kind,item['id'],d['row'][idx].tolist(),actual0.tolist(),actual1.tolist(),flush=True)
if __name__=='__main__':main()
