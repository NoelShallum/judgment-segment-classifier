"""Physical native tensor replacement, persisted state hashes and forward parity."""
from collect import P, load_model, collect_case, S, torch, np
from score import KEYS, native, extra, merge_rows, OUT, hsh
import json
from safetensors.torch import load_file

m=load_model();schema=m.create_schema().entities(S).build()
fixture=[('Counsel for the appellant submits that the hearing had been adjourned.',3),('The respondent denies that any notice was issued.',4),('We therefore dismiss the appeal.',6)]
base=collect_case(m,schema,fixture); outputs={}; orig={k:v.detach().clone() for k,v in m.named_parameters() if k in KEYS}
for method in ('edit','shuffled'):
    deltas=load_file(str(OUT/(method+'-delta.safetensors')));replacement=load_file(str(OUT/(method+'-replacement.safetensors')))
    with torch.inference_mode():
        for k in KEYS:
            param=m.get_parameter(k); param.copy_(replacement[k]);assert torch.equal(param,orig[k]+deltas[k]);assert hsh(param.numpy())==hsh(replacement[k].numpy())
    physical=collect_case(m,schema,fixture); actual=native(physical);calculated=native(base)
    for i,k in enumerate(KEYS):calculated+=merge_rows(base,extra(base,deltas[k].numpy(),i))/3
    err=float(abs(calculated-actual).max()); assert err<2e-4,err
    outputs[method]=dict(max_absolute_logit_error=err,argmax_match=bool(np.all(calculated.argmax(1)==actual.argmax(1))),native_labels=native(base).argmax(1).tolist(),stored_replacement_labels=actual.argmax(1).tolist(),tensor_sha256={k:hsh(m.get_parameter(k).detach().numpy()) for k in KEYS})
    with torch.inference_mode():
        for k in KEYS:m.get_parameter(k).copy_(orig[k])
(OUT/'forward-parity.json').write_text(json.dumps(outputs,indent=2));print(outputs,flush=True)
