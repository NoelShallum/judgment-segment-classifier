from pathlib import Path
p=Path('extract.py')
s=p.read_text().replace("if isinstance(layer,torch.nn.Linear):","if isinstance(layer,torch.nn.Linear) and layer is not model.model.layers[-1].mlp.Wo:")
s=s.replace("captured=[]\n def hook(_,args):\n  captured.append(args[0].detach().float().cpu().numpy()[0].copy())\n handle=model.readout.register_forward_pre_hook(hook)",'''captured={}
 def capture(key,position=None):
  def hook(_,args):
   value=args[0][0] if position is None else args[0][0,position]
   captured[key]=value.detach().float().cpu().numpy().copy()
  return hook
 hooks=[model.readout.register_forward_pre_hook(capture('hidden')),
        model.model.final_norm.register_forward_pre_hook(capture('before_normalization',-2)),
        model.model.layers[-1].mlp.Wo.register_forward_pre_hook(capture('mlp_activation',-2))]''')
s=s.replace('finally: handle.remove()\n return captured[0],logits[:7],len(ids)',"""finally:
  for handle in hooks: handle.remove()
 return captured['hidden'],logits[:7],len(ids),captured['before_normalization'],captured['mlp_activation']""")
s=s.replace('h,z,n=encode(model,tokens,temp,text)','h,z,n,r,u=encode(model,tokens,temp,text)')
s=s.replace('int(n),str(d)) for i,h,z,n,d in zip(blob[\'idx\'],blob[\'hidden\'],blob[\'logit\'],blob[\'ntok\'],blob[\'text_sha256\'])','int(n),str(d),r,u) for i,h,z,n,d,r,u in zip(blob[\'idx\'],blob[\'hidden\'],blob[\'logit\'],blob[\'ntok\'],blob[\'text_sha256\'],blob[\'before_normalization\'],blob[\'mlp_activation\'])')
s=s.replace('text_sha256=[saved[i][3] for i in indexes])','text_sha256=[saved[i][3] for i in indexes],before_normalization=np.stack([saved[i][4] for i in indexes]),mlp_activation=np.stack([saved[i][5] for i in indexes]))')
s=s.replace('vec,z,n=encode(model,tokenizer,temp,text)\n    saved[pos]=(vec,z,n,h)', 'vec,z,n,r,u=encode(model,tokenizer,temp,text)\n    saved[pos]=(vec,z,n,h,r,u)')
p.write_text(s)
p=Path('cohort.py')
s=p.read_text().replace("'controls':['original readout'", """'internal_edit':'Independent last feed-forward weight edit model.layers.27.mlp.Wo.weight. Fit an additive rank-at-most-six map from the final MLP activation to residual class-logit errors using closed-form ridge and the pseudoinverse of the centered original readout through final LayerNorm. Native head stays fixed. Same ridge strength, weighting, and target logits as the head edit. Prespecified shuffle control.',
  'controls':['original readout'""")
p.write_text(s)
