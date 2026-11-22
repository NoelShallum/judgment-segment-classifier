"""Produce a machine-readable summary of observed runs with immutable artifact references."""
import json
import fcntl
import os
from pathlib import Path
import numpy as np
from compare import evaluate
from fetch import ROOT,MODEL,sha,REV
def ref(p):
 p=Path(p)
 return {'path':str(p.resolve()),'sha256':sha(p)}
def _create_locked():
 plan=json.loads((ROOT/'plan.json').read_text());src=json.loads((ROOT/'source.json').read_text())
 doc={
  'model':'fastino/GLiDE-0.4B','source_url':'https://huggingface.co/fastino/GLiDE-0.4B/tree/'+REV,'source_revision':REV,
  'license':'Fastino Community License. Conditions include commercial-production agreement above USD 10M annual revenue and attribution for commercial use. See publisher LICENSE.',
  'source_files':{'encoder':src['files']['model-00001-of-00001.safetensors'],'readout':src['files']['readout.safetensors']},
  'source_tensor_elements':{'encoder':394781696,'decision_readout':261120,'total':395042816},
  'source_dtype':'bfloat16','native_options_in_order':plan['labels'],'gradient_steps':0,'source':ref(ROOT/'source.json'),
  'cohort_specification':ref(ROOT/'plan.json'),'controls':['untouched option tensor and untouched internal projection','shuffled training roles with identical fit algorithm and class mix'],
  'manipulations':{'readout':{'filename':'readout.safetensors','tensor':'weight','shape':[255,1024],'modified':'rows 0–6 using centered ridge on native option hidden states; rows 7–254 unchanged'},
                   'internal':{'filename':'model-00001-of-00001.safetensors','tensor':'model.layers.27.mlp.Wo.weight','shape':[1024,2624],'modified':'rank at most six before bfloat16 write, native option head frozen; additive map fitted from the final feed-forward input and the pre-normalization residual. Rounding can increase numerical rank.'}},
  'input_modes':{'alone':'one complete target sentence inside publisher seven-choice prompt','local':'one previous and one following sentence from same judgment around an explicitly marked target','packed-ordered':'shared fixed option list, at most eight consecutive sentences, one answer cue per sentence; original means unchanged weights under this new prompt.','packed-rotated':'matched sentence-token blocks, cyclically rotated, evaluated with the ordered weights','packed-alone':'matched packed-format singleton, fitted using the same 357 rows'},
  'official_test':'Untouched in this lane. All validation results use the validation CSV and the listed 16 judgment IDs.',
  'float32_smoke':ref(ROOT/'smoke-fp32.json'),'evaluation_metric':'pooled seven-class macro-F1 including zero for class with no support and no predictions; transfer to full benchmark is unverified.',
  'runs':{}
 }
 for f in sorted((ROOT/'weights').glob('*/*/*/validation.json')):
  wd=f.parent;backend,arm,tier=wd.parts[-3:]
  if not (wd/'paired.json').exists():evaluate(backend,arm,tier)
  validation=json.loads(f.read_text());paired=json.loads((wd/'paired.json').read_text());frozen=json.loads((wd/'receipt.json').read_text())
  proofs=[ref(wd/n) for n in ('receipt.json','validation.json','paired.json','predictions.json','runtime-proof.json','physical-checkpoint.json','natural-prior-frozen.json','natural-prior-validation.json','trust-frozen.json','trust-validation.json','ov-runtime-proof.json','discriminant-frozen.json','discriminant-validation.json') if (wd/n).exists()]
  score={}
  for method,res in validation['method'].items():
   other=paired['models'][method]
   score[method]={'macro_f1':res['metrics']['macro_f1'],'accuracy':res['metrics']['accuracy'],'per_role_f1':res['metrics']['per_role_f1'],'support':res['metrics']['support'],
                  'fixes_vs_original':res['fixed_vs_original'],'breaks_vs_original':res['broken_vs_original'],
                  'fixes_vs_incumbent':other['fixed_vs_incumbent'],'breaks_vs_incumbent':other['broken_vs_incumbent'],
                  'paired_case_bootstrap_95_vs_original_macro_f1':other['paired_macro_f1_diff_vs_native_95'],
                  'paired_case_bootstrap_95_vs_incumbent_macro_f1':other['paired_macro_f1_diff_vs_incumbent_95']}
  doc['runs'][f'{backend}/{arm}/{tier}']={'actual_cpu_or_gpu_precision':backend,'fit_case_ids':frozen['training_judgments'],'fit_sentences':frozen['train_sentences'],
      'fit_support':frozen['support'],'confirmation_case_ids':plan['val_judgment_ids'],'validation_sentences':validation['val_sentences'],
      'validation_position_selection':('All nonblank sentences in all sixteen shared judgments' if tier=='whole16' else ('First four seed-ranked indices in every shared judgment' if tier=='match64' else f"First {plan[tier]['validation_rows_per_shared_judgment']} seed-hash-ranked indices for every shared case")),
      'input_token_length':{'median':validation['median_tokens'],'maximum':validation['max_tokens']},
      'matched_incumbent':paired['incumbent'],'scores':score,'weights':frozen['files'],'evidence':proofs}
  supplements={}
  if (wd/'natural-prior-validation.json').exists():
   aux=json.loads((wd/'natural-prior-validation.json').read_text())
   supplements['corpus_priors_in_readout']={k:v for k,v in aux.items() if k!='predictions'}
  if (wd/'trust-validation.json').exists():
   aux=json.loads((wd/'trust-validation.json').read_text())
   supplements['10pct_frobenius_limit']={key:{k:v for k,v in x.items() if k!='predictions'} for key,x in aux.items()}
  if (wd/'discriminant-validation.json').exists():
   supplements['direct_discriminant_readout']=json.loads((wd/'discriminant-validation.json').read_text())
  doc['runs'][f'{backend}/{arm}/{tier}']['prespecified_supplementary_writes']=supplements
 if (ROOT/'text-overlap.json').exists():doc['case_and_literal_text_overlap_audit']=ref(ROOT/'text-overlap.json')
 if (ROOT/'content-plan.json').exists():doc['content_pooling_frozen_plan']=ref(ROOT/'content-plan.json')
 if (ROOT/'content-cue-traincv-plan.json').exists():doc['single_head_context_cue_train_only_selection_plan']=ref(ROOT/'content-cue-traincv-plan.json')
 doc['prospective_accelerator_constraint']='After the initial Intel FP32 studies the user restricted GPU work to NVIDIA T4. The CUDA content worker checks nvidia-smi before loading weights. Older Intel results are historical measurements. Current on-machine continuation is CPU-only.'
 if (ROOT/'glide-content-colab.zip').exists():doc['t4_content_worker']=ref(ROOT/'glide-content-colab.zip')
 if (ROOT/'attention-plan.json').exists():doc['conditional_attention_write_plan']=ref(ROOT/'attention-plan.json')
 doc['terminal_attention_preflights']=[ref(f) for f in (ROOT/'attention-write/implementation-check.json',ROOT/'attention-write/cpu-probes/diagnostic.json',ROOT/'attention-write/probe-per-head.json',ROOT/'attention-write/training-diagnostic.json',ROOT/'attention-write/train-per-head.json',ROOT/'attention-write/frozen.json',ROOT/'attention-write/validation.json') if f.exists()]
 if (ROOT/'packed-crossed-context.json').exists():doc['crossed_input_vs_weight_format']=ref(ROOT/'packed-crossed-context.json')
 doc['readout_explanation_studies']=[{'name':f.parent.name,'scores':ref(f),'frozen_fit':ref(f.with_name('fit.json'))} for f in sorted((ROOT/'readout-study').glob('*/validation.json'))]
 proofs=[ref(p) for p in sorted((ROOT/'openvino').glob('parity*.json'))]
 doc['precision_controls']={'torch_native_smoke_vs_openvino':proofs,'int8':'Native BF16 source weights, torch dynamic INT8 conversion for encoder linear layers except last MLP output projection, with FP32 embedding/normalization/final MLP projection/readout. Corrections rounded to donor bfloat16 and read back. The original and both corrected arms use exactly the same backend.'}
 doc['precision_controls']['matched_same_edited_tensor_backend_comparisons']=[ref(x) for x in sorted(ROOT.glob('*-backend-comparison.json'))]
 doc['precision_controls']['adaptive_unlabeled_quantization_repair']=[{'validation':ref(x),'frozen_fit':ref(x.with_name('frozen.json'))} for x in sorted((ROOT/'weights/int8').glob('*/*/quant-repair/validation.json'))]
 if (ROOT/'packed-complementarity.json').is_file():doc['secondary_incumbent_fusion_and_oracle']=ref(ROOT/'packed-complementarity.json')
 if (ROOT/'weights/ov-GPU-f32/packed-ordered/whole16/predictions.json').is_file():
  from positions import generate
  from sensitivity import evaluate as text_sensitivity
  doc['packed_scores_excluding_literal_text_reuse']=ref(text_sensitivity())
  doc['packed_position_and_label_invariance_results']=ref(generate())
  doc['packed_pre_registration']=ref(ROOT/'packed-full-plan.json')
 out=ROOT/'lane-summary.json';tmp=out.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(doc,indent=2));tmp.replace(out)
 print('summary',len(doc['runs']),flush=True)
def create():
 with (ROOT/'.lane-summary.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  return _create_locked()
if __name__=='__main__': create()
