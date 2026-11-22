"""Generate the short evidence table from completed, locked scores."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent;res=p/'results'
def read(name):return json.loads((res/name).read_text())
f=read('freeze.json');win=f['selected'];s=['# GLiNER2.5 role-edit measurements','','The FP32 checkpoint contains 73,881,879 parameters. The frozen edit changes all 81,920 values in three pre-existing weight matrices. Each additive update has numerical rank six. The supervised fit used 1,011 sentences in 12 judgments and used direct numerical linear algebra without backpropagation.','','| Checkpoint tensor | Dimensions | Updated values | Edit Frobenius norm | Shuffled-control norm |','|---|---:|---:|---:|---:|']
for k,v in f['tensors'].items():s.append(f"| `{k}` | {' × '.join(map(str,v['shape']))} | {v['changed_entries']:,} | {v['frobenius']:.4f} | {v['null_frobenius']:.4f} |")
s+=['',f"The five training-selector judgments contained {f['select_rows']} rows. The selected joint solve at λ = {win['reg']} and gain {win['gain']:g} scored {win['macro_f1']:.4f} macro F1, versus {f['selector_unchanged']:.4f} for unchanged weights. The selected normal-equation relative residual was {f['joint_solver'][str(win['reg'])]['relative_normal_equation_residual']:.5f} after {f['joint_solver'][str(win['reg'])]['iterations']} numerical iterations.",'']
if (res/'forward-parity.json').exists():
 q=read('forward-parity.json');s +=[f"Reloading the published checkpoint and replacing its actual tensors reproduced the cached logits on a three-sentence fixture within {q['edit']['max_absolute_logit_error']:.2e} for the selected edit and {q['shuffled']['max_absolute_logit_error']:.2e} for the shuffled control. All argmax labels matched.",'']
s+=['| Completed cohort | Method | Rows | Sentence macro F1 | Sentence accuracy | Exact labeled span F1 |','|---|---|---:|---:|---:|---:|']
for fname,label in [('check','Internal training-split check'),('four','First four shared validation judgments'),('sixteen','All sixteen shared validation judgments')]:
 if not (res/(fname+'.json')).exists():continue
 a=read(fname+'.json');
 for k,v in a['aggregate'].items():s.append(f"| {label} | {k} | {v['rows']:,} | {v['macro_f1']:.4f} | {v['accuracy']:.4f} | {v['exact_span_micro_f1']:.4f} |")
if (res/'sixteen.json').exists():
 a=read('sixteen.json');m=a['aggregate'];s+=['','## Paired uncertainty on sixteen judgments','','| Edit minus comparator | Macro F1 difference | 95% judgment-bootstrap interval |','|---|---:|---:|'];
 for k,metrics in a['paired_edit_minus'].items():
  lo,hi=metrics['macro_f1'];s.append(f"| {k} | {m['edit']['macro_f1']-m[k]['macro_f1']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
if (res/'sixteen-span-headroom.json').exists():
 a=read('sixteen-span-headroom.json');stats=a['measures'];arms=('incumbent','edit','edit_insert','edit_relocate','edit_oracle_insert','oracle_relocate');s+=['','## Repairing the incumbent spans','','The short-run experiment accepts each proposed span of up to three sentences lying strictly within an incumbent run. The separate relocation experiment moves each incumbent boundary by up to two sentences according to the GLiNER scores. The oracle rows use the human labels to measure available headroom.','','| Method | Sentence macro F1 | Exact span precision | Exact span recall | Exact span F1 | Short span recall |','|---|---:|---:|---:|---:|---:|']
 for arm in arms:
  m=stats[arm];v=m['all_exact'];q=m['short_exact'];s.append(f"| {arm} | {m['macro_f1']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {q['recall']:.4f} |")
 s+=['','| GLiNER weights | Proposals | Correct proposals | Eligible missed gold spans |','|---|---:|---:|---:|']
 for arm,detail in a['short_run_diagnostics'].items():
  n=detail['proposals'];s.append(f"| {arm} | {n['predicted']} | {n['matches']} | {detail['eligible_missed_gold']} |")
 lo,hi=stats['edit_insert']['gain_over_incumbent_ci95']['all_exact']['f1'];x,y=stats['edit_relocate']['gain_over_incumbent_ci95']['all_exact']['f1'];s+=['',f'Paired 95% case-bootstrap intervals for exact span F1 change were [{lo:+.4f}, {hi:+.4f}] for GLiNER-edited insertions and [{x:+.4f}, {y:+.4f}] for GLiNER-edited relocation.']
if (res/'publisher-classify-64.json').exists():
 a=read('publisher-classify-64.json');s+=['','## Publisher API on isolated sentences','','A separate, preselected sample includes 16 noncontiguous sentences from each of the first four judgments. The publisher API receives one sentence at a time. The span decoder and incumbent use their existing contextual predictions on those same row IDs.','','| Method | Sentence macro F1 | Sentence accuracy |','|---|---:|---:|'];
 for k,v in a['results'].items():s.append(f"| {k} | {v['macro_f1']:.4f} | {v['accuracy']:.4f} |")
s+=['','All main methods receive the corpus sentence boundaries. Fits, selection and primary confirmation use different judgment IDs. The first four validation judgments are included in the sixteen-judgment cohort. Native denotes a compulsory seven-role decode of the native span marginals. The labeled exact-span metric requires matching both endpoints and the role of a contiguous run of sentences. The incumbent is the unchanged lane from the shared case manifest.',''];(p/'RESULTS.md').write_text('\n'.join(s))
