import json
from pathlib import Path
from fetch import ROOT,sha
p=json.loads((ROOT/'plan.json').read_text())
m={'purpose':'Confirm a different decision-model input format with eight ordered sentences and independent sentence-level answer cues. All model candidates, contexts and metrics frozen here before any packed validation inference.',
   'source_plan_sha256':sha(ROOT/'plan.json'),'validation_manifest_sha256':p['source']['shared_manifest'],'fixed_fit_ids':p['fit_judgment_ids'],
   'fixed_validation_ids':p['val_judgment_ids'],'fit_size':357,'expected_full_validation_nonblank_rows':3558,
   'full_coverage':'score every nonblank row of all sixteen shared judgments. The earlier 64-sentence native-prompt pilot is contained within the 3,558. This is shared validation, not an official test result.',
   'fit_choice':'12 predeclared complete short official training judgments, fixed before packed results. Use the original and two prescribed direct writes (fixed ridge lambda 0.15, target margin 4, inverse-square-root frequency weights) plus matched shuffled-role controls. No tuning or SGD.',
   'inputs':'Current chunk contains up to 8 consecutive sentences and a single fixed preamble of seven allowable choices. A fixed answer cue follows each sentence. Only original judgment text and fixed strings are tokenized. No corpus label, previous predicted choice, or decision answer is input.',
   'group_size':'Greedy native token limit 512 for normal blocks, counting the prefix and all answer cues. One exceptionally long sentence uses its own block up to the model cap with both ends retained.',
   'checks':['Serialize each input/cue array and repeat after replacing every input-record label with a .invalid-label marker. Assert the sequence, target membership and cue offsets are identical for each case.',
             'Keep nonblank per-case offsets. Verify each target appears once in a full replay. Count role frequencies and support; compare to the frozen incumbent on the same 3,558 positions.',
             'Same-template singleton control: fit on the same 357 rows, score the 64 preselected positions with exactly one sentence and one answer cue per request.',
             'Order/position control: rotate the same sentence-token chunks by half their sentence count, preserving exact sentence sets, token budget, and readout cue per sentence. Apply the already-frozen ordered weights to the 64 fixed positions. Rotation moves absolute slot and the context seam; interpret both.',
             'Report original-position and block-size strata with counts and role support, and a paired 16-judgment bootstrap. Interchanging layouts measures input sensitivity.'],
   'secondary_evaluation':'For secondary controls compare ordered vs rotated vs singleton on the exact same 64 preselected positions. The local-label sample contains no gold Issue rows, so compare per-role and full-coverage scores separately.',
   'none_option':'The fixed options include the seven possible class names as required for decision prompting. Changing gold labels must leave every input token unchanged.'}
(ROOT/'packed-full-plan.json').write_text(json.dumps(m,indent=2))
print(sha(ROOT/'packed-full-plan.json'))
