"""Additional decision pages derived from the actual seven-role profile and input records."""
def add(parts):
    from comparison_brief import rd,page,hdr,txt,tbl,refs,pct,f1,ci,NAMES,URL
    URL.update({'Hierarchical donor':'https://huggingface.co/L-NLProc/LegalSeg_Hier_BiLSTM-CRF/tree/837c2a85ce9a6df21c9e3ef6d1b908b23b66d985/Hier_BiLSTM-CRF',
        'Authors’ source':'https://github.com/ShubhamKumarNigam/LegalSeg/tree/af1b45c52cab22a71ddea0bf13c987741b16f042',
        'Transformer donor':'https://huggingface.co/L-NLProc/LegalSeg_ToInLegalBERT/tree/18886ca80c318a80c1cca32a094f7bd4675e052b'})
    h=rd('hierarchical-validation')
    if not h:return parts
    names={'gemini':'Gemini Decision 1','associative_direct':'BERT A, associative','associative_context':'BERT B, associative',
           'hier_parent':'Hierarchy, original','hier_edited':'Hierarchy, edited','combined_excerpt':'BERT + hierarchy','internal_mlp':'BERT A, internal edit','cue_head':'BERT A + written cues','semantic_edited':'Gemma, role-name readout'}
    groups=['pilot','confirmation','paragraphs','homogeneous'];sets=[next(iter(rd('gemini-seven-'+k)['results'].values())) for k in groups]
    def counts(k):
        return [str(s['metrics'][k]['correct'])+'/'+str(s['rows']) if k in s['metrics'] else '—' for s in sets]
    def tab(k,m):return [k,pct(m['token_accuracy']),f1(m['exact_role_boundaries']['f1']),f1(m['exact_labeled_spans']['f1'])]
    results=[]
    for k,n in names.items():
        vals=counts(k)
        if any(x!='—' for x in vals):results.append([n]+vals)
    d=h['document'];e=h['excerpt'];cl=rd('gemini-seven-hier-spans');qa=rd('hierarchical-implementation-qa')
    pm=page('What happens on each input?',
        'Each number counts exact matches to the reference dominant role. The two sentence draws and the two paragraph constructions answer different questions.',
        tbl(['System','Balanced sentences','One per case','Multi-statement row','Joined same-role rows'],results,(.29,.17,.16,.19,.19)),
        txt('The same source bytes and seven choices went to both systems within each column. The first two groups have sentence labels. The third has one original annotation on a long source row. Every constituent row in the fourth has the same label. A dash means that run was outside the evaluated set.'),
        txt('The recurring comparison for a single excerpt is the 109.49M-parameter associative BERT edit. The directly edited 16.82M-parameter hierarchy also runs fully offline, including on complete judgments. Each architecture was selected on validation before its paired answers were scored. Later proposed families remain exploratory on these cohorts.'),
        refs('Official splits','Hierarchical donor','Gemini choice API'))
    pages=[pm]
    rows=[]
    for k,title in [('select','Development (10 cases)'),('confirm','Confirmation (10 cases)'),('test','Held-out test (20 cases)')]:
        v=d['groups'][k];rows.append([title,f1(v['parent']['exact_labeled_spans']['f1']),f1(v['programmed']['exact_labeled_spans']['f1']),pct(v['parent']['token_accuracy']),pct(v['programmed']['token_accuracy'])])
    tr=d['groups']['test']
    pages.append(page('A direct edit to document transitions',
        'This public model reads words in both directions within a unit and reads units in both directions across the judgment. A CRF then assigns the sequence of roles.',
        tbl(['Complete judgments','Parent span F1','Edited span F1','Parent token acc.','Edited token acc.'],rows,(.30,.17,.17,.18,.18)),
        txt(f'We selected one of {len(d["candidates"])} direct transition programs on exact labeled-span F1. The winner multiplied all 49 role-to-role scores by {d["inner_multiplier"]:.2f} and retained the original start and end scores. Less persistence allowed more true transitions to appear. The test comprised {tr["rows"]:,} recorded units from 20 cases. The 95% paired span-F1 gain interval was {ci(tr["parent_gap_ci_token_accuracy_exact_f1"][1])} percentage points.'),
        txt('For an isolated sentence or paragraph, the same architecture treats that one input as both the start and the end of a judgment. Zeroing its seven starting and seven ending weights gave the best excerpt profile among 28 programs. That edit changed 14 values. Its weighted confirmation accuracy moved from '+pct(e['groups']['confirm']['parent']['accuracy'])+' to '+pct(e['groups']['confirm']['programmed']['accuracy'])+'.'),
        hdr('Source checkpoint and implementation'),
        txt('The published archive contains the vocabulary and all parameters. The source samples random recurrent states at each forward pass. Our inference uses zero initial states in both the parent and the edits.'+(' On '+str(qa['upstream_parity']['rows'])+' labeled rows, the authors’ implementation with the same zero states reproduced every local role.' if qa else '')),
        refs('Hierarchical donor','Authors’ source','LegalSeg paper')))
    frag=rd('fragment-validation');fc=rd('gemini-seven-fragment-spans')
    if frag:
        q=frag['groups']['test'];m=q['programmed'];rows=[]
        for name,title in [('select','Development'),('confirm','Confirmation'),('test','Fresh test')]:
            x=frag['groups'][name]
            rows.append([title,str(x['cases']),f1(x['parent']['exact_labeled_spans']['f1']),f1(x['programmed']['exact_labeled_spans']['f1']),pct(x['programmed']['token_accuracy'])])
        pages.append(page('A passage cut from inside a judgment',
            'An excerpt beginning halfway through a case has different opening and ending cues. This experiment draws four to nine consecutive rows inside each case and hides their original boundaries.',
            tbl(['Source split','Cases','Original span F1','Edited span F1','Edited token acc.'],rows,(.26,.12,.21,.21,.20)),
            txt(f'We sampled {q["cases"]} fresh test cases without a role quota. They cover {q["rows"]} original rows and include {q["multi_role_cases"]} passages with multiple gold roles. The selected intervention multiplies interior transitions by {frag["internal_crf_multiplier"]:.2f} and the original start/end scores by {frag["edge_crf_multiplier"]:.2f}. Its 95% case-bootstrap interval for the exact-span improvement is {ci(q["gap_ci_token_accuracy_exact_f1"][1])} percentage points.'),
            txt('The released fragment interface lets callers specify that an input is clipped. A complete-judgment upload uses the document profile. Both return the same offset format and preserve all input characters.'),
            txt('The 12 passages used for the cloud comparison were separately selected to contain a transition and to be suitable for external processing. Their performance is reported independently. The 120-case draw provides a fresh local check of the change.'),
            refs('Official splits','Hierarchical donor','Gemini choice API')))
    if cl:
        inputs=[('Gemini Decision 1',cl['gemini']),('Whole-judgment hierarchy',cl['local'])]
        b=rd('gemini-seven-spans')
        if b:inputs.append(('BERT sequence selection',b['local']))
        if fc:inputs.append(('Interior-passage hierarchy',fc['local']))
        pages.append(page('Where Gemini leads on raw passages',
            'Twelve source passages from distinct judgments received the identical raw text and 64 automatically proposed units for both providers and all local models.',
            tbl(['System','Token accuracy','Transition F1','Exact labeled-span F1'],[tab(n,m) for n,m in inputs],(.40,.20,.17,.23)),
            txt('Gemini saw the entire passage. Each typed question identified one candidate unit and asked for one of seven roles. Adjacent equal roles were merged for all systems. A correct role near the right place earns no exact-span credit if either end is off. Token accuracy gives partial credit for the text covered correctly.'),
            txt('The whole-case hierarchy and Gemini agreed on '+str(cl['agreement_units'])+' of '+str(cl['proposed_units'])+' unit labels. The paired 95% interval for their token-accuracy difference is '+ci(cl['paired_document_gap_95pct'][0])+' percentage points. With these passage lengths and this sample, the cloud result is stronger.'),
            refs('Official splits','Gemini choice API')))
    official=rd('hierarchical-official-test')
    if official:
        r=official['row_classification'];names2=[('excerpt_parent','One nonblank excerpt · original'),('excerpt_programmed','One nonblank excerpt · edited'),('known_rows_parent','Complete case, named row cuts · original'),('known_rows_programmed','Complete case, named row cuts · edited'),('author_filtered_parent','Author-filtered rows · original'),('author_filtered_programmed','Author-filtered rows · edited')]
        arr=[[n,f'{r[k]["rows"]:,}',pct(r[k]['accuracy']),f1(r[k]['macro_f1'])] for k,n in names2];al=official['named_release_alignment'];rw=official['raw_full_case_segmentation']
        pages.append(page('Every official test judgment',
            'The complete local hierarchy was run on all 712 test cases. Row classification and finding exact role spans from an unmarked case have different metrics.',
            tbl(['Hierarchy input and profile','Rows','Accuracy','Macro-F1'],arr,(.54,.15,.16,.15)),
            txt(f'On all reconstructed raw judgments, the edited model covered {rw["programmed"]["tokens"]:,} tokens with {pct(rw["programmed"]["token_accuracy"])} token accuracy, {f1(rw["programmed"]["exact_role_boundaries"]["f1"])} transition F1 and {f1(rw["programmed"]["exact_labeled_spans"]["f1"])} exact labeled-span F1. The parent exact-span F1 was {f1(rw["parent"]["exact_labeled_spans"]["f1"])}.'),
            txt(f'The named public prediction file aligned on {al["author_filtered_rows"]:,} case-indexed rows after the authors’ token filters. The zero-state donor reproduced {pct(al["fraction"])} of those saved predictions. The reported paper-level 0.77 macro-F1, the named prediction file and these local inference runs are distinct reference points.'),
            refs('Official splits','Hierarchical donor','LegalSeg paper','Authors’ source')))
    fuse=rd('fusion-validation');cue=rd('cue-program-validation');graft=rd('graft-validation');mlp=rd('internal-mlp-validation')
    tests=[]
    if graft:tests.append(['Replace selected BERT classifier rows or biases',str(graft['number_of_variants']),pct(graft['confirmation_parent']['accuracy']),pct(graft['confirmation']['accuracy']),f1(graft['confirmation']['macro_f1'])])
    if cue:
        m=cue['scores']['confirm'];tests.append(['Add 25 authored discourse cues',str(len(cue['candidates'])),pct(m['parent']['accuracy']),pct(m['written']['accuracy']),f1(m['written']['macro_f1'])])
    if fuse:
        x=fuse['excerpt'];m=x['groups']['confirm'];tests.append(['Combine BERT and hierarchical features',str(len(x['candidates'])),pct(m['bert']['accuracy']),pct(m['combined']['accuracy']),f1(m['combined']['macro_f1'])])
    if mlp:
        x=mlp['groups']['confirm'];tests.append(['Write inside the final BERT feed-forward layer',str(len(mlp['candidates'])),pct(x['parent']['accuracy']),pct(x['programmed']['accuracy']),f1(x['programmed']['macro_f1'])])
    pages.append(page('Which weight changes survived validation?',
        'These programs use preset edits and score them on validation. The confirmation set has 160 rows per role from separate cases, reweighted to the source role mix.',
        tbl(['Edit family','Variants','Parent acc.','Edited acc.','Edited macro-F1'],tests,(.39,.12,.15,.15,.19)),
        txt('The classifier graft can copy a few speaker-specific biases even when its parent is the strongest control. The cue layer writes signed weights for phrases that identify argument, reasoning or disposition. Small gains on one metric can coincide with losses on another.'),
        txt('The combined readout concatenates 768 BERT and 200 recurrent features. The selected excerpt coefficient on BERT was '+(str(fuse['excerpt']['bert_fraction']) if fuse else 'pending')+'. The complete-judgment search assigned zero weight to BERT, so the deployable whole-case result needs only the smaller recurrent model.'),
        txt('The transformer-over-BERT repository refers to 225 serialized storage entries. Its pinned public release exposes 100. We can score the authors’ named predictions. That release does not supply all of the tensors needed to reconstruct the model.'),
        refs('Published weights','Hierarchical donor','Transformer donor','Associative memory code')))
    pages.append(page('What runs on a laptop?',
        'Model size and input contract determine which comparison applies to a product decision.',
        tbl(['Model','Stored parameters','FP32 or quantized weights','Developer input → output'],[
            ['Edited hierarchy','16.82M','67.27 MB + 3.18 MB vocabulary','Sentence or paragraph → one label. Whole case or clipped text → spans.'],
            ['Edited InLegalBERT','109.49M','437.97 MB, FP32','Sentence or paragraph → one label. Multi-unit text → spans.'],
            ['Combined readout','126.30M donor parameters','Both parents, approximately 505 MB','Single excerpt → one label.'],
            ['Gemma 4 E2B readout','4.65B stored in GGUF','4.97 GB Q8_0 + 43 KB head','Single excerpt → one of seven outputs.'],
            ['Gemini Decision 1','Provider managed','Cloud API','One choice per question, with optional document context.']],(.23,.21,.23,.33)),
        txt('The hierarchical files in the archive run offline after dependency installation. The BERT profiles reconstruct from a public checksum-pinned donor. The portable release includes role names, exact profile hashes and excerpts’ recorded predictions.'+(f' On an i5-8250U with one PyTorch thread, 20 warm hierarchical excerpt calls took a median {qa["median_warm_ms_20"]:.1f} ms and p95 {qa["p95_warm_ms_20"]:.1f} ms while other benchmarks were running.' if qa else '')),
        txt('Gemini Decisions evaluates choices and produces no generated prose. It can label an independently authored synthetic example. The near-perfect synthetic score measures recognition of explicit patterns. A second court or jurisdiction with this seven-role taxonomy would require a separately authorized, comparable gold corpus.'),
        refs('Hierarchical donor','Published weights','Quantized Gemma weights','Gemini choice API')))
    return parts[:1]+pages+parts[1:]
