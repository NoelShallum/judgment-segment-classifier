"""Freeze the research stocktake at the user's pause, using completed measurements and published sources."""
from urllib.parse import quote
def assemble(original):
    from comparison_brief import rd,page,hdr,txt,tbl,refs,pct,f1,ci,ROLE_NAMES,URL
    archive=rd('all-published-predictions')['files'];pub={x['file']:x for x in archive}
    def linkfile(key):return quote(pub[key]['url'],safe=':/%')
    URL.update({'Standalone predictions':linkfile('InLegalBert(i)_predictions.csv'),
                'Context predictions':linkfile('InLegalBert(i-1, label_p, i).csv'),
                'ToInLegalBERT predictions':linkfile('ToInlegalbertpredictions.csv'),
                'LegalSeg models':'https://huggingface.co/collections/L-NLProc/legalseg-models',
                'SeqTNS paper':'https://aclanthology.org/2025.findings-ijcnlp.50/',
                'MARRO paper':'https://cse.iitkgp.ac.in/~abhijnan/papers/bambroo_rhetorical_roles_AI_Law.pdf',
                'LegalEval task paper':'https://aclanthology.org/2023.semeval-1.318/',
                '2019 seven-role corpus':'https://arxiv.org/abs/1911.05405',
                'Isaacus':'https://isaacus.com/','Microsoft Presidio':'https://microsoft.github.io/presidio/'})
    P={p['title']:p for p in original}
    off=rd('hierarchical-official-test');h=rd('hierarchical-validation');doc=off['raw_full_case_segmentation'];row=off['row_classification']
    fresh=rd('stocktake-standalone-alignment');pause=rd('pause-snapshot');ceiling=rd('identical-inputs-ceiling')['test']
    cohorts=[next(iter(rd('gemini-seven-'+k)['results'].values())) for k in ('pilot','confirmation','paragraphs','homogeneous')]
    sen,con,para,join=cohorts
    d=con['paired_with_gemini']['internal_mlp'];sources=rd('published-baselines');named=sources['named_hierarchical']['score_on_exactly_aligned_docs']
    title=page('Judgment-role classifier stocktake',
        'AttoFlow research pause · 22 November 2026. The product accepts English text from an Indian judgment. A sentence or paragraph gets one of seven roles. A longer upload can return the character range of each role.',
        tbl(['Human-labeled check','Internal BERT edit','Edited hierarchy','Gemini Decision 1'],
            [[n]+[f'{c["metrics"][k]["correct"]}/{c["rows"]}' for k in ('internal_mlp','hier_edited','gemini')] for n,c in zip(['Balanced sentences','One sentence per judgment','Multi-statement source rows','Joined same-role rows'],cohorts)],(.39,.21,.21,.19)),
        txt(f'The internal BERT edit is one correct answer ahead of Gemini on 196 independent judgments and one behind on the 112-sentence balanced pilot. Its 196-case accuracy difference has a paired 95% interval of {ci(d["accuracy_gap_95pct"])} percentage points. Both missed more than half the human sentence labels.'),
        hdr('The clearest repeatable gain'),
        txt(f'On all 712 raw test judgments, halving 49 role-transition weights moved exact labeled-span F1 from {f1(doc["parent"]["exact_labeled_spans"]["f1"])} to {f1(doc["programmed"]["exact_labeled_spans"]["f1"])} and token accuracy from {pct(doc["parent"]["token_accuracy"])} to {pct(doc["programmed"]["token_accuracy"])}. Supplied the original sentence boundaries, the edited model scored {f1(row["known_rows_programmed"]["macro_f1"])} macro-F1 against LegalSeg’s reported 0.770.'),
        txt('The stocktake stops at the last completed comparisons. The evidence supports a document-boundary improvement. Sentence parity with Gemini and a new research state of the art remain unestablished.'),
        refs('LegalSeg paper','Official splits','Gemini choice API'))
    same=page('The new BERT against standalone LegalSeg',
        'The authors released predictions for their standalone InLegalBERT. We matched the existing local predictions to those records, retaining only inputs whose entire original text matches the local input byte for byte.',
        tbl(['System','Balanced set correct','Macro-F1','One per case correct','Macro-F1'],
            [[label, f'{fresh["pilot"]["systems"][k]["correct"]}/{fresh["pilot"]["rows"]}',f1(fresh['pilot']['systems'][k]['macro_f1']),f'{fresh["confirmation"]["systems"][k]["correct"]}/{fresh["confirmation"]["rows"]}',f1(fresh['confirmation']['systems'][k]['macro_f1'])] for k,label in [('archived_standalone','LegalSeg released predictions'),('internal_mlp','AttoFlow internal BERT edit'),('associative_direct','AttoFlow associative head'),('gemini','Gemini Decision 1')]],(.32,.18,.14,.22,.14)),
        txt('All 112 balanced excerpts matched exactly. Two of the 196 case-sampled excerpts changed during input cleanup, leaving 194 for the archive comparison. The internal edit was two answers ahead on the balanced set and nine behind on the 194. Its 95% paired accuracy-gap intervals were '+ci(fresh['pilot']['systems']['internal_mlp']['paired_gap_accuracy_macro_f1_ci_95'][0])+' and '+ci(fresh['confirmation']['systems']['internal_mlp']['paired_gap_accuracy_macro_f1_ci_95'][0])+' percentage points, respectively. The authors’ prediction file contains labels only. Its complete gold-label sequence matches all 149,881 official rows in order.'),
        hdr('Why archived and reloaded results differ'),
        txt('The Hub files named for the current sentence and for the previous gold role share SHA-256 bbe8af20…e7c18cea. Loading those bytes produced 68.5% agreement with the separately released standalone answers on a fixed 1,792-row sample. On those rows the archive scored 0.491 macro-F1, the reloaded parent 0.362, and our centroid edit 0.410. The checkpoint that produced the archive remains unverified.'),
        txt('Across all 149,868 nonblank test rows, archived standalone InLegalBERT scored '+pct(fresh['all_nonblank_standalone_archive']['accuracy'])+' accuracy and '+f1(fresh['all_nonblank_standalone_archive']['macro_f1'])+' macro-F1. The standalone edited hierarchy scored '+pct(row['excerpt_programmed']['accuracy'])+' and '+f1(row['excerpt_programmed']['macro_f1'])+'. The newer internal BERT edit has no full-test score at this pause.'),
        refs('Standalone predictions','Published weights','Official splits'))
    def rr(n,k):
        x=pub[k];return [n,f'{x["rows"]:,}',pct(x['accuracy']),f1(x['macro_f1'])]
    context=page('The prior research benchmark',
        'All systems in this comparison predict a role for each supplied sentence with access to judgment context. The official test contains 149,881 rows. The last two rows use the same author-filtered subset.',
        tbl(['Document-context system and evidence','Rows','Accuracy','Macro-F1'],[
            ['LegalSeg Hier-BiLSTM-CRF · paper Table 3','Paper test','62%','0.770'],
            rr('InLegalBERT with previous predicted role · archive','InLegalBert(i-1, label_p, i).csv'),
            rr('TransformerOverInLegalBERT · archive','ToInlegalbertpredictions.csv'),
            rr('InLegalBERT with two previous sentences · archive','InLegalBert(i-2, i-1, i)_predictions.csv'),
            rr('InLegalBERT with left and right sentences · archive','InLegalBert(i-1, i, i+1)_predictions.csv'),
            rr('GNN · archive','GNN_predictions.csv'),
            ['AttoFlow zero-state hierarchy · donor','149,881',pct(row['known_rows_parent']['accuracy']),f1(row['known_rows_parent']['macro_f1'])],
            ['AttoFlow transition edit','149,881',pct(row['known_rows_programmed']['accuracy']),f1(row['known_rows_programmed']['macro_f1'])],
            ['LegalSeg hierarchy · named prediction file','149,825',pct(named['accuracy']),f1(named['macro_f1'])],
            ['AttoFlow transition edit · same named rows','149,825',pct(row['author_filtered_programmed']['accuracy']),f1(row['author_filtered_programmed']['macro_f1'])]],(.55,.15,.15,.15)),
        txt(f'The local transition edit trails the paper headline by {(.77-row["known_rows_programmed"]["macro_f1"])*100:.1f} macro-F1 points, ToInLegalBERT’s released predictions by {(pub["ToInlegalbertpredictions.csv"]["macro_f1"]-row["known_rows_programmed"]["macro_f1"])*100:.1f}, and the named hierarchical release by {(named["macro_f1"]-row["author_filtered_programmed"]["macro_f1"])*100:.1f} on its exact filtered rows. The highest aligned public prediction file we scored gives 0.633 and is named for the previous predicted role. LegalSeg Table 3 lists 0.48 for that variant. The archive score describes the released outputs.'),
        txt('The paper’s hierarchical headline, the per-case prediction file and the reloaded public tensors are three distinct observations. No matching published exact character-span F1 was identified for our reconstructed raw-text task. Its 0.257 measures a different output.'),
        refs('LegalSeg paper','Context predictions','ToInLegalBERT predictions','Hierarchical donor'))
    other=page('How other research compares',
        'Related systems include typed decision models, sequential classifiers and multitask attention models. Corpus, role inventory and access to surrounding sentences determine whether a score transfers.',
        tbl(['Published system','Reported result','What the comparison establishes'],[
            ['TypeSafe Jev · Šuppa & Kopčan','0.520 macro-F1','149,875 English LegalSeg sentences, with the whole judgment in the input. Closest independent study.'],
            ['SeqTNS · T. D. et al.','0.83 macro-F1','Separate seven-role corpus from Bhattacharya et al. The classes include statute and precedent. The authors also report 0.78 weighted F1 on 13 roles.'],
            ['MTL-MARRO · Bambroo et al.','0.724 India, 0.617 UK macro-F1','Separate corpora of 150 Indian and 50 UK cases, five-fold cross-validation. Argument roles and role inventories differ from LegalSeg.'],
            ['AntContentTech · LegalEval 2023','0.8593 weighted F1','Shared-task winner on 13 roles. Weighted F1 gives frequent roles more weight than seven-class macro-F1.']],(.30,.21,.49)),
        txt('AttoFlow’s document-context edit scored 0.567 macro-F1 across the official rows. Jev’s 0.520 uses the same underlying benchmark with six romanised-Hindi sentences removed. The 0.047 numerical difference is a cross-study comparison. The models have no paired comparison on those exact 149,875 rows here. Jev and Gemini Decision 1 are separate systems.'),
        txt('The SeqTNS, MARRO and LegalEval results describe the research landscape. At this pause all active local model scores use the seven-role LegalSeg corpus or explicitly authored synthetic diagnostics.'),
        refs('Independent Jev evaluation','SeqTNS paper','2019 seven-role corpus','MARRO paper','LegalEval task paper'))
    models=page('The gradient-free experiments',
        'Each edit starts from released weights. Authored keys specify desired local behavior. Validation chooses among fixed interventions and disjoint cases measure whether the gain persists.',
        tbl(['Method and scope','Held-out finding'],[
            ['BERT centroid and prior · 20 settings','42 authored examples. Weighted confirmation accuracy 49.9% → 50.3%, macro-F1 0.405 → 0.445.'],
            ['Associative head writes · nine settings per parent','84 authored keys. A improved again to 51.4% / 0.454. Context-trained B declined from 57.0% / 0.497 to 56.1% / 0.495.'],
            ['Interpolate whole BERT checkpoints · six variants','The parent trained with two preceding sentences won. The weight midpoint had no confirmed advantage.'],
            ['BERT head grafts · 388 variants','Tried whole and role-group rows and biases. Copying two speaker biases gave 56.8% / 0.491 versus parent 57.0% / 0.497.'],
            ['Authored lexical cue weights · six gains','25 phrase features. Confirmation macro-F1 0.454 → 0.460, accuracy 51.4% → 51.3%.'],
            ['Final BERT feed-forward projection · 36 programs','An 84-key, rank-at-most-83 write changed the internal 768 × 3,072 projection and bias. Confirmation fell to 50.3% / 0.449. Paired sentence pilots gained one and two answers over the associative head.'],
            ['BERT plus hierarchical features · 14 excerpt choices','The selected 968-feature readout used 70% BERT B. Confirmation reached 56.9%. The 112 document programs selected zero BERT weight.'],
            ['Gemma 4 E2B · 12 A–G readouts','Copied existing vocabulary rows, adjusted bias and wrote output associations. The edit and original each got 9/56 in confirmation and 21/112 on the real balanced pilot.']],(.41,.59)),
        txt('The confidence intervals, raw scores and exact hashes are available in the evidence archive. The internal BERT experiment was introduced after earlier pilot aggregates had been examined. Neither pilot provides an independent confirmation of the choice of that model family.'),
        refs('Published weights','Associative memory paper','Quantized Gemma weights'))
    breakdown=page('What the 196-case near-tie contains',
        'The 196-case sample draws one eligible sentence per case without using a role quota. The earlier balanced pilot has 16 examples per class from 98 different judgments.',
        tbl(['Role','Internal BERT correct / 16','Gemini correct / 16'],[[n,str(sen['metrics']['internal_mlp']['confusion'][i][i])+'/16',str(sen['metrics']['gemini']['confusion'][i][i])+'/16'] for i,n in enumerate(ROLE_NAMES)],(.43,.31,.26)),
        txt(f'On 196 sentences both systems were correct on {d["both_correct"]}, only the BERT edit was correct on {d["only_local_correct"]}, only Gemini was correct on {d["only_gemini_correct"]}, and both were wrong on {d["both_wrong"]}. They returned the identical role on {d["seven_label_agreement_count"]}/196. Their macro-F1 scores were {f1(con["metrics"]["internal_mlp"]["macro_f1"])} and {f1(con["metrics"]["gemini"]["macro_f1"])}. No issue-labeled sentence appeared in that draw.'),
        txt('On the balanced 112, their macro-F1 scores were '+f1(sen['metrics']['internal_mlp']['macro_f1'])+' and '+f1(sen['metrics']['gemini']['macro_f1'])+'. On 168 fictional examples with explicit cues, Gemini matched 167 intended roles and the earlier centroid edit matched 111. The synthetic and real results answer different questions.'),
        txt(f'{ceiling["raw_inputs_with_incompatible_labels"]:,} distinct nonblank test strings recur with conflicting human roles. A deterministic sentence-only system must miss at least {ceiling["minimum_errors_on_nonblank_rows"]:,}/{ceiling["nonblank_rows"]:,} rows, even with a perfect lookup table. Its empirical accuracy ceiling is {ceiling["empirical_maximum_accuracy"]*100:.2f}%. Whole-judgment input can distinguish these occurrences.'),
        refs('Official splits','Gemini choice API'))
    history=page('Earlier controls and work paused',
        'These entries account for fitted comparators, exploratory runs and pending measurements. They matter when interpreting how much of a reported gain came from direct editing.',
        tbl(['Earlier comparison','Completed evidence'],[
            ['fastText lexical baseline','Fit on 1,122,507 training rows. On the full test it reached 47.6% accuracy and 0.400 macro-F1.'],
            ['Closed-form discriminant classifier','Estimated from 1,792 labeled training examples. Weighted validation confirmation was 52.2% / 0.449. This is a fitted statistical comparator.'],
            ['Gemini-screened synthetic augmentation','167 of 168 original examples passed agreement review. Adding screened examples to the fitted control moved confirmation to 53.5% / 0.459. Gemini supplied a choice on authored text.'],
            ['MiniLM fine-tuning','Stopped early and checkpoint preserved when the study returned to weight editing. No completed held-out result accompanies that checkpoint.'],
            ['Earlier external-corpus probes','A four-role paragraph pilot and an India/UK transfer probe preceded the seven-role-only instruction. They are excluded from all stocktake comparisons.']],(.35,.65)),
        hdr('State at the pause'),
        txt(f'The official BERT feature sweep had saved {pause["bert_official_cached_rows"]:,}/{pause["bert_official_total_rows"]:,} rows in {pause["bert_official_cached_blocks"]}/{pause["bert_official_total_blocks"]} verified-manifest blocks. A semantic role-name alternative to the Gemma letter prompt had {pause["semantic_decoder_saved_vectors"]}/{pause["semantic_decoder_expected_vectors"]} initial vectors. The 305 fresh sentence candidates have no Gemini requests or results. The full seven-role hierarchy evaluation is complete.'),
        txt('Colab CLI 0.8.0 was installed. This work used the local Intel i5-8250U CPU and integrated GPU. The Colab account had no active CLI login at the pause. Training, model inference and API benchmarking are paused.'),
        refs('Official splits','Gemini choice API'))
    scope=page('The broader AttoFlow study',
        'The companion workbook contains 25 open-model proposals in nine domains, their order, inputs, outputs, sizes, data, baselines and a weight-editing decision for each.',
        tbl(['Opportunity','Developer contract','What the research established'],[
            ['Commerce · canonical value linker','Merchant field and raw value plus an allowed vocabulary → canonical value or abstention','Held-out merchant dictionary matched 59.1% of 3,488 scored aligned values. An existing retrieval competitor makes the unseen-value benchmark important.'],
            ['Legal · judgment rhetorical roles','Sentence or paragraph → one role. A longer judgment → role spans','The present stocktake measures separate excerpt, paragraph and raw-document tasks on seven-role LegalSeg.'],
            ['Developer tools · repository issue labels','Issue title, body and repository label descriptions → ranked or multiple labels','The catalog describes public GitHub data, existing automation, a cross-repository holdout and a laptop-size readout. This was an opportunity study.']],(.22,.35,.43)),
        txt('The original landscape report also covers Horizon Labs, the clinical weight-space audit, the associative-memory paper and implementation, Isaacus legal models, Presidio’s recognizers and Gemini Decisions. The workbook specifies the intended experiments for the remaining domains. The trained donors, prototype edits and live comparisons detailed here concern the legal project.'),
        txt('All reports were produced from recorded results or cited publications. The kit includes the complete CPU-ready hierarchical model, BERT reconstruction instructions, input provenance and anonymized prediction records.'),
        refs('Horizon Labs','Clinical weight-space audit','Associative memory paper','Isaacus','Microsoft Presidio'))
    return [title,same,context,other,P['What happens on each input?'],breakdown,models,P['A direct edit to document transitions'],P['Every official test judgment'],P['A passage cut from inside a judgment'],P['Where does each role begin and end?'],P['Where Gemini leads on raw passages'],P['One label for a paragraph'],P['A seven-row instruction-model head'],P['What runs on a laptop?'],history,scope]
