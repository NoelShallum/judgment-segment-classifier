"""Render the ongoing seven-role span research from completed score records."""
from html import escape
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/"outputs/judgment-role/weight-editing-research.html"
html=[]
def h(n,text):html.append(f"<h{n}>{escape(text)}</h{n}>")
def p(text):html.append(f"<p>{escape(text)}</p>")
def table(head,rows):
    html.append("<div class='table'><table><thead><tr>"+''.join(f"<th>{escape(x)}</th>" for x in head)+"</tr></thead><tbody>")
    for row in rows:html.append("<tr>"+''.join(f"<td>{escape(str(x))}</td>" for x in row)+"</tr>")
    html.append("</tbody></table></div>")
def links(*links):
    html.append("<p class='links'>"+' · '.join(f"<a href='{escape(url,quote=True)}'>{escape(name)}</a>" for name,url in links)+"</p>")
def result(name):
    p=ROOT/"results"/(name+".json")
    return json.loads(p.read_text()) if p.exists() else None
def role_edge(name):
    path=ROOT/"astra-role-edge"/(name+".json")
    return json.loads(path.read_text()) if path.exists() else None
def span(m):return f'{m["exact_labeled_spans"]["f1"]:.3f}'
def pct(x):return f"{100*x:.1f}%"
def metric(title,cohort,current,comparison,remarks):
    return title,cohort,span(current),span(comparison),remarks
def main():
    h(1,"Weight editing for judgment spans")
    p("AttoFlow compares CPU-sized models for seven rhetorical roles in English Indian judgments. Full judgments and clipped passages are the primary input types. Every span has a role and half-open Unicode character offsets. Exact span F1 requires both offsets and the role to match.")
    whole_discourse=result("discourse-span-weights")
    trace=result("archive-span-paired-comparison")
    role_edge_study=role_edge("results")
    role_edge_test=role_edge_study["splits"].get("test") if role_edge_study else None
    role_edge_published=role_edge("independent-replay")
    interpolation_qa=result("role-edge-interpolation-qa")
    segment_qa=result("segment-weight-qa")
    phrase_sequence_score=result("sentence-phrase-sequence")
    phrase_referral_score=result("sentence-phrase-sequence-referral")
    phrase_archive_score=result("sentence-phrase-span-archive-audit")
    pair_archive=result("sentence-pairwise-archive-audit")
    pair_replay=result("sentence-pairwise-gemini-replay")
    joint_pair_archive=result("sentence-joint-pairwise-span-archive")
    mixedbread_overlay=result("mixedbread-referral-overlay")
    bert_overlay=(mixedbread_overlay or {}).get("splits",{}).get("test",{}).get("arms",{}).get("BERT only")
    template_overlay=result("sentence-template-memory-bert")
    template_official=(template_overlay or {}).get("splits",{}).get("test",{}).get("candidate")
    wide_live=result("sentence-wide-bert-benchmark")
    wide_cli=result("sentence-wide-bert-cli-qa")
    wide_official=(wide_live or {}).get("splits",{}).get("test",{})
    wide_local=wide_official.get("comparison",{}).get("routes",{}).get("confidence",{}).get("labels")
    if phrase_sequence_score and phrase_referral_score and phrase_archive_score and segment_qa:
        official=phrase_referral_score["splits"]["test"]["arms"]["issue-protected hybrid"]
        span_profile=phrase_sequence_score["splits"]["test"]["arms"]["spans"]
        labels=pair_archive["local"] if pair_archive else official["labels"]
        leading=bert_overlay["labels"] if bert_overlay and bert_overlay["labels"]["macro_f1"]>labels["macro_f1"] else labels
        if template_official and template_official["macro_f1"]>leading["macro_f1"]:
            leading=template_official
        if wide_cli and wide_local and wide_local["macro_f1"]>leading["macro_f1"]:
            leading=wide_local
        exact_span=(joint_pair_archive["edited_scores"]["exact_labeled_spans"]["f1"] if joint_pair_archive
                    else span_profile["spans"]["exact_labeled_span_f1"])
        h(2,"Best results by input and metric")
        table(("Input","Output","Official test result","Reference"),[
            ("Full judgment with sentences supplied","One role per sentence",
             f'{leading["macro_f1"]:.5f} macro-F1 and {100*leading["accuracy"]:.2f}% accuracy',
             "LegalSeg paper reports 0.770 macro-F1"),
            ("Full judgment with sentences supplied","Contiguous labeled spans",
             f'{exact_span:.5f} exact span F1',
             f'{phrase_archive_score["published_projection"]["exact_labeled_spans"]["f1"]:.5f} for the strongest aligned author archive'),
            ("Raw full judgment","Labeled character spans",
             f'{segment_qa["scores"]["exact_labeled_spans"]["f1"]:.5f} exact span F1',
             "Earlier segment-exact weight edit")])
        p("All three results use the 712 official LegalSeg test judgments. The released predictions we could align did not reproduce the published macro-F1. The paired span comparison uses identical supplied sentence cuts.")
        if joint_pair_archive:
            lo,hi=joint_pair_archive["paired_change_from_public"][0]
            p(f'The span edit improves exact span F1 over the strongest aligned author archive. The paired 95% interval is {100*lo:+.2f} to {100*hi:+.2f} points.')
        if pair_archive:
            lo,hi=pair_archive["paired_delta"]["macro_f1"]
            p(f'The sequence-only classifier scores {labels["macro_f1"]:.5f} macro-F1 using direct role comparisons. Its paired macro-F1 difference from the strongest aligned public predictions is {100*lo:+.2f} to {100*hi:+.2f} points at 95% confidence. The earlier issue-protected hybrids are reported separately.')
            if leading is wide_local:
                p("The highest runnable official-test point estimate uses the wider confidence route at a 20% review budget. CPU parity was checked on three confirmation judgments.")
                original_referral=result("sentence-bert-referral")
                if template_official and original_referral:
                    narrow=original_referral["splits"]["test"]["referred"]
                    wide_count=wide_official["comparison"]["routes"]["confidence"]["referred"]
                    p(f'The wider route improves the previous {template_official["macro_f1"]:.5f} macro-F1 point estimate by {100*(wide_local["macro_f1"]-template_official["macro_f1"]):.4f} points and increases the number of BERT windows from {narrow:,} to {wide_count:,}.')
            elif leading is template_official:
                p("The highest local point estimate combines the frozen local InLegalBERT review with exact-sentence memory. The added memory improved the 384-judgment holdout. Its official-test macro-F1 interval includes zero.")
            elif leading is not labels:
                p("The highest official-test point estimate adds an InLegalBERT review to the sequence-only classifier. Its gain over that classifier was inconclusive on both independent validation cohorts.")
        else:
            p("The leading sentence classifier in this table includes a rule informed by official test errors.")
        links(("LegalSeg paper and evaluation protocol","https://aclanthology.org/2025.findings-naacl.63/"))
    density_qa=result("short-span-density-qa")
    raw_joint=result("raw-joint-direct-qa")
    transfer_test=(result("whole-joint-transfer-edit") or {}).get("splits",{}).get("test")
    raw_local=result("raw-local-transfer")
    if raw_local:
        h(2,"Transferring the local readout to raw judgments")
        p(f'The supplied-sentence weights were applied directly to the raw splitter intervals of all {raw_local["cases"]} official test judgments. The splitter and local readout weights stay fixed.')
        table(("Profile","Exact-span micro-F1","Token accuracy"),[
            ("Raw joint",f'{raw_local["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(raw_local["parent"]["token_accuracy"])),
            ("Raw local transfer",f'{raw_local["candidate"]["exact_labeled_spans"]["f1"]:.4f}',pct(raw_local["candidate"]["token_accuracy"]))])
        if raw_local["paired"]["exact_span_f1"][0]>0 and raw_local["paired"]["token_accuracy"][0]>0:
            p("The optional raw-local profile shares the supplied-sentence local weights. Segment-exact remains the stronger profile when exact span F1 is the priority.")
        for field,desc in (("exact_span_f1","exact-span F1"),("token_accuracy","token accuracy")):
            lo,hi=raw_local["paired"][field]
            p(f'The paired 95% difference in {desc} was {100*lo:+.2f} to {100*hi:+.2f} percentage points.')
    raw_verified=(raw_joint and transfer_test and raw_joint["cases"]==transfer_test["cases"]==712
                  and raw_joint["scores"]==transfer_test["standalone_sentence_joint"])
    raw_eligible=(raw_verified and raw_joint["paired_to_token_balanced"]["token_accuracy"][0]>0
                  and raw_joint["paired_to_role_edge_balanced"]["token_accuracy"][0]>0)
    if whole_discourse:
        best=whole_discourse["tasks"]["whole"]["profiles"]
        upright=best["exact"]["scores"]["test"]["candidate"]
        practical=best["balanced"]["scores"]["test"]["candidate"]
        h(2,"The full-judgment result")
        first=[["Discourse exact-span profile",span(upright),pct(upright["token_accuracy"])],
               ["Balanced local discourse profile",span(practical),pct(practical["token_accuracy"])]]
        if role_edge_test:
            first=[[f"Role-pair transition edit, {name}",span(t["candidate"]),pct(t["candidate"]["token_accuracy"])]
                   for name,t in role_edge_test.items()]+first
        if interpolation_qa:
            x=interpolation_qa["scores"]
            first.insert(1,["Interpolated role-pair profile",span(x),pct(x["token_accuracy"])])
        if segment_qa:
            x=segment_qa["scores"]
            first.insert(0,["Experimental segment-exact profile",span(x),pct(x["token_accuracy"])])
        if raw_eligible:
            x=raw_joint["scores"]
            first.insert(1,["Raw-joint token accuracy profile",f'{x["exact_labeled_spans"]["f1"]:.4f}',
                           f'{100*x["token_accuracy"]:.2f}%'])
        if trace:
            previous=trace["published_projection"]
            first.append(["Strongest archived projection, original cuts supplied",span(previous),pct(previous["token_accuracy"])])
        table(("Profile on 712 cases","Exact span F1","Token accuracy"),first)
        p("These are the seven LegalSeg roles. The quoted sentence-label benchmark is assessed separately, and the raw-span lead is limited to the tested comparisons.")
    core=result("hierarchical-official-test")["raw_full_case_segmentation"]
    frag=result("fragment-validation")["groups"]["test"]
    cloud=result("gemini-seven-spans")
    rows=[metric("Original 49-weight transition edit","712 complete test judgments",core["programmed"],core["parent"],"Same encoder and candidate boundaries"),
          metric("Original clipped-passage transition edit","120 held-out passages",frag["programmed"],frag["parent"],"Absolute improvement 0.110"),
          metric("Earlier edited BERT decoder","12 Gemini-matched passages",cloud["local"],cloud["gemini"],"Comparison is Gemini Decision 1")]
    for stem,tag,field in [("role-specific-transitions","Role-specific CRF shifts","candidate"),("duration-weights","Length-dependent CRF costs","candidate"),("contrastive-hierarchy-edit","Same-topic classifier write","candidate"),("preserved-emitter-edit","Retention-constrained classifier write","candidate")]:
        q=result(stem)
        if not q:continue
        for name,trial in q.get("tasks",q).items():
            test=trial.get("scores",{}).get("test")
            if test:
                units=test.get("cases",712 if name=="whole" else 120)
                cohort=f'{units} '+("complete test judgments" if name=="whole" else "clipped passages")
                rows.append(metric(tag,cohort,test[field],test.get("previous",test.get("baseline")),"Preset selected on development"))
    trade=result("transition-token-tradeoff")
    if trade:
        score=trade["scores"]["test"]
        rows.append(metric("Transition edit with token constraint","712 complete judgments",score["candidate"],score["previous"],
                           f'Token accuracy {pct(score["candidate"]["token_accuracy"])} versus {pct(score["previous"]["token_accuracy"])}'))
    sizes=result("context-size-validation")
    if sizes:
        for name,trial in sizes["spans"].items():
            score=trial["scores"]["test"]
            rows.append(metric("Context size comparison",f'{score["cases"]} {name} inputs',score["chosen"],score["full_context"],
                               f'{trial["selected_context_units"]} units, with 0 denoting full supplied text'))
    discourse=result("discourse-span-weights")
    if discourse:
        for name,trial in discourse["tasks"].items():
            for label,head in trial["profiles"].items():
                q=head["scores"]["test"]
                rows.append(metric(f"Discourse edit, {label}",f'{q["cases"]} {name} inputs',q["candidate"],q["incumbent"],
                                   f'Cue gain {head["cue_gain"]:g}, position gain {head["position_gain"]:g}'))
    cut=result("boundary-span-validation")
    if cut:
        for name,trial in cut.items():
            score=trial["results"]["test"]
            rows.append(metric("Boundary proposal "+trial["selected"],f'{score["cases"]} '+("complete judgments" if name=="whole" else "clipped passages"),score["candidate"],score["incumbent"],"Same CRF weights"))
    for model,name in [("minilm","MiniLM-L6"),("granite","Granite R2 Small"),("gte","GTE ModernBERT Base")]:
        q=result("encoder-span-"+model) or result("encoder-span-"+model+"-accelerated")
        if q:
            for task,trial in q["tasks"].items():
                for method,lab in [("best","validation selection"),("standalone_encoder","encoder-only emissions with a CRF")]:
                    m=trial[method];z=m["scores"]["test"]
                    rows.append(metric(f"{name}, {lab}",f'{z["cases"]} '+("complete judgments" if task=="whole" else "clipped passages"),z["candidate"],z["hierarchy"],
                                       f'{m["mode"]}; encoder fraction {m["new_encoder_fraction"]:g}'))
        for stem,kind in (("encoder-sparse-memory-","sparse associative memory"),("encoder-contextual-keys-","readout with contextualized role keys")):
            special=result(stem+model)
            if special:
                for task,trial in special["tasks"].items():
                    z=trial["best"]["scores"]["test"]
                    rows.append(metric(f"{name}, {kind}",f'{z["cases"]} {task} inputs',z["candidate"],z["hierarchy"],
                                       "Uses explicit authored role keys"))
        nr=result("encoder-norm-edit-"+model)
        if nr:
            for task,tt in nr["tasks"].items():
                m=tt["scores"]["test"]
                rows.append(metric(f"{name}, internal weight edit",task,m["edited"],m["parent"],
                                   f'{tt["changed_pretrained_parameters"]} pretrained normalization coefficients changed'))
    h(2,"Completed exact-span measurements")
    table(("Method","Evaluation set","Exact span F1","Comparison","Protocol"),rows)
    p("The full 712-case result and the smaller whole-case encoder screens have different sampling frames. The 120 clipped passages use the same case identities across the local follow-ups. The 12 screened passages form a separate API benchmark. Macro-F1 for classification at known sentence boundaries is a separate measure.")
    trade=result("role-specific-transitions")
    if trade and "whole" in trade["tasks"]:
        v=trade["tasks"]["whole"]["scores"]["test"]
        lo,hi=v["span_and_token_accuracy_gap_95pct"][0]
        p(f'The selected full-judgment transition edit gains {lo*100:.1f} to {hi*100:.1f} exact-span F1 points in a paired 95% interval. Token accuracy changes from {pct(v["previous"]["token_accuracy"])} to {pct(v["candidate"]["token_accuracy"])}. Validation selected for exact spans, so this score reflects a different accuracy trade-off.')
    role=result("transition-per-role-audit")
    if role:
        table(("Role","Earlier exact-span F1","Role-specific exact-span F1"),
              [(k.replace("_"," "),f'{v["f1"]:.3f}',f'{role["role_specific"]["exact_spans"][k]["f1"]:.3f}') for k,v in role["incumbent"]["exact_spans"].items()])
    if sizes:
        h(2,"Single-label contract")
        p("A standalone English sentence or paragraph returns one of seven classes. A full judgment or a passage with several roles can instead return a list of labeled spans. A paragraph with more than one rhetorical function can therefore be explored in span mode while the single-label interface retains the requested multiclass contract.")
        p("The compact BERT model has 512 total subword positions. With normal single-sequence encoding that leaves 510 positions for text per window. Its inference wrapper scores longer passages over overlapping windows and aggregates the scores. The 16.82M-parameter hierarchical model has no fixed BERT-style token window. It reads each candidate unit with a word-level LSTM and then the complete unit sequence with a document-level LSTM. The largest test judgment had 59,996 whitespace tokens. That is an observed input length.")
        table(("Source","Whole excerpt correct","Contextual-unit vote correct","Gemini correct"),[(k,
            f'{v["literal_excerpt"]["local_correct"]}/{v["literal_excerpt"]["rows"]}',
            f'{v["contextual_units_vote"]["local_correct"]}/{v["contextual_units_vote"]["rows"]}',
            f'{v["literal_excerpt"]["gemini_correct"]}/{v["literal_excerpt"]["rows"]}')
           for k,v in sizes["single_label_diagnostics"].items()])
        for k,trial in sizes["spans"].items():
            q=trial["scores"]["test"]
            table(("Context","{0} case exact span F1".format(q["cases"]),"Token accuracy"),
                  [[("Full supplied text" if width=="0" else width+" candidate units"),span(q["context_only"][width]),
                    pct(q["context_only"][width]["token_accuracy"])] for width in ("1","3","6","12","0")])
        p("Every context-width setting shares a sequence decoder. Width one means one unit per encoder call. The single-label upload comparison is a separate task.")
        p("The source label describes an intact official row or a paragraph constructed from same-role rows. Original typeset paragraph structure is absent from the LegalSeg CSV.")
    lengths=result("whole-length-strata")
    if lengths:
        h(2,"Full judgments by length")
        block=lengths["profiles"]["test"]
        table(("Whitespace tokens","Judgments","Exact profile span F1","Balanced profile span F1","Balanced token accuracy"),[
              (label,a["judgments"],f'{a["exact_f1"]:.3f}',f'{b["exact_f1"]:.3f}',pct(b["token_accuracy"]))
              for key,label in (("0","Fewer than 2,000"),("2000","2,000 to 7,999"),("8000","8,000 or more"))
              for a,b in [(block["exact"]["bands"][key],block["balanced"]["bands"][key])]])
        latency=result("whole-span-cpu-latency")
        if latency:
            table(("Validation input length","Automatic units","CPU time, median of three"),[
              (f'{q["whitespace_tokens"]:,}',q["model_units"],f'{q["median_warm_s"]:.2f} s') for q in latency["results"]])
            p("These illustrative times use the token-balanced discourse weights, already loaded, on an Intel Core i5-8250U with two PyTorch threads while other experiments were running.")
        p("The 712-case test has a median 4,371 whitespace tokens and a maximum of 59,996. This is a descriptive cohort breakdown. It does not measure a tokenizer's maximum length or establish why the scores fall on longer judgments.")
    error_detail=result("current-discourse-span-error-decomposition")
    if error_detail:
        h(2,"What an exact-span miss looks like")
        categories=(("exact","Exact match"),("unreachable_endpoint","At least one endpoint missing from the splitter"),
                    ("exact_bounds_wrong_role","Both predicted offsets match and the role differs"),
                    ("same_role_one_unit_away","Overlapping same-role prediction within one unit at both ends"),
                    ("same_role_other_overlap","Overlapping same-role prediction farther away"),
                    ("no_same_role_overlap","No overlapping same-role prediction"))
        blocks=error_detail["profiles"]
        table(("Gold span outcome","Exact profile","Balanced profile"),[
            (label, *[f'{block["all_spans"][key]:,} ({pct(block["all_spans"][key]/sum(block["all_spans"].values()))})'
                      for block in blocks.values()]) for key,label in categories])
        table(("Gold role","Gold spans","Exact profile precision","Exact profile recall","Balanced profile recall"),[
            (role.replace("_"," "),f'{sum(x.values())-x["predicted"]:,}',pct(x["exact"]/max(1,x["predicted"])),
             pct(x["exact"]/max(1,sum(x.values())-x["predicted"])),pct(y["exact"]/max(1,sum(y.values())-y["predicted"])))
            for role,x in blocks["exact"]["by_role"].items() for y in [blocks["balanced"]["by_role"][role]]])
        p("Each gold span is counted once, in the first category it satisfies. Available endpoints can still be missed. The near-miss rows show where a different decoder might help. They do not add to the measured exact-span F1.")
    entry=result("role-entry-diagnostic")
    if entry:
        h(2,"Separating role starts from role duration")
        e=entry["test"]
        variants=(("parent","Original"),("entry","Seven written role-entry costs"),("pooled_entry","One shared cost"),("diagonal","Diagonal transition control"))
        table(("Profile","Exact-oriented F1","Token accuracy","Balanced F1","Token accuracy"),
          [(title,span(e["exact"][k]["scores"]),pct(e["exact"][k]["scores"]["token_accuracy"]),
             span(e["balanced"][k]["scores"]),pct(e["balanced"][k]["scores"]["token_accuracy"])) for k,title in variants])
        p("Both selection cohorts picked zero edit gain. The nonzero setting was frozen separately as a diagnostic before its confirmation and test scores. The shared cost reached 0.2894, slightly ahead of the seven role-specific costs. The earlier discourse profile scored 0.286.")
    if role_edge_test:
        h(2,"Transitions conditioned on both neighboring roles")
        rows=[]
        cohorts=[(k,v) for k,v in role_edge_study["splits"].items() if k not in ("independent","test")]
        disjoint=role_edge("text-disjoint-validation")
        if disjoint:
            cohorts.append(("Additional text-disjoint validation",disjoint["results"]))
        cohorts.append(("Official test",role_edge_test))
        for split,records in cohorts:
            for name,q in records.items():
                lo,hi=q["paired_gap_95pct"]["exact_span_f1"]
                rows.append([split,name,q["cases"],span(q["candidate"]),span(q["parent"]),
                             pct(q["candidate"]["token_accuracy"]),pct(q["parent"]["token_accuracy"]),
                             f'{100*lo:+.2f} to {100*hi:+.2f} points'])
        table(("Cohort","Profile","Judgments","Exact span F1","Parent F1","Token accuracy","Parent accuracy","95% interval for F1 gain"),rows)
        p(f'Each boundary feature adds a written 7-by-7 score table to the decoder. The table uses frequencies from {role_edge_study["train"]["cases"]:,} official training judgments reconstructed with the same splitter as inference. The 16.82M-parameter encoder is paired with {role_edge_study["train"]["parameters"]:,} written edge coefficients. Validation chose the full interaction at gain 0.2, with zero gradient or optimizer steps.')
        table(("Control on full test","Exact profile span F1","Balanced profile span F1"),[
            (name.replace("_"," "),span(role_edge_test["exact"]["controls"][name]["scores"]),
             span(role_edge_test["balanced"]["controls"][name]["scores"])) for name in role_edge_test["exact"]["controls"]])
        role_detail=role_edge_test["exact"]
        table(("Role","Official spans","Role-pair span F1","Discourse parent span F1"),[
            (k.replace("_"," "),f'{v["gold"]:,}',f'{v["span_f1"]:.3f}',f'{role_detail["parent"]["roles"][k]["span_f1"]:.3f}')
            for k,v in role_detail["candidate"]["roles"].items()])
        new_latency=result("role-edge-cpu-latency")
        if new_latency:
            table(("Judgment words","Detected units","Role-pair CPU median","Discourse CPU median"),[
                (f'{r["whitespace_tokens"]:,}',r["candidate_units"],f'{r["medians"]["edited"]:.2f} s',
                 f'{r["medians"]["parent"]:.2f} s') for r in new_latency["cases"]])
            p("Three warm CPU runs per profile on each case include automatic boundary detection and offset output. Other experiment processes were running. These measurements describe the three sampled inputs.")
        if role_edge_published and "official_test" in role_edge_published:
            comp=role_edge_published["official_test"]["profiles"]
            table(("Profile","Fewer than 2,000 words","2,000 to 7,999 words","8,000 words or more"),[
                (name,*[span(p["by_length"][band]["scores"]) for band in ("under_2000","2000_to_7999","8000_or_more")])
                for name,p in comp.items()])
            table(("Role-pair profile","Exact span F1","95% interval for F1 gain over best archive"),[
                (name,span(q["scores"]),f'{100*lo:+.2f} to {100*hi:+.2f} points') for name,q in comp.items()
                for lo,hi in [q["paired_gap_against_archive_95pct"]["exact_span_f1"]]])
    frontier=[("Decoder-weight interpolation",result("role-edge-interpolation")),
              ("Feature-group scaling",result("role-edge-feature-mix")),
              ("Posterior span decoding",result("role-edge-posterior"))]
    frontier=[(name,trial) for name,trial in frontier if trial and "test" in trial["splits"]]
    if frontier:
        h(2,"Exact spans and token accuracy after the role-pair edit")
        table(("Method","Profile","Cohort","Cases","Exact span F1","Parent F1","Token accuracy","Parent accuracy","95% interval for F1 gain"),[
            (label,name,split,q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
             f'{q["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),
             pct(q["parent"]["token_accuracy"]),f'{100*lo:+.2f} to {100*hi:+.2f} points')
            for label,trial in frontier for split in ("confirm","independent","test")
            for name,q in trial["splits"][split].items() for lo,hi in [q["gap_ci95"]["exact_span_f1"]]])
        p("All candidates were frozen on 72 selection judgments. Two further validation cohorts contain 72 and 127 judgments. The interpolation physically combines the two existing sets of emission adjustments and transition weights. Feature scaling changes the relative contribution of words near the detected boundary, punctuation and position.")
        p("The posterior decoder compares plausible maximal role spans across eight proposed paths. It computes the probability of complete spans under the CRF, subtracts a span count cost and adds a token reward. This is an approximation to maximizing corpus F1. Its forward and backward probabilities and interval search matched exhaustive enumeration on small synthetic sequences. The test gain is small and its paired interval includes zero, so the downloadable exact profile continues to use the role-pair Viterbi decoder.")
        links(("F-measure optimization under structured uncertainty","https://jmlr.csail.mit.edu/papers/v15/waegeman14a.html"),
              ("Semi-Markov spans for legal rhetorical roles","https://arxiv.org/abs/2302.06448"))
    searched=result("role-pair-search")
    known_search=result("known-sentence-pair-edit")
    positional_search=result("role-pair-position-edit")
    if searched or known_search or positional_search:
        h(2,"Transition weights selected for each output format")
        p("The experiments change static role-pair weights and the relative contribution of lexical evidence for each pair. One method adjusts pair frequencies toward the annotated training counts. A second tries individual weight changes and scores the whole decoded sequence on the target metric. Both use the same 144 training judgments to derive weights and freeze the validation choice before test replay. They leave the neural encoder frozen and use zero gradient steps.")
        if searched:
            table(("Raw judgment cohort","Search objective","Selected edit","Cases","Exact-span F1","Parent F1","Token accuracy","Parent accuracy"),[
                (s,k,searched["profiles"][k]["name"],q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
                 f'{q["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),pct(q["parent"]["token_accuracy"]))
                for s,data in searched["splits"].items() for k,q in data.items()])
        if positional_search:
            method,gain=positional_search["choice"]
            p(f'The position study adjusts pair frequencies separately in four equally sized parts of each judgment. Validation selected the {method} correction at gain {gain:g}. The physical edit applies it through the existing position entries in the lexical table.')
            table(("Position-study cohort","Cases","Exact-span F1","Parent F1","Token accuracy","Parent accuracy"),[
                (s,q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
                 f'{q["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),
                 pct(q["parent"]["token_accuracy"])) for s,q in positional_search["splits"].items()])
        if known_search:
            table(("Supplied sentence cohort","Cases","Sentence macro-F1","Parent macro-F1","Sentence accuracy","Parent accuracy"),[
                (s,q["cases"],f'{q["candidate"]["macro_f1"]:.4f}',f'{q["parent"]["macro_f1"]:.4f}',
                 pct(q["candidate"]["accuracy"]),pct(q["parent"]["accuracy"])) for s,q in known_search["splits"].items()])
            p(f'The sentence-boundary study chose {known_search["choice"]} on development. It scores each original nonblank sentence with its judgment context. LegalSeg reports 0.770 macro-F1 in its strongest published sentence run.')
            sentence_qa=result("known-sentence-pair-edit-qa")
            if sentence_qa:
                q=sentence_qa["scores"]
                base=sentence_qa["parent_scores"]
                lo,hi=sentence_qa["gap_ci95_macro_f1_accuracy"][0]
                p(f'The downloadable sentence-context weights score {q["macro_f1"]:.3f} macro-F1 and {pct(q["accuracy"])} accuracy on {q["rows"]:,} nonblank test sentences. The paired 95% interval for macro-F1 gain over the preceding profile is {100*lo:+.2f} to {100*hi:+.2f} points. The API accepts an ordered JSON array and returns one role per supplied sentence.')
                table(("Role","Sentence-context F1","Preceding F1"),[
                    (role.replace("_"," "),f'{f1:.3f}',f'{base["per_role_f1"][role]:.3f}')
                    for role,f1 in q["per_role_f1"].items()])
        links(("Minimum error rate training","https://aclanthology.org/P03-1021/"),
              ("LegalSeg paper","https://aclanthology.org/2025.findings-naacl.63/"))
    segment_write=result("segment-weight-write")
    segment_freeze=result("segment-weight-frozen")
    if segment_write and segment_freeze and "test" in segment_write["splits"]:
        h(2,"Scoring complete spans from raw judgments")
        p(f'A new score estimates whether a whole span has the exact start, end and rhetorical role. A closed-form weight write uses {segment_freeze["train_candidates"]:,} candidate spans from {len(segment_freeze["train_cases"]):,} official training judgments, with {segment_freeze["train_exact_matches"]:,} exact matches. Inputs include the frozen CRF probability of the span, votes from eight proposed label sequences, the span length and position, and the model scores near the span boundaries.')
        p("The decoder chooses adjacent spans with different roles that cover the entire judgment. One candidate set takes spans directly from the eight sequences. The expanded set also moves one endpoint of a span to a neighboring cut proposed by another sequence.")
        chosen=segment_write["choice"]
        if chosen:
            _,extended,share,cost=chosen
            shape="expanded" if extended else "original"
            p(f'Validation selected the {shape} candidate set, gave the written readout {share:.0%} weight and subtracted a fixed cost of {cost:g} for each span.')
        else:
            p("Validation selected the released role-edge profile. Both candidate sets were measured on subsequent cohorts to quantify how many gold spans each could represent.")
        table(("Cohort","Cases","Exact span F1","Role-edge exact F1","Token accuracy","95% interval for F1 change"),[
            (cohort,q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
             f'{q["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),
             f'{100*q["gap_ci95"]["exact_span_f1"][0]:+.2f} to {100*q["gap_ci95"]["exact_span_f1"][1]:+.2f} points')
            for cohort,q in segment_write["splits"].items()])
        if segment_qa:
            p("The portable segment-exact profile uses the same 16.82M-parameter encoder and a 2.4 KB adapter. It reproduces the official test scores across all 712 judgments. Its gains were mixed on the two validation cohorts. The official estimate shows a small exact-span improvement and a lower token accuracy point estimate.")
            p("This comparison uses the public test previously replayed in other research stages. The paired interval describes uncertainty over judgments for the frozen comparison.")
        latency=result("segment-exact-cpu-latency")
        if latency:
            table(("Input words","Candidate units","Role-edge exact median","Segment-exact median"),[
                (f'{q["whitespace_tokens"]:,}',q["candidate_units"],f'{q["medians"]["role-edge exact"]:.2f} s',
                 f'{q["medians"]["segment-exact"]:.2f} s') for q in latency["cases"]])
            p("Timings include splitting, encoding and output offsets with two CPU threads after warm-up. Three runs per profile on each of three judgments provide an illustrative latency comparison.")
        q=segment_write["splits"]["test"]["candidate_oracles"]
        table(("Candidate set on official test","Gold spans present in set","F1 with gold indicators supplied"),[
            (label,pct(q[key]["exact_span_recall_ceiling"]),f'{q[key]["exact_hit_oracle"]["f1"]:.3f}')
            for key,label in (("False","Eight proposed sequences"),("True","With nearby endpoint changes"))])
        p("The gold-indicator replay picks a valid tiling with one point for each exact match and a quarter-point penalty per span. It uses the correct test offsets and labels to diagnose candidate coverage.")
        endpoint_write=result("segment-posterior-ends-write")
        endpoint_frozen=result("segment-posterior-ends-frozen")
        if endpoint_write and endpoint_frozen and "test" in endpoint_write["splits"]:
            h(3,"Adding endpoints omitted by the proposed sequences")
            p(f'A second experiment adds plausible boundaries using the probability that each role starts or ends at a raw-text unit. It proposes one ending per length range for each likely starting point. Training uses {endpoint_frozen["train_candidates"]:,} candidate spans with {endpoint_frozen["train_exact_matches"]:,} exact matches.')
            table(("Cohort","Cases","Exact span F1","Role-edge exact F1","Token accuracy","Candidate span recall"),[
                (cohort,v["cases"],f'{v["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
                 f'{v["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(v["candidate"]["token_accuracy"]),
                 pct(v["candidate_oracles"]["True"]["exact_span_recall_ceiling"]))
                for cohort,v in endpoint_write["splits"].items()])
            v=endpoint_write["splits"]["test"]
            a,b=v["gap_ci95"]["exact_span_f1"]
            p(f'The test gold-indicator replay reaches {v["candidate_oracles"]["True"]["exact_hit_oracle"]["f1"]:.3f} exact span F1. The selected scorer gave the learned readout full weight. The paired 95% interval for the change in F1 is {100*a:+.2f} to {100*b:+.2f} points.')
        links(("Filtered Semi-Markov CRF","https://aclanthology.org/2023.findings-emnlp.17/"))
        preference=result("segment-preference-edit")
        preference_frozen=result("segment-preference-frozen")
        if preference and preference_frozen and "test" in preference["splits"]:
            h(3,"Ranking competing spans and complete paths")
            c=preference_frozen["counts"]
            p(f'A closed-form residual compares {c["local_pairs"]:,} pairs of correct spans and competing spans with similar boundaries. It also compares {c["preferred_paths"]:,} judgment-level alternatives with more exact matches. The experiment compares pointwise, local preference and complete-path objectives, each using the frozen segment-exact scorer as its starting point.')
            choice=preference["choice"]
            if choice:
                modes={"point":"pointwise","local":"local preference","paths":"complete-path preference","joint":"joint local and complete-path preference"}
                objective,_,gain,cost=choice
                proposal="the expanded endpoint set" if cost is not None else "the original segment-exact candidate set"
                p(f'Validation selected the {modes[objective]} objective at residual scale {gain:g} using {proposal}.')
            else:
                p("Validation selected the segment-exact baseline.")
            table(("Cohort","Cases","Candidate span F1","Segment-exact F1","Token accuracy","95% interval for F1 change"),[
                (k,v["cases"],f'{v["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
                 f'{v["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(v["candidate"]["token_accuracy"]),
                 f'{100*v["gap_ci95"]["exact_span_f1"][0]:+.2f} to {100*v["gap_ci95"]["exact_span_f1"][1]:+.2f} points')
                for k,v in preference["splits"].items()])
            p("The local preference score improved both validation point estimates. The official test point estimate was slightly lower than segment-exact and the paired interval crossed zero.")
            links(("Pairwise regularized least squares for ranking","https://staff.cs.utu.fi/~aatapa/publications/inpPaTsAiBoSa07a.pdf"))
        coverage=result("segment-coverage-audit")
        if coverage:
            h(3,"Where correct spans fall out of the candidate set")
            p(f'The audit attributes missing gold spans in {coverage["cases"]} judgments to the first stage that excludes them. The last column gives the exact maximum corpus F1 attainable with each fixed candidate set when gold annotations are available to the search.')
            table(("Candidate set","Missing raw endpoint","Missing offset pair","Missing role for present pair","Exact span available","Maximum possible F1"),[
                (label,*[f'{q["counts"][key]:,}' for key in ("missing_raw_endpoint","offset_pair_missing_from_candidates",
                   "role_missing_for_present_offset_pair","exact_span_present")],f'{q["exact_corpus_f1_upper_bound"]["f1"]:.4f}')
                for label,q in zip(("Nearby endpoint changes","Posterior endpoints"),coverage["candidates"].values())])
            q=coverage["candidates"]["posterior_endpoints"]["roles"]
            table(("Role","Gold spans","Exact candidates present","Offset pair missing","Role missing for present pair"),[
                (role.replace("_"," "),f'{v["gold"]:,}',pct(v["exact_span_present"]/v["gold"]),
                 f'{v["offset_pair_missing_from_candidates"]:,}',f'{v["role_missing_for_present_offset_pair"]:,}')
                for role,v in q.items()])
        short_span=result("short-span-proposals-audit")
        if short_span:
            p("A separate proposal experiment adds every short interval through a fixed number of raw units. Each interval keeps the two or three roles with the highest frozen posterior. The table quantifies search headroom with gold annotations used only for scoring.")
            table(("Maximum short interval length","Roles per short interval","Candidate spans","Gold span coverage","Maximum possible F1"),[
                (q["max_units"],q["roles_per_interval"],f'{q["candidates"]:,}',pct(q["recall"]),f'{q["oracle"]["f1"]:.4f}')
                for q in short_span["runs"].values()])
        dense=result("short-span-density-edit")
        dense_frozen=result("short-span-density-frozen")
        if dense and dense_frozen and "test" in dense["splits"]:
            h(3,"Scoring the larger candidate set from raw text")
            p(f'The extended score uses a closed-form fit on {dense_frozen["candidates"]:,} candidate spans from {len(dense_frozen["train_cases"]):,} training judgments. It estimates an exact match, a role starting at the left boundary, and a role ending at the right boundary. The three scores use frozen model features and seven original roles.')
            choice=dense["choice"]
            if choice:
                _,_,tier,mix,boundary,cost,fee=choice
                width,roles=((4,2),(8,2),(16,2),(16,3))[tier-2]
                p(f'Validation selected short intervals up to {width} units and {roles} roles per interval. It assigned {mix:.0%} of the score to the new weights, with a {boundary:.0%} contribution from the boundary heads within that score. The span cost was {cost:g}, plus {fee:g} for a candidate outside the original set.')
            else:
                x=max(dense_frozen["grid"],key=lambda row:row[-2])
                p(f'The best expanded score reached {x[-2]:.4f} F1 on selection at {pct(x[-1])} token accuracy. The selection rule kept the profile with the original candidate set.')
            table(("Cohort","Cases","Exact span F1","Segment-exact F1","Token accuracy","Available gold spans","95% interval for F1 change"),[
                (cohort,q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
                 f'{q["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),
                 pct(q["covered_gold"]/q["candidate"]["exact_labeled_spans"]["gold"]),
                 f'{100*q["gap_ci95"]["exact_span_f1"][0]:+.2f} to {100*q["gap_ci95"]["exact_span_f1"][1]:+.2f} points')
                for cohort,q in dense["splits"].items()])
            if density_qa and density_qa["scores"]==dense["splits"]["test"]["candidate"]:
                choices=density_qa["chosen_span_sources"][1:]
                picked=sum(q["predicted"] for q in choices)
                matches=sum(q["exact_matches"] for q in choices)
                core=density_qa["same_weights_original_lattice"]["exact_labeled_spans"]["f1"]
                lower,upper=density_qa["gap_ci95_same_weights_original_lattice"]["exact_span_f1"]
                p(f'The independent NumPy decoder reproduced all {density_qa["cases"]} test judgments. The wider search selected {picked:,} spans outside the original set, including {matches:,} exact matches. Restricting these same weights to the original set produced {core:.4f} F1. The paired 95% interval for the contribution of the new candidates was {100*lower:+.2f} to {100*upper:+.2f} points.')
                short=choices[1:]
                p(f'Candidates available only through short-interval enumeration accounted for {sum(q["predicted"] for q in short):,} predictions and {sum(q["exact_matches"] for q in short):,} exact matches. The rest of the predictions outside the original set came from the earlier posterior endpoint proposals.')
            links(("Separate span detection and classification in named entity recognition","https://aclanthology.org/2022.semeval-1.221/"))
        repair=result("novel-span-repair")
        repair_freeze=result("novel-span-repair-frozen")
        if repair and repair_freeze and "test" in repair["splits"]:
            h(3,"Targeted repairs for spans outside the original search")
            c=repair_freeze["counts"]
            p(f'The scorer evaluates local changes to the existing segment-exact prediction. Each candidate inserts one new span and can touch up to four neighboring spans. The training split yielded {c["candidates"]:,} candidates across {len(repair_freeze["train_cases"]):,} judgments, including {c["helpful"]:,} that add exact matches and {c["harmful"]:,} that remove them.')
            _,_,mode,scope,threshold=repair["choice"]
            task="direct change in expected exact-match utility" if mode==0 else "separate probabilities of adding and losing exact matches"
            which="short-interval enumeration" if scope=="short" else "short-interval enumeration and posterior endpoints"
            p(f'Validation selected {task} at threshold {threshold:g}, using candidates from {which}. A dynamic program leaves at least one original span between edits so the changes can be scored independently.')
            table(("Cohort","Cases","Exact span F1","Segment-exact F1","Token accuracy","Repairs","Add matches","Lose matches","95% interval for F1 change"),[
                (k,q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
                 f'{q["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),
                 q["edits"]["accepted"],q["edits"]["helpful"],q["edits"]["harmful"],
                 f'{100*q["gap_ci95"]["exact_span_f1"][0]:+.2f} to {100*q["gap_ci95"]["exact_span_f1"][1]:+.2f} points')
                for k,q in repair["splits"].items()])
            test=repair["splits"]["test"]
            a=test["candidate"]["exact_labeled_spans"]
            b=test["parent"]["exact_labeled_spans"]
            p(f'The official test added {a["matched"]-b["matched"]:,} exact matches while increasing the prediction count by {a["predicted"]-b["predicted"]:,}. Its paired F1 interval crossed zero.')
            links(("The Foundations of Cost-Sensitive Learning","https://ijcai.org/proceedings/2001/133"))
        local_oracle=result("novel-span-repair-audit")
        if local_oracle:
            p("A gold-assisted search measures the maximum pooled F1 that these local repairs can attain. It uses the annotations to choose mutually compatible changes.")
            table(("Permitted proposals","Gold-selected repairs","Exact matches","Maximum possible F1"),[
                (label,f'{v["oracle_edits"]:,}',f'{v["matched"]:,}',f'{v["max_f1"]:.4f}')
                for key,label in (("short","Short intervals"),("all","Short intervals and posterior endpoints"))
                for v in [local_oracle["oracle"][key]]])
    transfer=result("whole-joint-transfer-edit")
    transfer_freeze=result("whole-joint-transfer-frozen")
    if transfer and transfer_freeze and "test" in transfer["splits"]:
        h(2,"Transferring joint sentence decoding to raw judgments")
        p("The raw-text splitter defines candidate units. The frozen joint decoder labels those units with and without its sequence transition scores. Contiguous labels become span proposals, and a closed-form readout scores the combined lattice for exact offset and role matches.")
        p(f'The training lattice contains {transfer_freeze["train_arcs"]:,} proposals. The transferred decoder adds {transfer_freeze["train_novel_arcs"]:,} that were absent from the incumbent lattice, including {transfer_freeze["train_exact_novel"]:,} exact gold spans.')
        _,_,strength,cost,fee,wide=transfer["choice"]
        scope="the expanded lattice" if wide else "the original lattice with the new evidence"
        p(f'Validation selected {scope}, a residual weight of {strength:g}, a cost of {cost:g} per span and a cost of {fee:g} per added proposal.')
        table(("Cohort","Cases","Corrected span F1","Segment-exact F1","Direct joint transfer F1","Token accuracy","New proposals","New exact proposals"),[
            (s,q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',f'{q["preceding"]["exact_labeled_spans"]["f1"]:.4f}',
             f'{q["standalone_sentence_joint"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),
             f'{q["candidate_coverage"]["joint_added_arcs"]:,}',f'{q["candidate_coverage"]["joint_added_exact_arcs"]:,}')
            for s,q in transfer["splits"].items()])
        lo,hi=transfer["splits"]["test"]["gap_ci95"]["exact_span_f1"]
        p(f'The 95% case-paired interval for exact span F1 change versus segment-exact is {100*lo:+.2f} to {100*hi:+.2f} points.')
        qa=result("whole-joint-transfer-qa")
        if qa and qa["candidate"]==transfer["splits"]["test"]["candidate"]:
            c=qa["original_arcs_only"]["exact_labeled_spans"]["f1"]
            p(f'The independent decoder reproduced the official result. It used {qa["joint_arcs_selected"]:,} new proposals, including {qa["exact_joint_arcs_selected"]:,} exact matches. Keeping the same scores and restricting them to the incumbent proposals gives {c:.4f} exact span F1.')
        if raw_verified:
            h(3,"Direct role assignment from raw text")
            table(("Full-judgment profile","Exact span F1","Token accuracy"),[
                (name,f'{raw_joint[key]["exact_labeled_spans"]["f1"]:.4f}',
                 f'{100*raw_joint[key]["token_accuracy"]:.2f}%')
                for key,name in (("scores","Direct joint transfer"),("older_token_balanced","Previous token-balanced"),
                                 ("role_edge_balanced","Balanced role-edge"))])
            lo,hi=raw_joint["paired_to_token_balanced"]["token_accuracy"]
            elo,ehi=raw_joint["paired_to_token_balanced"]["exact_span_f1"]
            p(f'The paired 95% interval versus the previous token-balanced profile is {100*lo:+.2f} to {100*hi:+.2f} percentage points for token accuracy and {100*elo:+.2f} to {100*ehi:+.2f} points for exact span F1.')
            if raw_eligible:
                p(f'The raw-joint option uses the same checkpoint as sentence-joint and returns Unicode spans. Live inference reproduced the raw predictions on {len(raw_joint["verified_live_cases"])} official judgments. Segment-exact has the highest measured raw exact-span F1.')
        links(("Filtered Semi-Markov CRF","https://aclanthology.org/2023.findings-emnlp.17/"))
    role_retention=result("sentence-role-retention")
    score_calibration=result("sentence-emission-calibration")
    composition=result("sentence-edit-composition")
    gate=result("sentence-calibration-gate")
    label_change=result("sentence-label-change")
    archive_rows=result("known-sentence-numerical-archives")
    if role_retention or score_calibration or composition or gate or label_change:
        h(2,"Improving reasoning and decision in supplied sentences")
        p("The released sentence-context profile improved its overall score while losing F1 on reasoning and decision. The transition experiment searches for edited weights that retain F1 on both roles. The emission experiment uses a closed-form correction fitted to the errors of the existing decoder. The composition study varies how much of each frozen edit to use. Evaluation retains the seven LegalSeg roles and one label per supplied sentence.")
        rows=[]
        for label,trial in (("Transition edit",role_retention),("Emission calibration",score_calibration),
                            ("Composition",composition),("Selective calibration",gate),("Pair-specific changes",label_change)):
            if not trial:
                continue
            for cohort,q in trial["splits"].items():
                if "candidate" not in q:
                    q=q["retention"]
                a=q["candidate"]
                b=q["preceding"]
                lo,hi=q["paired_to_preceding"]["macro_f1"]
                rows.append([label,cohort,q["cases"],f'{a["macro_f1"]:.4f}',f'{b["macro_f1"]:.4f}',pct(a["accuracy"]),
                    f'{a["per_role_f1"]["reasoning"]:.3f}',f'{a["per_role_f1"]["decision"]:.3f}',
                    f'{100*lo:+.2f} to {100*hi:+.2f} points'])
        table(("Method","Cohort","Cases","Macro-F1","Preceding F1","Accuracy","Reasoning F1","Decision F1","95% interval for macro-F1 gain"),rows)
        if composition:
            structural,emission=composition["choice"]
            p(f'Development selected transition-edit strength {structural:g} and emission calibration strength {emission:g} in the composition study.')
        if gate:
            mode,threshold=gate["choice"]
            picked="full calibration for every sentence" if mode=="full" else f'{mode} at threshold {threshold:g}'
            p(f'The selective calibration study tested score-margin thresholds and a reasoning-only correction. Development selected {picked}. Selection allowed at most a 0.5-point drop in sentence accuracy and reasoning F1.')
        audit=result("sentence-calibration-change-audit")
        if audit:
            p(f'Comparing the individual outputs on {audit["rows"]:,} sentences, calibration changed {audit["changed"]:,} labels. {audit["fixed"]:,} changes corrected an error, and {audit["broken"]:,} changed a correct label to an incorrect one.')
            table(("Preceding role","Calibrated role","Changed labels","Corrected errors","New errors","Net correct"),[
                (a["previous"].replace("_"," "),a["new"].replace("_"," "),f'{a["changed"]:,}',f'{a["became_correct"]:,}',
                 f'{a["became_incorrect"]:,}',f'{a["net_correct"]:+,}') for a in audit["edits"][:6]])
        if label_change:
            name,threshold=label_change["choice"]
            p(f'The pair-specific study estimates each proposed label change’s net benefit on training cases. It compares overall accuracy and rarity-weighted benefits, with or without score-margin buckets. Validation selected {name} at threshold {threshold:g}.')
            p("Both candidate sequences use the full judgment context. The final profile retains selected changes one sentence at a time after decoding.")
        verified_change=result("sentence-label-change-qa")
        if verified_change:
            a,b=verified_change["scores"],verified_change["preceding"]
            minimum,maximum=verified_change["paired_to_preceding"]["macro_f1"]
            p(f'The sentence-calibrated checkpoint exactly reproduced the experiment on {verified_change["cases"]:,} judgments. It raises macro-F1 from {b["macro_f1"]:.3f} to {a["macro_f1"]:.3f} and reasoning F1 from {b["per_role_f1"]["reasoning"]:.3f} to {a["per_role_f1"]["reasoning"]:.3f}. The paired 95% interval for macro-F1 gain is {100*minimum:+.2f} to {100*maximum:+.2f} points. The CLI uses the same ordered JSON sentence input as sentence-context.')
            ref=verified_change["archive_scores"]
            min_a,max_a=verified_change["paired_to_archive"]["macro_f1"]
            p(f'The leading aligned archive scores {ref["macro_f1"]:.3f}. The paired 95% interval for the calibrated checkpoint’s macro-F1 difference against this archive is {100*min_a:+.2f} to {100*max_a:+.2f} points.')
            low_acc,high_acc=verified_change["paired_to_archive"]["accuracy"]
            p(f'Against that same archive, the calibrated checkpoint scores {100*a["accuracy"]:.2f}% accuracy versus {100*ref["accuracy"]:.2f}%, a paired 95% interval of {100*low_acc:+.2f} to {100*high_acc:+.2f} points. It also has a higher accuracy point estimate than the other seven aligned archives.')
        links(("Constrained learning with confusion-matrix metrics","https://proceedings.mlr.press/v84/narasimhan18a.html"))
    hidden=result("sentence-hidden-edit")
    hidden_frozen=result("sentence-hidden-edit-frozen")
    lexical=result("sentence-lexical-fusion-edit")
    repair=result("sentence-hidden-repair")
    repair_frozen=result("sentence-hidden-repair-frozen")
    if repair and repair["choice"] and "test" not in repair["splits"]:
        repair=None
    if hidden and hidden_frozen and lexical:
        h(2,"Testing richer sentence readouts")
        values=hidden_frozen["values"]
        hbest=max(v[0] for k,v in values.items() if k.startswith("linear-"))
        rbest=max(v[0] for k,v in values.items() if k.startswith("random_features-"))
        p(f'The encoder produced {hidden_frozen["train_width"]}-dimensional states for {hidden_frozen["train_rows"]:,} sentences across {len(hidden_frozen["train_cases"]):,} official training judgments. The experiment wrote either a fresh readout or an adjustment to existing scores. Both used a closed-form ridge solve, with the encoder frozen. The nonlinear readout added fixed random ReLU features.')
        p(f'On the 72-case selection set, the strongest linear write scored {hbest:.4f} macro-F1, the strongest nonlinear write scored {rbest:.4f}, and the released profile scored {values["released"][0]:.4f}. Both the macro-F1 and role-retention selection rules chose the release. This experiment left the official test outside its readout comparison.')
        rows=[]
        for label,study in (("Frozen lexical model blend",lexical),("Conditional hidden readout",repair)):
            if not study or not study.get("choice"):
                continue
            for cohort,q in study["splits"].items():
                a,b=q["candidate"],q["preceding"]
                lo,hi=q["paired_to_preceding"]["macro_f1"]
                rows.append([label,cohort,q["cases"],f'{a["macro_f1"]:.4f}',f'{b["macro_f1"]:.4f}',pct(a["accuracy"]),
                             f'{a["per_role_f1"]["reasoning"]:.3f}',f'{a["per_role_f1"]["decision"]:.3f}',
                             f'{100*lo:+.2f} to {100*hi:+.2f} points'])
        table(("Method","Cohort","Cases","Macro-F1","Released F1","Accuracy","Reasoning F1","Decision F1","95% interval for macro-F1 change"),rows)
        p("The lexical experiment tried direct score mixtures and a closed-form fusion of frozen fastText and neural scores. Selection assigned 25 percent of the emission score to the fusion head. Decision F1 improved on the official test while reasoning F1 and overall macro-F1 fell.")
        if repair and repair_frozen:
            p(f'The conditional readout fits separate correction weights for each currently predicted role. It estimates the reliability of specific role changes and score margins on {len(repair_frozen["reliability_cases"])} separate training judgments. The selection rule permits up to a 0.2-point drop in accuracy.')
            if repair["choice"] is None:
                p("The selection rule retained the release. Evaluation stopped at the selection cohort for this edit.")
            else:
                model_name,margin,utility=repair["choice"]
                kind="linear" if model_name.startswith("linear-") else "nonlinear"
                gate_description=" and bypassed the role-change reliability filter" if utility<0 else f' and required estimated net benefit above {utility:g}'
                p(f'The selected {kind} correction required a score margin of {margin:g}{gate_description}. The saved tensors exactly reproduced the selection scores before they were applied to the later cohorts.')
                test=repair["splits"]["test"]
                macro=test["candidate"]["macro_f1"]-test["preceding"]["macro_f1"]
                p(f'The corrected readout changed {test["modified"]:,} of {test["candidate"]["rows"]:,} test labels. Macro-F1 changed by {100*macro:+.2f} points. The selection gain disappeared on both validation cohorts.')
        p("Each comparison freezes its candidate before replaying the official test. Successive research stages have used the same public test split, so the reported intervals quantify case-level sampling uncertainty for each comparison. A fresh labeled set would be needed to verify a small gain independently after this sequence of experiments.")
        links(("RanPAC paper on fixed random projections and closed-form readouts","https://proceedings.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html"))
    prototypes=result("sentence-prototype-edit")
    prototype_freeze=result("sentence-prototype-frozen")
    if prototypes and prototype_freeze and "test" in prototypes["splits"]:
        h(3,"Multiple prototypes per rhetorical role")
        p(f'This experiment writes cosine prototypes from {len(prototype_freeze["training_cases"]):,} LegalSeg training judgments. The frozen document encoder supplies recurrent states for sentences with their judgment context. The study compares nearest-prototype votes, direct class scores, and closed-form corrections to the current scores.')
        shrink,number,mode,_,gain=prototypes["choice"]
        setting={"vote":"prototype voting","residual":"a residual readout","class":"a blended class readout"}[mode]
        norm="regularized covariance whitening" if shrink else "per-feature scale normalization"
        p(f'Validation selected {number} prototypes per role with {norm}, {setting}, and a scale of {gain:g}. All settings were frozen before the confirmation cohorts and official test.')
        table(("Cohort","Cases","Macro-F1","Sentence-calibrated F1","Accuracy","Labels changed","95% interval for F1 change"),[
            (s,r["cases"],f'{r["candidate"]["macro_f1"]:.4f}',f'{r["preceding"]["macro_f1"]:.4f}',
             pct(r["candidate"]["accuracy"]),f'{r["modified"]:,}',
             f'{100*r["paired_to_preceding"]["macro_f1"][0]:+.2f} to {100*r["paired_to_preceding"]["macro_f1"][1]:+.2f} points')
            for s,r in prototypes["splits"].items()])
        test=prototypes["splits"]["test"]
        a,b=test["candidate"],test["preceding"]
        lo,hi=test["paired_to_preceding"]["reasoning"]
        p(f'The prototype variant changed {test["modified"]:,} supplied-sentence labels. Reasoning F1 rose from {b["per_role_f1"]["reasoning"]:.4f} to {a["per_role_f1"]["reasoning"]:.4f}, with a paired 95% interval of {100*lo:+.2f} to {100*hi:+.2f} points. The aggregate macro-F1 change was small.')
        links(("Infinite Mixture Prototypes for Few-shot Learning","https://proceedings.mlr.press/v97/allen19b.html"))
        prototype_gate=result("sentence-prototype-gate-edit")
        prototype_gate_freeze=result("sentence-prototype-gate-frozen")
        if prototype_gate and prototype_gate_freeze and "test" in prototype_gate["splits"]:
            h(4,"Filtering proposed role changes")
            cal=prototype_gate_freeze["calibration_changes"]
            p(f'The gate uses {len(prototype_gate_freeze["calibration_cases"])} training judgments excluded from prototype fitting. On those cases, the prototype edit proposed {cal["altered"]:,} label changes, including {cal["helpful"]:,} that corrected errors and {cal["harmful"]:,} that introduced them.')
            selected,_,threshold=prototype_gate["choice"]
            cue="cosine-score margins" if selected=="margin" else "the labels of neighboring sentences"
            p(f'Validation selected {cue} and a utility threshold of {threshold:g}. A per-role-pair estimate shrinks rare patterns toward their average utility.')
            table(("Cohort","Cases","Gated F1","Ungated F1","Sentence-calibrated F1","Labels changed","95% interval versus sentence-calibrated"),[
                (s,r["cases"],f'{r["candidate"]["macro_f1"]:.4f}',f'{r["prototype_ungated"]["macro_f1"]:.4f}',
                 f'{r["preceding"]["macro_f1"]:.4f}',f'{r["modified"]:,}',
                 f'{100*r["paired_to_preceding"]["macro_f1"][0]:+.2f} to {100*r["paired_to_preceding"]["macro_f1"][1]:+.2f} points')
                for s,r in prototype_gate["splits"].items()])
    mini=result("sentence-minilm-fusion-edit")
    frozen_mini=result("sentence-minilm-fusion-frozen")
    if mini and frozen_mini and "test" in mini["splits"]:
        h(3,"A second encoder for low-confidence sentences")
        p(f'The 22.7M-parameter MiniLM encoder runs from a pinned quantized graph. A closed-form correction uses {len(frozen_mini["training_cases"]):,} LegalSeg training judgments. The hierarchy ranks risk using its original and calibrated role probabilities. Only the highest-risk fraction of each judgment is eligible for a new label.')
        p(f'MiniLM reads each routed sentence in isolation within a {frozen_mini["sentence_token_limit"]}-token cap. For longer sentences it retains tokens from both ends. The comparison includes corrections from the hierarchy alone and from MiniLM, with or without hierarchy features.')
        p("The quantized graph showed small changes in output when a sentence was encoded alone and with other sentences. These results use the benchmark's fixed batching and cached vectors.")
        table(("Frozen profile","Input for correction","Sentences eligible","Readout weight","Minimum margin"),[
            (k,v[0].replace("_"," "),pct(v[2]),pct(v[3]),f'{v[4]:g}') for k,v in mini["choices"].items()])
        table(("Cohort","Cases","Profile","Macro-F1","Sentence-calibrated F1","Accuracy","Labels changed","95% interval for F1 change"),[
            (name,d["cases"],profile.replace("_"," "),f'{q["scores"]["macro_f1"]:.4f}',f'{d["preceding"]["macro_f1"]:.4f}',
             pct(q["scores"]["accuracy"]),f'{q["changed"]:,}',
             f'{100*q["gap_ci95"]["macro_f1"][0]:+.2f} to {100*q["gap_ci95"]["macro_f1"][1]:+.2f} points')
             for name,d in mini["splits"].items() for profile,q in d["profiles"].items()])
        p(f'The official MiniLM replay encoded {mini["splits"]["test"]["sentences_encoded"]:,} supplied sentences. Its aggregate macro-F1 interval crossed zero relative to sentence-calibrated. The hierarchy-only correction had the highest macro-F1 and accuracy point estimates in this comparison.')
        links(("MiniLM checkpoint and ONNX assets","https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx"),
              ("Stacked generalization","https://doi.org/10.1016/s0893-6080%2805%2980023-1"))
    batch_audit=result("sentence-minilm-batch-audit")
    if batch_audit:
        h(4,"Quantized batching check")
        table(("Validation case","Sentences","Encoded sentences","Mean embedding cosine","Changed predictions"),[
            (q["case"],q["rows"],q["queried"],f'{q["mean_cosine"]:.5f}',q["label_disagreements"]) for q in batch_audit["cases"]])
        p("This check compares encoding the same sentences in a judgment-only batch and in the benchmark batches that mix judgments.")
    token_table=result("sentence-minilm-token-table-edit")
    token_freeze=result("sentence-minilm-token-table-frozen")
    if token_table and token_freeze and "test" in token_table["splits"]:
        h(3,"Using pretrained token embeddings without the transformer")
        size=len(token_freeze["training_cases"])
        p(f'This comparison uses the frozen MiniLM input embedding matrix. It pools token embeddings by inverse document frequency estimated on {size} training judgments, then writes a seven-role correction. The hierarchy routes the same low-confidence sentences.')
        table(("Cohort","Cases","Profile","Macro-F1","Sentence-calibrated F1","Accuracy","Labels changed","95% interval for F1 change"),[
            (name,d["cases"],("Token embeddings" if profile=="semantic_fusion" else "Hierarchy only"),
             f'{q["scores"]["macro_f1"]:.4f}',f'{d["preceding"]["macro_f1"]:.4f}',pct(q["scores"]["accuracy"]),
             f'{q["changed"]:,}',f'{100*q["gap_ci95"]["macro_f1"][0]:+.2f} to {100*q["gap_ci95"]["macro_f1"][1]:+.2f} points')
            for name,d in token_table["splits"].items() for profile,q in d["profiles"].items()])
        links(("MiniLM model card","https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"))
    reviewed_routing=result("sentence-uncertainty-qa")
    if reviewed_routing:
        h(4,"Portable hierarchy-only correction")
        a=reviewed_routing["scores"]
        b=reviewed_routing["calibrated"]
        c=reviewed_routing["context"]
        p(f'The portable correction reproduced the official scores on all {reviewed_routing["cases"]:,} judgments and matched live inference on {len(reviewed_routing["verified_live_cases"])} cases. The readout uses the original hierarchy, its two posterior estimates, and the nearby predicted roles.')
        table(("Profile","Macro-F1","Accuracy","Reasoning F1"),[
            (name,f'{q["macro_f1"]:.4f}',f'{100*q["accuracy"]:.2f}%',f'{q["per_role_f1"]["reasoning"]:.4f}')
            for name,q in (("Hierarchy-only correction",a),("Sentence-calibrated",b),("Sentence-context",c))])
        for label,key,metric_ in (("sentence-calibrated","paired_to_calibrated","accuracy"),
                                    ("sentence-context","paired_to_context","macro_f1")):
            lo,hi=reviewed_routing[key][metric_]
            description="accuracy" if metric_=="accuracy" else "macro-F1"
            p(f'The paired 95% interval for the {description} change versus {label} is {100*lo:+.2f} to {100*hi:+.2f} points.')
        if (reviewed_routing["paired_to_context"]["macro_f1"][0]>0
            and reviewed_routing["paired_to_calibrated"]["accuracy"][0]>0 and a["accuracy"]>=c["accuracy"]):
            p("The downloaded sentence-uncertainty profile exposes this operating point with the same supplied-sentence input and adds about 2 KB to the calibrated checkpoint.")
        latency=result("sentence-uncertainty-cpu-latency")
        if latency:
            table(("Supplied sentences","Input words","Calibrated median","Uncertainty median"),[
                (q["candidate_units"],f'{q["whitespace_tokens"]:,}',f'{q["medians"]["sentence-calibrated"]:.2f} s',
                 f'{q["medians"]["sentence-uncertainty"]:.2f} s') for q in latency["cases"]])
            p("Each median uses three warmed measurements with two CPU threads. The MiniLM benchmark was running concurrently.")
    routing_audit=result("sentence-minilm-routing-audit")
    if routing_audit:
        h(3,"How many existing errors can a second encoder see?")
        p(f'The routing audit covers {routing_audit["cases"]:,} test judgments and uses gold labels only after selecting candidates. The oracle assigns the correct role to every routed sentence.')
        table(("Routing signal","Fraction per case","Sentences selected","Existing errors routed","Share of all errors","Oracle macro-F1"),[
            (kind,pct(float(rate)),f'{q["queried"]:,}',f'{q["routed_errors"]:,}',pct(q["error_recall"]),f'{q["oracle"]["macro_f1"]:.4f}')
            for kind,rs in routing_audit["methods"].items() for rate,q in rs.items()])
    gemma_probe=result("embeddinggemma-q4-cpu-probe")
    if gemma_probe:
        h(2,"CPU feasibility for a second sentence encoder")
        timings=gemma_probe["median_seconds_by_batch_size"]
        p(f'The pinned EmbeddingGemma Q4 ONNX export ran locally with two CPU threads. It returned {gemma_probe["output_dimensions"]} dimensions using the published classification prompt. On {gemma_probe["probe_count"]} probes, the minimum cosine similarity between batched and individual embeddings was {gemma_probe["batch_vs_individual_min_cosine"]:.6f}. Warm inference took {timings["1"]*1000:.1f} ms for one probe and {timings[str(gemma_probe["probe_count"])]*1000:.1f} ms for the mixed-length batch.')
        p(f'The feasibility run used a {gemma_probe["probe_max_tokens"]}-token cap. The model card lists a 2,048-token maximum. LegalSeg classification accuracy and macro-F1 still require a separate evaluation.')
        links(("EmbeddingGemma model card","https://huggingface.co/google/embeddinggemma-300m"),
              ("Pinned Q4 export","https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/tree/5090578d9565bb06545b4552f76e6bc2c93e4a66"))
    gemma2_probe=result("embeddinggemma2-q8-cpu-probe")
    if gemma2_probe:
        h(3,"EmbeddingGemma 2 CPU probe")
        times=gemma2_probe["median_seconds"]
        p(f'The text-only 270M-parameter model was verified with the pinned Q8 GGUF and llama.cpp on three CPU threads. The model returns {gemma2_probe["output_dims"]} normalized dimensions. On these synthetic examples, median warm latency was {times["1"]*1000:.1f} ms for a single input and {times["5"]*1000:.1f} ms for five together. LegalSeg role accuracy remains unmeasured for this model.')
        links(("EmbeddingGemma 2 model card","https://gemma.ai/models/embeddinggemma/model-card-v2/"),
              ("Pinned GGUF revision","https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/tree/6ea3e3a309d12a4779cfda98b8c19800d2d5da71"))
    gemma=result("embeddinggemma-routed-write")
    gemma_freeze=result("embeddinggemma-routed-frozen")
    if gemma and gemma_freeze and "confirm" in gemma["splits"]:
        h(2,"EmbeddingGemma weight edits on ambiguous sentences")
        training_cases=len({case for case,_ in gemma_freeze["train_rows"]})
        p(f'Three closed-form seven-role classifiers use frozen hierarchy features, EmbeddingGemma vectors, or both. All use the same {len(gemma_freeze["train_rows"]):,} labeled examples from {training_cases:,} of {len(gemma_freeze["train_cases"]):,} source training judgments. The second encoder uses the classification prompt and a {gemma_freeze["token_cap"]}-token cap. Each head can revise up to {gemma_freeze["per_case_route"]} ambiguous inputs per judgment using the same incumbent scores.')
        gemma_split=dict(gemma["splits"])
        independent=result("embeddinggemma-routed-independent")
        if independent and independent["freeze_sha256"]==gemma["freeze_sha256"]:
            gemma_split["independent"]=independent["results"]
        table(("Cohort","Features","Judgments","Routed sentences","Routed macro-F1","Parent routed macro-F1","All-sentence macro-F1","Parent all-sentence macro-F1","Changed labels"),[
            (name,arm,q["cases"],q["routed"],f'{q["routed_scores"]["macro_f1"]:.4f}',f'{q["routed_parent"]["macro_f1"]:.4f}',
             f'{q["full"]["macro_f1"]:.5f}',f'{q["full_parent"]["macro_f1"]:.5f}',q["changed"])
            for name,arms in gemma_split.items() for arm,q in arms.items()])
        labels={"hierarchy":"hierarchy only","gemma":"EmbeddingGemma only","combined":"combined"}
        for key,q in gemma["splits"]["confirm"].items():
            lo,hi=q["routed_ci"]["macro_f1"]
            p(f'The confirmation interval versus sentence-joint on the routed subset is {100*lo:+.2f} to {100*hi:+.2f} macro-F1 points for {labels[key]}.')
        qa=result("embeddinggemma-routed-qa")
        if qa and qa["freeze_sha256"]==gemma["freeze_sha256"]:
            lo,hi=qa["splits"]["confirm"]["paired_to_hierarchy_routed"]["combined"]["macro_f1"]
            p(f'A separate replay reproduced the predictions. Against the matched hierarchy-only edit the combined edit has a confirmation interval of {100*lo:+.2f} to {100*hi:+.2f} routed macro-F1 points.')
        p("This screen uses development and held-out validation judgments. The official test comparison and published standing still use sentence-joint.")
        if "independent" in gemma_split:
            q=gemma_split["independent"]["combined"]
            lo,hi=q["full_ci"]["macro_f1"]
            p(f'The combined edit did not reproduce the routed gain on the text-disjoint cohort. Its all-sentence macro-F1 change had a 95% interval of {100*lo:+.2f} to {100*hi:+.2f} points versus sentence-joint. The release retains sentence-joint.')
        links(("EmbeddingGemma research paper","https://arxiv.org/abs/2509.20354"))
    local_context=result("gemma2-neighbor")
    local_context_freeze=result("gemma2-neighbor-frozen")
    if local_context and local_context_freeze and "independent" in local_context["splits"]:
        h(2,"Immediate sentence context in EmbeddingGemma 2")
        p(f'Three groups of output weights were written from {local_context_freeze["training_rows"]:,} labeled training items. They compare hierarchy features alone, hierarchy with the target sentence embedded by itself, and hierarchy with target and immediate context embedded together. The same target text is used in both EmbeddingGemma 2 formats. Linear and quadratic kernels estimate the correction to the incumbent role scores without backpropagation.')
        table(("Cohort","Inputs to edit","Cases","Routed sentences","Routed macro-F1","Parent routed macro-F1","All-sentence macro-F1","Parent all-sentence macro-F1","Changed labels"),[
            (group,arm,q["cases"],q["routed"],f'{q["routed_scores"]["macro_f1"]:.4f}',f'{q["routed_parent"]["macro_f1"]:.4f}',
             f'{q["full"]["macro_f1"]:.5f}',f'{q["full_parent"]["macro_f1"]:.5f}',q["changed"])
            for group,heads in local_context["splits"].items() for arm,q in heads.items()])
        q=local_context["splits"]["independent"]["neighbors"]
        lo,hi=q["full_ci"]["macro_f1"]
        p(f'The case-paired 95% interval for the contextual edit was {100*lo:+.2f} to {100*hi:+.2f} macro-F1 points on all sentences of the independent validation judgments. At most {local_context_freeze["per_case_route"]} sentences in each judgment could receive an edit.')
        p("The contextual arm changed four labels on the text-disjoint judgments. Both the target-only and hierarchy-only arms selected zero correction during validation. The release retains sentence-joint.")
        links(("EmbeddingGemma 2 paper","https://arxiv.org/abs/2610.37364"))
    complement=result("sentence-author-complementarity")
    if complement:
        h(2,"Which errors can another sentence model repair?")
        p(f'This diagnostic compares frozen profile predictions on the same {complement["rows"]:,} sentences. When either source assigns the human label, a gold-assisted chooser uses that label. Otherwise it retains sentence-joint’s prediction. Accuracy is the ceiling for choosing among these outputs. Macro-F1 depends on the fallback used for incorrect predictions.')
        table(("Archived model","Only archive correct","Only sentence-joint correct","Neither correct","Gold-assisted chooser accuracy","Gold-assisted chooser macro-F1"),[
            (name.replace("_"," "),f'{q["only_archive_correct"]:,}',f'{q["only_local_correct"]:,}',f'{q["neither_correct"]:,}',pct(q["correctness_oracle"]["accuracy"]),
             f'{q["correctness_oracle"]["macro_f1"]:.4f}') for name,q in complement["comparisons"].items()])
        leading=result("known-sentence-numerical-archives")["best_by_macro_f1"]
        pair=complement["comparisons"][leading]
        table(("Human role","Only leading author archive correct","Only sentence-joint correct"),[
            (name.replace("_"," "),f"{a:,}",f"{b:,}")
            for name,a,b in zip(complement["role_order"],pair["only_archive_correct_by_role"],pair["only_local_correct_by_role"])])
        q=complement["all_source_correctness_oracle"]
        p(f'Combining the available labels from sentence-joint and all eight archives gives a ceiling of {pct(q["accuracy"])} accuracy. Using the local fallback yields {q["macro_f1"]:.4f} macro-F1. The chooser uses the human labels for this diagnostic.')
    repairs=result("sentence-joint-repair-routing")
    repaired_frozen=result("sentence-joint-repair-routing-frozen")
    if repairs and repaired_frozen and "test" in repairs["splits"]:
        h(2,"Ranking sentences for a second classifier")
        p(f'The hierarchy is frozen. A closed-form readout estimates whether sentence-joint has made a mistake using labeled examples from {len(repaired_frozen["training_cases"]):,} training judgments. All ranking methods see the same candidate rows and use the same budget within each judgment.')
        held=repairs["splits"]["test"]
        table(("Budget","Ranking","Sentences referred","Incumbent errors found","Author-correctable errors found","Gold-assisted macro-F1"),[
            (budget,name,f'{q["queries"]:,}',f'{q["errors_found"]:,}',f'{q["archive_recoverable_errors_found"]:,}',
             f'{q["archive_assisted"]["macro_f1"]:.4f}') for budget,ranks in held["budgets"].items()
             for name,q in ranks.items()])
        p("An author-correctable error is one that the strongest aligned archive labeled correctly. Gold labels are used only to count errors and score the diagnostic. The table measures the potential benefit of changing the referral order.")
        focus=held["budgets"]["3 per case"]
        large=held["budgets"]["30%"]
        p(f'At three referrals per judgment, the simplest confidence ranking found {focus["low assigned probability"]["archive_recoverable_errors_found"]:,} author-correctable errors, compared with {focus["smallest utility gap"]["archive_recoverable_errors_found"]:,} for the current ranking and {focus["written risk"]["archive_recoverable_errors_found"]:,} for the written risk score. At 30% referral, the written risk score found {large["written risk"]["archive_recoverable_errors_found"]:,} versus {large["smallest utility gap"]["archive_recoverable_errors_found"]:,}.')
        lo,hi=large["written risk"]["archive_assisted_gap_to_current_route_ci"]["macro_f1"]
        p(f'At the 30% budget, the 95% case-paired difference in gold-assisted macro-F1 is {100*lo:+.2f} to {100*hi:+.2f} points. These scores assume the second source is selected correctly for each available repair.')

        links(("Selective Classification for Deep Neural Networks","https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html"))
    ngram=result("sentence-ngram-residual")
    ngram_freeze=result("sentence-ngram-residual-frozen")
    if ngram and ngram_freeze and "independent" in ngram["splits"]:
        h(2,"A compact lexical correction for referred sentences")
        choices=ngram_freeze["choice"]
        p(f'A sparse lexical memory uses unigram and bigram counts from {ngram_freeze["memory_training_rows"]:,} labeled sentences in {len(ngram_freeze["memory_training_cases"]):,} training judgments. Three sets of weights encode the target, previous sentence and following sentence. Seven-role correction weights are solved on {len(ngram_freeze["readout_training_cases"]):,} other training judgments. The base classifier remains frozen.')
        if choices[0]=="unchanged":
            p("Selection retained the original classifier.")
        else:
            arm,rarity,ridge,route,gain,threshold=choices
            rate="three sentences per judgment" if route==0. else f"{100*route:g}% of sentences per judgment"
            p(f'Selection used {arm} features and referred up to {rate} by low assigned-role probability. The frozen correction gain was {gain:g} and the acceptance threshold was {threshold:g}.')
        table(("Cohort","Judgments","Edited macro-F1","Sentence-joint macro-F1","Edited accuracy","Sentence-joint accuracy","Changed labels"),[
            (name,q["cases"],f'{q["candidate"]["macro_f1"]:.5f}',f'{q["baseline"]["macro_f1"]:.5f}',
             pct(q["candidate"]["accuracy"]),pct(q["baseline"]["accuracy"]),f'{q["changed"]:,}')
            for name,q in ngram["splits"].items()])
        if "test" in ngram["splits"]:
            q=ngram["splits"]["test"]
            a,b=q["paired"]["macro_f1"]
            p(f'The case-paired 95% difference on official test macro-F1 was {100*a:+.2f} to {100*b:+.2f} points.')
            acc=q["candidate"]["correct"]-q["baseline"]["correct"]
            p(f'The edit classified {acc:,} additional official test sentences correctly. Its accuracy gain is {100*q["paired"]["accuracy"][0]:+.2f} to {100*q["paired"]["accuracy"][1]:+.2f} percentage points at 95% confidence. The macro-F1 interval includes zero and reasoning F1 declined, so sentence-joint remains the supported profile.')
            qa=result("sentence-ngram-residual-qa")
            if qa and qa["candidate"]==q["candidate"]:
                lo,hi=qa["paired_to_aligned_archive"]["macro_f1"]
                p(f'Independent replay reproduced the predictions. Against the strongest aligned author archive, the paired macro-F1 interval is {100*lo:+.2f} to {100*hi:+.2f} points.')
        links(("Baselines and Bigrams","https://aclanthology.org/P12-2018/"))
    pair=result("sentence-ngram-pair-gate")
    pair_freeze=result("sentence-ngram-pair-gate-frozen")
    if pair and pair_freeze and "independent" in pair["splits"]:
        h(2,"Guarding lexical edits by the two affected roles")
        p(f'The lexical weights stay frozen. The new table estimates the accuracy and macro-F1 effects of changes between each pair of roles at different score gaps. It is written from {len(pair_freeze["train_cases"]):,} training judgments outside the lexical memory and correction fit.')
        mode=pair_freeze["choice"][0]
        if mode=="guard-reasoning":
            p("Selection retained the simple rule that preserves existing reasoning labels. It can still assign reasoning to a sentence whose original label was different.")
        p(f'Development selected the {mode.replace("-"," ")} option with a reasoning-F1 tolerance of 0.001 and an accuracy tolerance of 0.0003 relative to sentence-joint.')
        table(("Cohort","Judgments","Gated macro-F1","Lexical macro-F1","Sentence-joint macro-F1","Gated accuracy","Gated reasoning F1","Changed labels"),[
            (name,q["cases"],f'{q["candidate"]["macro_f1"]:.5f}',f'{q["lexical"]["macro_f1"]:.5f}',f'{q["baseline"]["macro_f1"]:.5f}',pct(q["candidate"]["accuracy"]),
             f'{q["candidate"]["per_role_f1"]["reasoning"]:.5f}',f'{q["changed"]:,}')
            for name,q in pair["splits"].items()])
        if "test" in pair["splits"]:
            q=pair["splits"]["test"]
            lo,hi=q["paired_to_baseline"]["macro_f1"]
            p(f'The paired 95% difference against sentence-joint was {100*lo:+.2f} to {100*hi:+.2f} macro-F1 points.')
            count=q["candidate"]["correct"]-q["baseline"]["correct"]
            reason=q["candidate"]["per_role_f1"]["reasoning"]-q["baseline"]["per_role_f1"]["reasoning"]
            r0,r1=q["paired_to_baseline"]["reasoning"]
            a0,a1=q["paired_to_baseline"]["accuracy"]
            p(f'The selected guard scored {count:+,} correct sentence labels and a reasoning-F1 change of {reason:+.5f} against sentence-joint. The case-paired 95% intervals are {100*r0:+.2f} to {100*r1:+.2f} reasoning-F1 points and {100*a0:+.2f} to {100*a1:+.2f} accuracy percentage points. The overall macro-F1 interval includes zero, and the release retains sentence-joint.')
            qa=result("sentence-ngram-pair-gate-qa")
            if qa and qa["candidate"]==q["candidate"]:
                a,b=qa["paired_to_aligned_archive"]["macro_f1"]
                p(f'Independent replay matched the result. The strongest aligned author archive scores {qa["aligned_archive"]["macro_f1"]:.5f} macro-F1, with a case-paired difference of {100*a:+.2f} to {100*b:+.2f} points against it.')
        links(("On the Bayes-Optimality of F-Measure Maximizers","https://www.jmlr.org/papers/v15/waegeman14a.html"))
    graph_affinity=result("sentence-document-graph-affinity")
    graph=result("sentence-document-graph")
    graph_freeze=result("sentence-document-graph-frozen")
    if graph_affinity or (graph and graph_freeze and "independent" in graph["splits"]):
        h(2,"Long-range sentence relationships within a judgment")
        if graph_affinity:
            q=graph_affinity["overall"]
            p(f'The weighted cosine graph links sentences at least five positions apart within one judgment. On {graph_affinity["sentences_with_remote_edges"]:,} official test sentences with remote edges, {pct(q["semantic_neighbor_agreement"])} of neighbor weight joins sentences with the same human role. Uniformly choosing a remote sentence from the same case gives {pct(q["uniform_remote_agreement"])} expected agreement. Human labels enter this diagnostic only after edges are fixed.')
            if "distance_matched_agreement" in q:
                p(f'A control that matches edge direction and distance gives {pct(q["distance_matched_agreement"])} expected agreement.')
            table(("Human role","Supported sentences","Semantic-neighbor agreement","Uniform remote agreement","Distance- and direction-matched agreement"),[
                (name.replace("_"," "),f'{r["sentences"]:,}',pct(r["semantic_neighbor_agreement"]),pct(r["uniform_remote_agreement"]),
                 pct(r["distance_matched_agreement"]) if "distance_matched_agreement" in r else "Pending")
                for name,r in graph_affinity["role"].items()])
        if graph and graph_freeze and "independent" in graph["splits"]:
            p(f'We wrote linear role corrections from {len(graph_freeze["train_cases"]):,} training judgments. Candidates range from local scores to similarity-weighted role votes and two-hop document context. The frozen candidate uses {graph_freeze["choice"][0]} features.')
            table(("Cohort","Judgments","Selected candidate macro-F1","Sentence-joint macro-F1","Candidate accuracy","Changed labels"),[
                (group,r["cases"],f'{r["candidate"]["macro_f1"]:.5f}',f'{r["baseline"]["macro_f1"]:.5f}',pct(r["candidate"]["accuracy"]),f'{r["changed"]:,}')
                for group,r in graph["splits"].items()])
            if "test" in graph["splits"]:
                lo,hi=graph["splits"]["test"]["paired"]["macro_f1"]
                p(f'The official macro-F1 difference from sentence-joint has a case-paired 95% interval of {100*lo:+.2f} to {100*hi:+.2f} points.')
        graph_comparison=result("sentence-document-graph-comparison")
        if graph_comparison and graph and graph_comparison["freeze_sha256"]==graph["freeze_sha256"] and "test" in graph_comparison["splits"]:
            q=graph_comparison["splits"]["test"]
            table(("Frozen features","Official macro-F1","Official accuracy","Changed labels"),[
                (name,f'{x["scores"]["macro_f1"]:.5f}',pct(x["scores"]["accuracy"]),f'{x["changed"]:,}')
                for name,x in q["arms"].items()])
            p("Each feature set uses its own settings chosen on validation. The comparison uses the same input sentences and human labels throughout.")
            a,b=q["arms"]["two-hop"]["paired_to_local"]["macro_f1"]
            p(f'The two-hop readout has the highest official point estimate. Its paired difference from the validation-selected local readout spans {100*a:+.3f} to {100*b:+.3f} macro-F1 points.')
            local=q["arms"]["local"]
            lo,hi=local["paired_to_aligned_archive"]["macro_f1"]
            p(f'The local readout’s paired difference against the best aligned author archive has a 95% interval of {100*lo:+.2f} to {100*hi:+.2f} macro-F1 points.')
            portable=result("sentence-local-readout-portable-qa")
            if portable and portable["scores"]==local["scores"] and local["paired_to_baseline"]["macro_f1"][0]>0:
                p(f'The sentence-local CPU profile reproduced all {portable["official_cases"]} official cases with the same frozen encoder. Live inference also reproduced the per-sentence predictions on {len(portable["live_cases"])} judgments.')
            doc_portable=result("sentence-document-readout-portable-qa")
            if doc_portable and doc_portable["scores"]==q["arms"]["two-hop"]["scores"]:
                p("The optional sentence-document profile uses the frozen two-hop readout. It reproduced the official scores and live predictions with the CPU implementation.")
                table(("Sentences in judgment","Local seconds","Two-hop seconds"),[
                    (v["sentences"],f'{v["cpu_seconds"]["sentence-local"]:.3f}',f'{v["cpu_seconds"]["sentence-document"]:.3f}')
                    for v in doc_portable["latency"]])
        reserve=result("sentence-graph-holdout")
        reserve_freeze=result("sentence-graph-holdout-frozen")
        if reserve and reserve_freeze:
            h(3,"Fresh validation judgments")
            p(f'The {reserve["cases"]} validation judgments contain {reserve["baseline"]["rows"]:,} supplied sentences. The sample excludes cases logged by earlier experiments or found in encoder caches and removes exact duplicates. It includes an equal number of judgments from each length quartile. Weights and settings were frozen before the sample was evaluated.')
            table(("Readout","Macro-F1","Accuracy","Paired macro-F1 difference from local"),[
                ("sentence-joint",f'{reserve["baseline"]["macro_f1"]:.5f}',pct(reserve["baseline"]["accuracy"]),""),
                *[(name,f'{v["scores"]["macro_f1"]:.5f}',pct(v["scores"]["accuracy"]),
                  " to ".join(f'{100*x:+.3f}' for x in v["paired_to_local"]["macro_f1"])+" points")
                  for name,v in reserve["arms"].items()]])
        links(("LegalSeg sentence graph","https://aclanthology.org/2025.findings-naacl.63/"),
              ("Similarity graphs and label propagation","https://www.aladdin.cs.cmu.edu/papers/pdfs/y2003/zgl.pdf"))
    context_budget=result("sentence-context-budget")
    if context_budget:
        h(2,"How much judgment context do the released sentence profiles need?")
        p(f'The benchmark re-encodes the same {context_budget["source_sentences"]:,} supplied sentences from {context_budget["cases"]} held-out judgments. Each shorter input consists of consecutive sentences scored within that excerpt. The block boundaries are controlled and may differ from the original paragraphs.')
        table(("Maximum sentences per input","Profile","Macro-F1","Accuracy","Difference from full macro-F1 at 95% confidence"),[
            ("Full judgment" if n=="None" else n,profile,f'{v["scores"]["macro_f1"]:.5f}',pct(v["scores"]["accuracy"]),
             " to ".join(f'{100*x:+.2f}' for x in v["paired_to_full"]["macro_f1"])+" points" if "paired_to_full" in v else "")
            for n in ("1","4","16","64","None") for profile,v in context_budget["scores"][n].items()])
        p(f'The full-judgment scores reproduce the prior held-out results. The actual CPU interface also reproduced the re-encoded predictions in {len(context_budget["checked_live_chunks"])} sample chunks.')
        links(("LegalSeg context experiments","https://aclanthology.org/2025.findings-naacl.63/"),
              ("Context retrieval for long legal documents","https://aclanthology.org/2025.argmining-1.15/"))
    capacity=result("sentence-input-size")
    if capacity:
        h(3,"Input size tested on CPU")
        p(f'The {capacity["encoder_parameters"]:,}-parameter hierarchical encoder has no fixed context-window setting. It tokenizes each supplied sentence into words and reads the sentence vectors in order. Practical input size depends on memory and runtime. The document graph compares pairs of sentence representations in batches, so its compute grows quadratically with sentence count.')
        table(("Largest observed input","Judgment","Word tokens","Supplied sentences","Output labels"),[
            (v["purpose"],v["case"],f'{v["tokens"]:,}',f'{v["sentences"]:,}',f'{v["labels_returned"]:,}')
            for v in capacity["probes"]])
        p("These are tested input sizes from the official split. Each input returned one role per supplied sentence. The tokenizer counts words after splitting ASCII punctuation.")
    isolated=result("isolated-role-readout")
    isolated_freeze=result("isolated-role-readout-frozen")
    if isolated and isolated_freeze and "test" in isolated["splits"]:
        h(2,"Writing a classifier for isolated sentences")
        method=isolated_freeze["choice"]
        p(f'The frozen encoder sees one sentence per input. A closed-form readout is chosen from recurrent states, lexical log-ratios and simple text cues. The labeled readout fit uses {len(isolated_freeze["training_cases"]):,} training judgments outside the lexical-memory fit. Validation selected {method[0]} features with {method[1] if len(method)>1 else "the previous weights"}.')
        table(("Cohort","Judgments","New isolated macro-F1","Previous isolated macro-F1","New isolated accuracy"),[
            (group,r["cases"],f'{r["candidate"]["macro_f1"]:.5f}',f'{r["baseline"]["macro_f1"]:.5f}',pct(r["candidate"]["accuracy"]))
            for group,r in isolated["splits"].items()])
        official=isolated["splits"]["test"]
        lo,hi=official["paired"]["macro_f1"]
        p(f'The 95% paired macro-F1 difference against the previous isolated hierarchy ranges from {100*lo:+.2f} to {100*hi:+.2f} points.')
        archive=result("known-sentence-numerical-archives")
        if archive:
            ref=archive["archives"]["standalone"]["scores"]
            p(f'The aligned one-sentence InLegalBERT archive scores {ref["macro_f1"]:.5f} macro-F1 and {pct(ref["accuracy"])} accuracy on these same nonblank test sentences.')
        if context_budget and isolated["splits"]["holdout"]["candidate"]["rows"]==context_budget["source_sentences"]:
            single=isolated["splits"]["holdout"]["candidate"]["macro_f1"]
            four=context_budget["scores"]["4"]["sentence-document"]["scores"]["macro_f1"]
            sixteen=context_budget["scores"]["16"]["sentence-document"]["scores"]["macro_f1"]
            p(f'On the same held-out rows, the isolated readout gives {single:.5f} macro-F1. The document profile gives {four:.5f} with blocks of up to four sentences and {sixteen:.5f} with up to sixteen.')
        isolated_qa=result("isolated-role-readout-portable-qa")
        if isolated_qa and isolated_qa["scores"]==official["candidate"]:
            a,b=isolated_qa["paired_to_standalone_author"]["macro_f1"]
            p(f'The optional sentence-isolated profile reproduces the official scores on CPU and treats each entry in a batch independently. Its paired macro-F1 difference from the aligned author archive spans {100*a:+.2f} to {100*b:+.2f} points.')
            if archive:
                rlo,rhi=isolated_qa["paired_to_standalone_author"]["reasoning"]
                p(f'Reasoning is the strongest isolated comparison with the author archive. Its F1 rises from {ref["per_role_f1"]["reasoning"]:.4f} to {official["candidate"]["per_role_f1"]["reasoning"]:.4f}, with a paired 95% gain of {100*rlo:+.2f} to {100*rhi:+.2f} points.')
    fusion=result("sentence-readout-fusion")
    fusion_freeze=result("sentence-readout-fusion-frozen")
    if fusion and fusion_freeze and "test" in fusion["splits"]:
        h(2,"Adding isolated sentence evidence to the document readout")
        p(f'The frozen recurrent encoder sees sentences both in the judgment and independently. A direct probability blend and two closed-form calibrators predict one role per supplied sentence. Both calibrators use the same {fusion_freeze["fit_rows"]:,} rows from {len(fusion_freeze["cases_for_fit"])} official training judgments outside the graph, isolated readout and lexical-memory fits. The document-only calibrator measures how much of any gain comes from recalibrating the existing document scores.')
        p("Each arm was selected on the same 72 validation judgments by macro-F1 with an accuracy floor. Confirmation, the 384-case held-out validation cohort and the official test did not affect the weights or settings.")
        table(("Cohort","Method","Macro-F1","Accuracy","Changed labels","Paired 95% difference from sentence-document"),[
            (group,method,f'{v["macro_f1"]:.5f}',f'{100*v["accuracy"]:.3f}%',count,
             " to ".join(f'{100*x:+.3f}' for x in ci)+" points" if ci else "")
            for group,r in fusion["splits"].items()
            for method,v,count,ci in [("sentence-document",r["base"],0,None),
                *[(name,a["scores"],a["changed"],a["paired_to_base"]["macro_f1"]) for name,a in r["arms"].items()]]])
        official=fusion["splits"]["test"]
        lo,hi=official["paired_isolated_contribution"]["macro_f1"]
        p(f'On the official test, the paired 95% macro-F1 difference between the calibrator with both inputs and the document-only calibrator is {100*lo:+.3f} to {100*hi:+.3f} points.')
        document_score=official["arms"]["document only"]
        dlo,dhi=document_score["paired_to_base"]["accuracy"]
        ilo,ihi=official["paired_isolated_contribution"]["accuracy"]
        p(f'The document-only calibrator gets {document_score["scores"]["correct"]-official["base"]["correct"]:,} additional sentences correct on the official test, with a paired accuracy gain of {100*dlo:+.3f} to {100*dhi:+.3f} points. Adding isolated evidence gets {official["arms"]["document and isolated"]["scores"]["correct"]-document_score["scores"]["correct"]:,} more point-estimate predictions correct, with a paired accuracy difference of {100*ilo:+.3f} to {100*ihi:+.3f} points.')
        roles=("none","facts","issue","petitioner_argument","respondent_argument","reasoning","decision")
        table(("Official-test method","None F1","Facts F1","Issue F1","Petitioner F1","Respondent F1","Reasoning F1","Decision F1","Paired macro-F1 difference from the aligned author archive"),[
            ("sentence-document",*(f'{official["base"]["per_role_f1"][role]:.4f}' for role in roles),""),
            *[(name,*(f'{r["scores"]["per_role_f1"][role]:.4f}' for role in roles),
               " to ".join(f'{100*x:+.3f}' for x in r["paired_to_aligned_author"]["macro_f1"])+" points")
              for name,r in official["arms"].items()]])
        references=result("known-sentence-numerical-archives")
        if references:
            highest=max(references["archives"].items(),key=lambda item:item[1]["scores"]["accuracy"])
            ref=references["archives"][references["best_by_macro_f1"]]["scores"]
            lo,hi=document_score["paired_to_aligned_author"]["accuracy"]
            p(f'The previous-predicted-role InLegalBERT archive leads the aligned author files at {ref["macro_f1"]:.5f} macro-F1 and scores {100*ref["accuracy"]:.2f}% accuracy. ToInLegalBERT has the highest aligned archive accuracy at {100*highest[1]["scores"]["accuracy"]:.2f}%. Compared with the macro-F1 leader, the document-only edit has a paired 95% accuracy gain of {100*lo:+.2f} to {100*hi:+.2f} points.')
        deployed=result("sentence-document-calibrated-portable-qa")
        if deployed and deployed["scores"]==document_score["scores"]:
            p(f'The optional sentence-document-calibrated profile reproduced all {deployed["cases"]:,} official judgments through the CPU predictor, including live per-sentence parity checks. It uses the existing document encoder and an additional {deployed["tensors"]["document_calibration.matrix"][0]}-feature readout.')
        links(("Stacked generalization","https://doi.org/10.1016/s0893-6080%2805%2980023-1"),
              ("LegalSeg paper","https://aclanthology.org/2025.findings-naacl.63/"))
    bert_referral=result("sentence-bert-referral")
    bert_freeze=result("sentence-bert-referral-frozen")
    if bert_referral and bert_freeze:
        h(2,"Referring uncertain sentences to InLegalBERT")
        p(f'The {bert_freeze["encoder_parameters"]:,}-parameter public LegalSeg InLegalBERT checkpoint supplies frozen activations for selected sentences. The public checkpoint named for the current sentence is byte-identical to the release named for the previous sentence and its true label. The two-predecessor release has distinct weights. This comparison holds weights fixed across both input formats and supplies no human labels at inference. The backbone supports 512 WordPiece positions including special tokens. The measured referral recipe uses at most {bert_freeze["max_tokens"]} per call. Full judgment context enters through the AttoFlow document model.')
        p(f'The referral rule prioritizes ambiguous assignments involving issue or respondent argument and selects at most {bert_freeze["max_referrals_per_judgment"]} sentences per judgment. Validation considers native BERT scores, hidden features, class prototypes and a direct seven-row classifier replacement. A closed-form seven-role readout fits {bert_freeze["training_referred_rows"]:,} referred training sentences across {len(bert_freeze["fit_cases"])} cases. The control fits the same judgments and receives the same referrals and document features.')
        floor=bert_freeze["selection_baseline"]["accuracy"]-bert_freeze["accuracy_floor_points"]/100
        descriptions={"document":"Document features","scores":"Native BERT scores","hidden":"Native scores and pooled hidden features",
                      "prototypes":"Native scores and class prototypes","direct head":"Direct BERT classifier replacement"}
        selected_variants=[]
        for arm,runs in bert_freeze["selection_grid"].items():
            for variant in dict.fromkeys(r[0][0] for r in runs):
                settings,macro,accuracy=max((r for r in runs if r[0][0]==variant and r[2]>=floor),
                                            key=lambda r:(r[1],r[2],r[0]))
                selected_variants.append((arm,descriptions[variant],f'{macro:.5f}',f'{100*accuracy:.3f}%',
                                          "Yes" if settings==bert_freeze["frozen_modes"][arm] else ""))
        table(("Candidate","Selection readout","Selection macro-F1","Selection accuracy","Frozen for confirmation"),selected_variants)
        table(("Cohort","Method","Referred fraction","Macro-F1","Accuracy","Issue F1","Respondent argument F1","Paired macro-F1 change against parent"),[
            (group,name,f'{100*r["referred"]/r["base"]["rows"]:.2f}%',f'{v["macro_f1"]:.5f}',f'{100*v["accuracy"]:.3f}%',
             f'{v["per_role_f1"]["issue"]:.4f}',f'{v["per_role_f1"]["respondent_argument"]:.4f}',
             " to ".join(f'{100*x:+.3f}' for x in ci)+" points" if ci else "")
            for group,r in bert_referral["splits"].items()
            for name,v,ci in [("sentence-document-calibrated",r["base"],None),
                *[(arm,a["scores"],a["paired_to_parent"]["macro_f1"]) for arm,a in r["arms"].items()]]])
        table(("Cohort","Method","Changed labels","Paired macro-F1 change versus document-only control"),[
            (group,arm,a["changed"],
             " to ".join(f'{100*x:+.3f}' for x in a["paired_to_same_referral_document_control"]["macro_f1"])+" points")
            for group,r in bert_referral["splits"].items() for arm,a in r["arms"].items()
            if arm!="document control"])
        p("The oracle is a diagnostic that corrects every routed error with its human label. Its gap from the actual edit measures how much useful evidence remains uncaptured within the routing budget.")
        table(("Cohort","Current macro-F1","Oracle macro-F1","Current accuracy","Oracle accuracy"),[
            (group,f'{r["base"]["macro_f1"]:.5f}',f'{r["oracle"]["macro_f1"]:.5f}',
             f'{100*r["base"]["accuracy"]:.3f}%',f'{100*r["oracle"]["accuracy"]:.3f}%')
            for group,r in bert_referral["splits"].items()])
        role_keys=("issue","respondent_argument")
        table(("Cohort","Role","Errors reached by referral","Total baseline errors","Error coverage"),[
            (group,name.replace("_"," "),r["referral_errors_by_role"][i],r["total_true_by_role"][i]-r["base"]["confusion"][i][i],
             f'{100*r["referral_errors_by_role"][i]/max(1,r["total_true_by_role"][i]-r["base"]["confusion"][i][i]):.1f}%')
            for group,r in bert_referral["splits"].items() for i,name in zip((2,4),role_keys)])
        execution=result("sentence-bert-referral-backend-parity")
        if execution and execution["frozen_sha256"]==bert_referral["freeze_sha256"]:
            p(f'The CPU and Intel graphics backends were compared on {execution["vectors"]} encoded validation inputs. Switching those inputs to CPU changed {execution["changed_labels"]} final labels. The largest difference in candidate readout scores was {execution["max_readout_error"]:.2g}.')
        cli_check=result("sentence-bert-referral-cli-qa")
        if cli_check and cli_check["freeze_sha256"]==bert_referral["freeze_sha256"]:
            p(f'The experimental CPU predictor accepts a JSON array of sentences and returns one label per sentence. It reproduced the cached benchmark predictions, including changed labels, on {len(cli_check["verified_cases"])} live judgments. The normal CPU release retains the smaller document model.')
        official=bert_referral["splits"].get("test")
        if official and "author_archive" in official:
            best=official["arms"]["BERT previous two and target"]
            changes=best["paired_priority_roles_to_parent"]
            issue=" to ".join(f'{100*x:+.2f}' for x in changes["issue"])
            response=" to ".join(f'{100*x:+.2f}' for x in changes["respondent_argument"])
            gain=best["scores"]["correct"]-official["base"]["correct"]
            p(f'The contextual BERT edit gets {gain:,} additional official test sentences correct. The paired 95% gains are {issue} points for issue F1 and {response} points for respondent-argument F1. The label oracle for this routing budget reaches {official["oracle"]["macro_f1"]:.5f} macro-F1. Reaching the published 0.770 would require improvements on sentences outside the current referral set.')
            if "priority_errors_predicted_reasoning" in official:
                missed=official["priority_errors_predicted_reasoning"]
                p(f'The routing rule preserves every existing reasoning label. That choice excludes {missed["issue"]:,} missed issues and {missed["respondent_argument"]:,} missed respondent arguments that the baseline labeled reasoning.')
            table(("Official-test method","Macro-F1","Accuracy","Paired macro-F1 change from the aligned author archive"),[
                (official["author_archive"]["name"],f'{official["author_archive"]["scores"]["macro_f1"]:.5f}',
                 f'{100*official["author_archive"]["scores"]["accuracy"]:.3f}%',""),
                *[(arm,f'{a["scores"]["macro_f1"]:.5f}',f'{100*a["scores"]["accuracy"]:.3f}%',
                   " to ".join(f'{100*x:+.3f}' for x in a["paired_to_author_archive"]["macro_f1"])+" points")
                  for arm,a in official["arms"].items()]])
        links(("LegalSeg paper and InLegalBERT input modes","https://aclanthology.org/2025.findings-naacl.63/"),
              ("InLegalBERT pretraining corpus","https://huggingface.co/law-ai/InLegalBERT"))
    dense=result("sentence-dense-readout")
    dense_plan=result("sentence-dense-readout-frozen")
    if dense and dense_plan:
        h(2,"Writing readouts for every sentence")
        p(f'The 16.82M-parameter encoder remains frozen. All three readouts fit {dense_plan["fit_rows"]:,} sentences from {len(dense_plan["fit_cases"])} official training judgments. The control uses the existing model scores. The multiscale readout adds the two-hop graph channels, lexical cues and separate before-and-after windows of 2, 8, 24 and 64 sentences. A second readout adds fixed nonlinear features and interactions with the current role. Each ridge solve tests a replacement score or a residual correction.')
        form={"current scores":"Current scores","multiscale":"Multiscale linear","nonlinear":"Multiscale nonlinear"}
        table(("Readout","Input features","Selected score","Reasoning labels eligible","Selection macro-F1"),[
            (form[arm],v["features"],v["selection"][3],"No" if v["selection"][-1] else "Yes",
             f'{v["selection_macro_f1"]:.5f}') for arm,v in dense_plan["arms"].items()])
        reserve=result("sentence-dense-readout-prospective")
        if reserve and "prospective" in dense["splits"]:
            p(f'A second holdout contains {reserve["cohort_size"]} judgments selected by length before fitting. Eligible cases excluded exact duplicate judgment text within validation and judgment IDs from previous benchmark receipts and encoder caches.')
        table(("Cohort","Method","Macro-F1","Accuracy","Errors fixed","Correct labels changed incorrectly","Paired macro-F1 change from released model"),[
            (group,arm,f'{q["macro_f1"]:.5f}',f'{100*q["accuracy"]:.3f}%',fixed,broken,
             " to ".join(f'{100*x:+.3f}' for x in ci)+" points" if ci else "")
            for group,split in dense["splits"].items()
            for arm,q,fixed,broken,ci in [("Released document calibration",split["parent"],0,0,None),
                *[(form[arm],r["scores"],r["fixed"],r["regressed"],r["paired_parent"]["macro_f1"])
                  for arm,r in split["arms"].items()]]])
        if "test" in dense["splits"]:
            rows=dense["splits"]["test"]["arms"]
            table(("Official-test readout","Issue F1","Respondent-argument F1","Paired macro-F1 difference from aligned archive"),[
                (form[k],f'{r["scores"]["per_role_f1"]["issue"]:.5f}',f'{r["scores"]["per_role_f1"]["respondent_argument"]:.5f}',
                 " to ".join(f'{100*x:+.3f}' for x in r["paired_author"]["macro_f1"])+" points")
                for k,r in rows.items()])
        unrestricted=result("sentence-dense-unrestricted")
        if unrestricted:
            h(3,"Allowing revisions to all seven roles")
            p("The unrestricted comparison freezes the highest-scoring unguarded setting on selection. Existing reasoning predictions are eligible for revision in every judgment.")
            table(("Cohort","Unrestricted macro-F1","Unrestricted accuracy","Previously predicted reasoning labels revised","Paired macro-F1 change"),[
                (g,f'{r["scores"]["macro_f1"]:.5f}',f'{100*r["scores"]["accuracy"]:.3f}%',r["reasoning_predictions_revised"],
                 " to ".join(f'{100*x:+.3f}' for x in r["paired_parent"]["macro_f1"])+" points")
                for g,r in unrestricted["splits"].items()])
        live=result("sentence-dense-readout-cli-qa")
        if live and live["freeze_sha256"]==dense["freeze_sha256"]:
            p(f'The experimental CPU predictor reproduced the {form[live["arm"]].lower()} benchmark predictions, including edited labels, on {len(live["cases"])} live judgments. It accepts ordered sentences from one judgment and returns one role per entry.')
            if "issue_guard" in live:
                p(f'The preserve-issues CPU option also matched the benchmark labels on a live {live["issue_guard"]["sentences"]}-sentence judgment, including {live["issue_guard"]["preserved"]} labels retained by that option.')
        combined=result("sentence-dense-referral")
        combined_plan=result("sentence-dense-referral-frozen")
        if combined and combined_plan:
            arm,rule=combined_plan["choice"]
            h(3,"Testing dense and InLegalBERT edits together")
            p(f'The agreement and priority rules were compared on selection judgments. The frozen choice was {rule.lower()} with the {form[arm].lower()} readout.')
            table(("Cohort","BERT referral macro-F1","Combined macro-F1","Combined accuracy","Labels changed beyond BERT","Paired macro-F1 change"),[
                (g,f'{r["bert"]["macro_f1"]:.5f}',f'{r["candidate"]["macro_f1"]:.5f}',f'{100*r["candidate"]["accuracy"]:.3f}%',
                 r["changed_vs_bert"]," to ".join(f'{100*x:+.3f}' for x in r["paired_bert"]["macro_f1"])+" points")
                for g,r in combined["splits"].items()])
        issue_guard=result("sentence-dense-issue-guard")
        if issue_guard:
            h(3,"Preserving issue labels")
            p("The official test exposed an issue F1 loss in the dense edit. This follow-up keeps every decision about issue membership from the incumbent. The test and earlier validation replays are exploratory for this rule. The new validation cohort was reserved separately and excludes the first prospective cohort.")
            table(("Cohort","Method","Macro-F1","Accuracy","Issue F1","Paired macro-F1 change from that method’s incumbent"),[
                (g,label,f'{r["scores"]["macro_f1"]:.5f}',f'{100*r["scores"]["accuracy"]:.3f}%',
                 f'{r["scores"]["per_role_f1"]["issue"]:.5f}'," to ".join(f'{100*x:+.3f}' for x in r["paired"]["macro_f1"])+" points")
                for g,k in issue_guard["splits"].items()
                for label,r in [("Dense, preserve issue",k["guarded_dense"]),
                    *([("BERT referrals with dense edits elsewhere",k["combined"])] if "combined" in k else [])]])
            prospect=issue_guard["splits"].get("role-guard")
            if prospect:
                raw=prospect["unguarded_dense"]["scores"]
                filtered=prospect["guarded_dense"]["scores"]
                low,high=prospect["guarded_vs_dense"]["paired"]["macro_f1"]
                p(f'On the second prospective cohort, the dense edit scored {raw["macro_f1"]:.5f} macro-F1 before preserving issue labels and {filtered["macro_f1"]:.5f} afterward. The paired macro-F1 difference was {100*low:+.3f} to {100*high:+.3f} points.')
            oldtest=issue_guard["splits"].get("test",{})
            if "combined_vs_aligned_author" in oldtest:
                lo,hi=oldtest["combined_vs_aligned_author"]["macro_f1"]
                ref=oldtest["aligned_author"]["scores"]
                p(f'The aligned author archive scores {ref["macro_f1"]:.5f} macro-F1. The exploratory combined candidate’s paired 95% difference is {100*lo:+.3f} to {100*hi:+.3f} points on the official test.')
        links(("Legal judgment sequence modeling","https://www.ltg.ed.ac.uk/np/publications/ltg/papers/Hachey2005Sequence.pdf"),
              ("Context retrieval in long legal documents","https://aclanthology.org/2025.argmining-1.15/"),
              ("Fixed nonlinear reservoirs and linear readouts","https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Jaeger_RNN_tutorial.pdf"))
    recurrent=result("sentence-recurrent-inverse")
    recurrent_plan=result("sentence-recurrent-inverse-frozen")
    if recurrent and recurrent_plan:
        h(2,"Editing weights inside the bidirectional LSTM")
        source=recurrent_plan["source"]
        p(f'This experiment isolates an internal weight edit in the {16816110:,}-parameter hierarchy. It builds role-score corrections for {source["incumbent_errors"]:,} released-model errors across {source["rows"]:,} training sentences from {len(source["cases"])} judgments. Correctly classified training sentences have a zero correction target.')
        p(f'The same targets fit a readout-only control. For the LSTM variants, a damped inverse projects the requested score change into hidden-state space. A one-step gate inverse maps the hidden-state change to a candidate-cell preactivation change. Reconstructing the original hidden states from the documented gate equations had a maximum absolute error of {source["max_baseline_lstm_reconstruction_error"]:.2g}. Closed-form regression writes the native candidate-cell weight rows and biases in both directions.')
        if any(not v["within_accuracy_floor"] for v in recurrent_plan["arms"].values()):
            p("When every setting for a method missed the selection accuracy floor of 0.10 percentage points below the released model, the strongest macro-F1 setting was kept as a diagnostic.")
        table(("Method","Native tensors written","Values replaced","Selection macro-F1","Selection accuracy"),[
            (k,len(v["tensors"]),v["values_replaced"],f'{v["select_macro_f1"]:.5f}',f'{100*v["select_accuracy"]:.3f}%')
            for k,v in recurrent_plan["arms"].items()])
        checked=result("sentence-recurrent-inverse-weight-qa")
        if checked and checked["freeze_sha256"]==recurrent["freeze_sha256"]:
            p("Tensor checks verified that the native edit touches the candidate-state rows in both directions. Three live judgments confirmed that the gate writes change the hidden states. The matched output-layer write affects the seven native emission rows.")
        shown={"parent":"Released document calibration","dense parent":"Dense readout","dense with gate":"Dense readout with edited gates"}
        table(("Cohort","Method","Macro-F1","Accuracy","Changed labels","Paired macro-F1 change from that method’s incumbent"),[
            (group,shown.get(name,name),f'{scores["macro_f1"]:.5f}',f'{100*scores["accuracy"]:.3f}%',n,
               " to ".join(f'{100*x:+.3f}' for x in ci)+" points" if ci else "")
            for group,d in recurrent["splits"].items()
            for name,scores,n,ci in [("parent",d["parent"],0,None),("dense parent",d["dense_parent"],0,None),
                *[(k,v["scores"],v["changed"],v["paired"]["macro_f1"]) for k,v in d["arms"].items()]]])
        if "test" in recurrent["splits"]:
            current=recurrent["splits"]["test"]
            table(("Official-test method","Issue F1","Respondent-argument F1","Paired macro-F1 difference from readout-only","Paired macro-F1 difference from aligned archive"),[
                (shown.get(name,name),f'{v["scores"]["per_role_f1"]["issue"]:.5f}',f'{v["scores"]["per_role_f1"]["respondent_argument"]:.5f}',
                 " to ".join(f'{100*x:+.3f}' for x in v["paired_to_readout"]["macro_f1"])+" points" if "paired_to_readout" in v else "",
                 " to ".join(f'{100*x:+.3f}' for x in v["paired_author"]["macro_f1"])+" points") for name,v in current["arms"].items()])
        refit=result("sentence-recurrent-refit")
        refit_plan=result("sentence-recurrent-refit-frozen")
        if refit and refit_plan and refit_plan["source_gate_plan_sha256"]==recurrent["freeze_sha256"]:
            h(3,"Refitting the same head on each encoder")
            p(f'The two multiscale heads use the same ridge settings and {refit_plan["fit_rows"]:,} rows. Each head is fitted to its own encoder’s activations. This comparison checks whether the edited encoder can improve predictions when its downstream readout is recalibrated.')
            table(("Cohort","Original-state macro-F1","Edited-state macro-F1","Original-state accuracy","Edited-state accuracy","Paired macro-F1 change"),[
                (g,f'{d["original state"]["macro_f1"]:.5f}',f'{d["edited state"]["macro_f1"]:.5f}',
                   f'{100*d["original state"]["accuracy"]:.3f}%',f'{100*d["edited state"]["accuracy"]:.3f}%',
                   " to ".join(f'{100*x:+.3f}' for x in d["paired_gate_vs_original"]["macro_f1"])+" points")
                for g,d in refit["splits"].items()])
        budget=result("sentence-recurrent-context")
        if budget and budget["weight_freeze_sha256"]==recurrent["freeze_sha256"]:
            h(3,"Context budget for the recurrent weight edit")
            p(f'The same {budget["rows"]:,} sentences from {len(budget["cases"])} withheld judgments are classified as contiguous windows. The gate weights were frozen using full judgments. The short-window results measure transfer of the same weights and decoder.')
            table(("Maximum sentences per input","Released macro-F1","Edited macro-F1","Edited accuracy","Paired macro-F1 change"),[
              ("Full judgment" if k=="None" else k,f'{v["parent"]["macro_f1"]:.5f}',f'{v["edited"]["macro_f1"]:.5f}',
               f'{100*v["edited"]["accuracy"]:.3f}%'," to ".join(f'{100*x:+.3f}' for x in v["paired_to_parent"]["macro_f1"])+" points")
              for k,v in budget["windows"].items()])
        links(("Direct weight writes for classifiers","https://arxiv.org/abs/2112.01008"),
              ("Damped weight editing for factual associations","https://arxiv.org/abs/2202.05262"),
              ("PyTorch LSTM gate equations","https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html"))
    small=result("legal-bert-small-referral")
    small_plan=result("legal-bert-small-referral-frozen")
    if small and small_plan:
        h(2,"A smaller legal encoder on the same referrals")
        p(f'The pilot adds {small_plan["parameters"]:,} parameters from the six-layer LEGAL-BERT Small encoder to the existing full-judgment CPU model. Each referred input contains the target sentence and up to two predecessors within {small_plan["window_cap"]} WordPiece tokens. The readout averages only target-sentence states and fits {small_plan["rows_fit"]:,} training rows. The matched control receives the same document scores and referrals.')
        big=result("sentence-bert-referral")
        table(("Cohort","Method","Macro-F1","Accuracy","Paired macro-F1 change from released document calibration"),[
            (g,k,f'{v["macro_f1"]:.5f}',f'{100*v["accuracy"]:.3f}%',
                " to ".join(f'{100*x:+.3f}' for x in ci)+" points" if ci else "")
            for g,q in small["splits"].items()
            for k,v,ci in [("Released document calibration",q["parent"],None),
                *[(a,r["scores"],r["paired_parent"]["macro_f1"]) for a,r in q["models"].items()],
                *([("109M InLegalBERT contextual referral",big["splits"][g]["arms"]["BERT previous two and target"]["scores"],
                    big["splits"][g]["arms"]["BERT previous two and target"]["paired_to_parent"]["macro_f1"])] if big else [])]])
        verified_small=result("legal-bert-small-referral-qa")
        if verified_small and verified_small["freeze_sha256"]==small["freeze_sha256"]:
            lo,hi=verified_small["paired_small_vs_large"]["macro_f1"]
            p(f'The small encoder’s paired macro-F1 difference from the 109M reference is {100*lo:+.3f} to {100*hi:+.3f} points on the same confirmation judgments. The GPU and PyTorch CPU target pooling agreed within {verified_small["max_abs_pooling_drift"]:.2g} on {verified_small["checked_pooling_inputs"]} checked inputs.')
        links(("LEGAL-BERT paper","https://aclanthology.org/2020.findings-emnlp.261/"),
              ("Pinned LEGAL-BERT Small model","https://huggingface.co/nlpaueb/legal-bert-small-uncased/tree/0e23f7a9a39f59768ea7e09766d8ee308580fb17"))
    ambiguity=result("sentence-identical-input-audit")
    if ambiguity:
        h(2,"Identical text with different gold roles")
        p(f'This audit groups {ambiguity["rows"]:,} official test rows by exactly the information supplied to a deterministic classifier. Within each identical input group, predicting its most frequent gold role gives the maximum possible group accuracy. The resulting ceiling measures ambiguity in these observed inputs.')
        table(("Supplied input","Conflicting input groups","Rows in conflicting groups","Minimum unavoidable errors","Observed accuracy ceiling"),[
            (k,v["conflicting_inputs"],v["rows_in_conflicting_inputs"],v["minimum_errors"],f'{100*v["accuracy_ceiling"]:.3f}%')
            for k,v in ambiguity["modes"].items()])
        exact=ambiguity["modes"]["isolated sentence, exact"]["most_frequent_role_conflicts"]
        table(("Most frequent conflicting gold roles for identical sentences","Distinct inputs"),[
            (" and ".join(r["roles"]),r["inputs"]) for r in exact])
        links(("LegalSeg taxonomy and context benchmarks","https://aclanthology.org/2025.findings-naacl.63/"))
    phrases=result("sentence-phrase-readout")
    phrase_plan=result("sentence-phrase-readout-frozen")
    lexicon=result("sentence-phrase-readout-lexical-frozen")
    if phrases and phrase_plan and lexicon:
        h(2,"Writing a phrase readout with judgment context")
        p(f'The 16.82M-parameter judgment encoder is paired with {lexicon["columns"]:,} word and character n-gram features. The sparse readout learns seven role scores and seven corrections to the released model’s score distribution. Four judgment-level folds produce out-of-fold lexical scores for the {lexicon["fit_rows"]:,} training sentences. Two Cholesky solves per fold test different damping strengths.')
        if "layout_recovery" in lexicon:
            p("The provisional pass exposed a mismatch between the solver’s column-major arrays and the tensor serializer’s row-major layout. The lexical matrix was restored in the correct layout. All reported validation, test and context-window results were recomputed with the corrected weights.")
        lexical_proof=result("sentence-phrase-readout-lexical-qa")
        if lexical_proof and lexical_proof["file_sha256"]==lexicon["weights_sha256"]:
            p(f'The saved lexical weights passed an independent replay of the weighted normal equations over all {lexical_proof["fit_rows"]:,} fit rows. The normalized maximum equation residual was {lexical_proof["normalized_max_normal_equation_residual"]:.2g}.')
        p("The dense control and both phrase readouts use the same training judgments. The target-phrase arm receives scores for the sentence itself. The contextual arm also receives separate before-and-after averages over 2, 8, 24 and 64 supplied sentences. Validation selects the ridge strength, blend and role-protection rule.")
        table(("Method","Dense readout features","Role protection chosen","Selection macro-F1","Selection accuracy"),[
            (k,v["width"],v["choice"][-1],f'{v["select_macro_f1"]:.5f}',f'{100*v["select_accuracy"]:.3f}%')
            for k,v in phrase_plan["arms"].items()])
        reserve=result("sentence-dense-readout-phrase-prospect")
        if reserve:
            p(f'The new prospective cohort contains {reserve["cohort_size"]} judgments. The selection excluded IDs in earlier benchmark receipts and encoder caches, including both earlier 196-judgment cohorts.')
        table(("Cohort","Method","Macro-F1","Accuracy","Labels revised","Paired macro-F1 change from dense control"),[
            (g,k,f'{v["macro_f1"]:.5f}',f'{100*v["accuracy"]:.3f}%',n,
              " to ".join(f'{100*x:+.3f}' for x in ci)+" points" if ci else "")
            for g,q in phrases["splits"].items()
            for k,v,n,ci in [("Released document calibration",q["parent"],0,None),
                *[(k,v["scores"],v["changed"],v["paired_dense"]["macro_f1"]) for k,v in q["arms"].items()]]])
        if "test" in phrases["splits"]:
            q=phrases["splits"]["test"]
            table(("Official-test method","Issue F1","Respondent-argument F1","Paired macro-F1 difference from aligned author archive"),[
                (k,f'{v["scores"]["per_role_f1"]["issue"]:.5f}',f'{v["scores"]["per_role_f1"]["respondent_argument"]:.5f}',
                  " to ".join(f'{100*x:+.3f}' for x in v["paired_author"]["macro_f1"])+" points") for k,v in q["arms"].items()])
        referral_phrase=result("sentence-phrase-referral")
        referral_recipe=result("sentence-phrase-referral-frozen")
        if referral_phrase and referral_recipe and referral_recipe["phrase_freeze_sha256"]==phrases["freeze_sha256"]:
            h(3,"Pairing the phrase readout with BERT referrals")
            p("The fixed composition uses the BERT prediction for every referred sentence and the phrase edit elsewhere. Both phrase variants use their existing settings.")
            table(("Cohort","Phrase readout outside BERT referrals","BERT macro-F1","Combined macro-F1","Combined accuracy","Paired macro-F1 gain versus BERT"),[
                (g,a,f'{q["bert"]["macro_f1"]:.5f}',f'{w["scores"]["macro_f1"]:.5f}',f'{100*w["scores"]["accuracy"]:.3f}%',
                 " to ".join(f'{100*x:+.3f}' for x in w["paired_bert"]["macro_f1"])+" points")
                for g,q in referral_phrase["splits"].items() for a,w in q["arms"].items()])
        phrase_context=result("sentence-phrase-context")
        if phrase_context and phrase_context["freeze_sha256"]==phrases["freeze_sha256"]:
            h(3,"Transferring the phrase readout to shorter inputs")
            p(f'The same {phrase_context["rows"]:,} sentences from {len(phrase_context["cases"])} judgments were presented as full judgments and consecutive sentence windows. This is a controlled context test using sentence boundaries.')
            table(("Maximum sentences per input","Released macro-F1","Phrase-edited macro-F1","Phrase-edited accuracy"),[
                ("Full judgment" if k=="None" else k,f'{v["parent"]["macro_f1"]:.5f}',f'{v["edited"]["macro_f1"]:.5f}',
                 f'{100*v["edited"]["accuracy"]:.3f}%') for k,v in phrase_context["windows"].items()])
        short_phrases=result("sentence-phrase-short-validation")
        short_receipt=result("sentence-phrase-short-validation-frozen")
        if short_phrases and short_receipt and short_receipt["model_freeze_sha256"]==phrases["freeze_sha256"]:
            h(3,"Four-sentence transfer on a separate cohort")
            p(f'The separate diagnostic uses {short_phrases["cases"]} judgments sampled by length from the new cohort. It compares the frozen dense and contextual phrase readouts with the released decoder on the same four-sentence windows.')
            table(("Method","Macro-F1","Accuracy"),[
                (m,f'{q["macro_f1"]:.5f}',f'{100*q["accuracy"]:.3f}%') for m,q in short_phrases["metrics"].items()])
            lo,hi=short_phrases["paired_phrases_to_dense"]["macro_f1"]
            p(f'The paired 95% difference from the matched dense readout is {100*lo:+.2f} to {100*hi:+.2f} macro-F1 points.')
        lexical_only=result("sentence-phrase-standalone")
        if lexical_only:
            h(3,"Classifying with phrase weights alone")
            p("This benchmark uses the phrase coefficient table by itself. One mode receives the supplied sentence. The second also pools lexical scores from up to two neighboring sentences on each side. Smoothing and class-prior settings were selected before confirmation or test.")
            table(("Cohort","Method","Macro-F1","Accuracy","Paired macro-F1 difference from phrase-only sentence input"),[
                (g,mode,f'{v["macro_f1"]:.5f}',f'{100*v["accuracy"]:.3f}%',
                     " to ".join(f'{100*x:+.3f}' for x in ci)+" points" if ci else "")
                for g,q in lexical_only["splits"].items()
                for mode,v,ci in [*[(m,r["scores"],r.get("paired_to_isolated",{}).get("macro_f1"))
                                      for m,r in q["modes"].items()],
                                   *([("Released isolated encoder",q["previous_isolated"],None)] if "previous_isolated" in q else []),
                                   *([("Aligned author sentence-only archive",q["aligned_isolated_author"],None)] if "aligned_isolated_author" in q else [])]])
            isolated_test=lexical_only["splits"].get("test",{})
            if "aligned_isolated_author" in isolated_test:
                lo,hi=isolated_test["modes"]["sentence only"]["paired_to_isolated_author"]["macro_f1"]
                p(f'The phrase-only sentence classifier’s paired macro-F1 difference from the aligned author sentence-only archive is {100*lo:+.3f} to {100*hi:+.3f} points on the official test.')
        cli_phrase=result("sentence-phrase-readout-cli-qa")
        if cli_phrase and cli_phrase["freeze_sha256"]==phrases["freeze_sha256"]:
            p(f'The experimental CPU command reproduced the {cli_phrase["arm"]} benchmark labels for {len(cli_phrase["cases"])} live judgments.')
        links(("Sequence modeling for legal sentence classification","https://www.ltg.ed.ac.uk/np/publications/ltg/papers/Hachey2005Sequence.pdf"),
              ("Baselines and Bigrams","https://aclanthology.org/P12-2018/"))
    sequenced=phrase_sequence_score
    plan=result("sentence-phrase-sequence-frozen")
    if sequenced and plan:
        h(2,"Choosing sentence roles together across a judgment")
        p(f'The sequence edit starts with frozen phrase readout scores and fits weights on {plan["training_edges"]:,} adjacent sentence pairs from {len(plan["train_cases"])} training judgments. The three candidate edge models use stationary role-pair counts, a learned probability of a role change, or a conditional 49-pair score matrix. The selection cohort chooses a recipe for sentence macro-F1 and a recipe for exact labeled span F1. A zero-edge replay reproduced the independent phrase labels.')
        table(("Profile","Transition features","Strength","Same-role bonus","Preserve reasoning membership"),[
            (name,w[0],w[1],w[2],"Yes" if w[3] else "No") for name,w in plan["rules"].items()])
        table(("Cohort","Method","Sentence macro-F1","Sentence accuracy","Exact labeled span F1","Role boundary F1","Paired exact span F1 change"),[
            (group,name,f'{q["labels"]["macro_f1"]:.5f}',f'{100*q["labels"]["accuracy"]:.3f}%',
             f'{q["spans"]["exact_labeled_span_f1"]:.5f}',f'{q["spans"]["role_boundary_f1"]:.5f}',
             " to ".join(f'{100*x:+.3f}' for x in q["span_intervals"]["exact_labeled_span_f1"])+" points" if "span_intervals" in q else "")
            for group,d in sequenced["splits"].items()
            for name,q in [("Independent phrase labels",d["baseline"]),*d["arms"].items()]])
        duration=result("sentence-phrase-duration-frozen")
        duration_score=result("sentence-phrase-duration")
        if duration and duration_score and duration["source_sequence_sha256"]==sequenced["freeze_sha256"]:
            p(f'The semi-Markov check estimates role change hazards from {duration["fit"]["gold_runs"]:,} gold spans with an age cap of {duration["age_cap"]} sentences. The decoder matched exhaustive path scoring on four short test lengths. For the role profile, validation selected a duration strength of {duration["choices"]["roles"]:g}. For the span profile it selected {duration["choices"]["spans"]:g}.')
        with_bert=phrase_referral_score
        with_bert_plan=result("sentence-phrase-sequence-referral-frozen")
        if with_bert and with_bert_plan and with_bert_plan["source_sequence_sha256"]==sequenced["freeze_sha256"]:
            h(3,"Adding sequence weights to the BERT profiles")
            p("The fixed composition anchors the selected span decoder to the earlier BERT referral and exploratory issue-protected hybrid. The rule protects BERT-referred sentences and issue membership. It adds no fitted coefficients.")
            table(("Cohort","Starting model","Baseline macro-F1","Edited macro-F1","Edited accuracy","Paired macro-F1 change"),[
                (g,n,f'{q["baseline"]["macro_f1"]:.5f}',f'{q["labels"]["macro_f1"]:.5f}',
                 f'{100*q["labels"]["accuracy"]:.3f}%'," to ".join(f'{100*x:+.3f}' for x in q["paired_labels"]["macro_f1"])+" points")
                for g,d in with_bert["splits"].items() for n,q in d["arms"].items()])
            official=with_bert["splits"].get("test",{}).get("arms",{}).get("issue-protected hybrid")
            if official:
                lo,hi=official["paired_author"]["macro_f1"]
                p(f'The edited hybrid’s paired 95% macro-F1 difference from the best aligned author archive is {100*lo:+.3f} to {100*hi:+.3f} points. The hybrid includes the earlier issue rule designed after inspecting official test errors.')
        archive=phrase_archive_score
        if archive and archive["sequence_freeze_sha256"]==sequenced["freeze_sha256"]:
            h(3,"Aligned author span projection with the same sentence cuts")
            q=archive["edited_projection"]
            old=archive["published_projection"]
            lo,hi=archive["paired_to_archive"][0]
            p(f'The span profile scores {q["exact_labeled_spans"]["f1"]:.5f} exact labeled span F1 beside {old["exact_labeled_spans"]["f1"]:.5f} for the strongest aligned author projection. The paired 95% gain ranges from {100*lo:+.2f} to {100*hi:+.2f} points. Evaluation by sentence index and character offset agreed for all {archive["cases"]} cases.')
        raw_check=result("sentence-phrase-sequence-raw")
        if raw_check and raw_check["sequence_freeze_sha256"]==sequenced["freeze_sha256"]:
            h(3,"Raw judgment transfer with automatically proposed sentence cuts")
            p(f'The raw experiment uses {raw_check["predicted_sentence_units"]:,} units from the automatic sentence splitter on the same {raw_check["cases"]} official cases. Each method predicts labeled character spans. The earlier segment-exact model is included as a raw-input reference.')
            scrub=result("sentence-phrase-raw-input-qa")
            if scrub and scrub["exact_emission_parity_after_replacing_gold_metadata"]:
                p(f'The scoring-input audit replayed all {scrub["units"]:,} raw units after replacing gold metadata with a placeholder. The model emissions matched exactly.')
            rows=[]
            for name,x in raw_check["profiles"].items():
                q=x["scores"]
                ci=x["paired_to_phrase_baseline"][0]
                rows.append((name,f'{q["exact_labeled_spans"]["f1"]:.5f}',f'{q["exact_role_boundaries"]["f1"]:.5f}',
                             f'{100*q["token_accuracy"]:.3f}%'," to ".join(f'{100*x:+.3f}' for x in ci)+" points"))
            previous=result("segment-weight-qa")
            if previous and previous["cases"]==raw_check["cases"]:
                b=previous["scores"]
                rows.append(("Earlier segment-exact raw profile",f'{b["exact_labeled_spans"]["f1"]:.5f}',
                             f'{b["exact_role_boundaries"]["f1"]:.5f}',f'{100*b["token_accuracy"]:.3f}%',""))
            table(("Method","Exact labeled span F1","Role boundary F1","Token accuracy","Paired span F1 change from phrase baseline"),rows)
        checked=result("sentence-phrase-sequence-cli-qa")
        if checked and checked["sequence_freeze_sha256"]==sequenced["freeze_sha256"]:
            p(f'The CPU command matched the supplied-sentence benchmark on a {checked["supplied"]["sentences"]}-sentence judgment. The raw-text command passed a character-offset and metric replay on a {checked["raw"]["units"]}-unit judgment.')
        links(("LegalSeg and its author benchmarks","https://aclanthology.org/2025.findings-naacl.63/"),
              ("Semi-Markov conditional random fields","https://proceedings.neurips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html"),
              ("Sequence modeling for legal judgments","https://www.ltg.ed.ac.uk/np/publications/ltg/papers/Hachey2005Sequence.pdf"))
    repair=result("sentence-pairwise-repair")
    repair_freeze=result("sentence-pairwise-repair-frozen")
    consensus=result("sentence-pairwise-consensus")
    if repair and repair_freeze and consensus and all("test" in q["splits"] for q in (repair,consensus)):
        h(2,"Writing a separate decision boundary for each role pair")
        n=sum(repair_freeze["role_counts"])
        p(f'The encoders and sequence decoder stay fixed. {len(repair_freeze["pairs"])} role pairs each receive a closed-form readout fitted on the relevant subset of {n:,} labeled sentences from {len(repair_freeze["fit_cases"])} training judgments. The challenger method compares the incumbent directly with supported alternative roles. The consensus method projects the pair margins onto a consistent set of seven scores. The 72 development judgments choose the rule for each incumbent.')
        rows=[]
        for method,record in (("Direct challenger",repair),("Reconciled pair scores",consensus)):
            for cohort,trial in record["splits"].items():
                for name,outcome in trial["arms"].items():
                    before,after=outcome["baseline"],outcome["labels"]
                    ci=outcome["paired_labels"]["macro_f1"]
                    rows.append((cohort,trial["cases"],method,name,f'{before["macro_f1"]:.5f}',f'{after["macro_f1"]:.5f}',
                                 f'{100*before["accuracy"]:.2f}%',f'{100*after["accuracy"]:.2f}%',
                                 " to ".join(f'{100*v:+.3f}' for v in ci),f'{outcome["changed"]:,}'))
        table(("Cohort","Judgments","Edit","Incumbent","Base macro-F1","Edited macro-F1","Base accuracy","Edited accuracy",
               "Paired macro-F1 change in points","Changed labels"),rows)
        for cohort in ("holdout","test"):
            outcome=repair["splits"][cohort]["arms"]["BERT hybrid"]
            count,one,two=outcome["baseline_confusions_ranked"][0]
            errors=outcome["baseline"]["rows"]-outcome["baseline"]["correct"]
            p(f'The leading {cohort} confusion was {one} versus {two}. It accounted for {count:,} of {errors:,} incumbent errors ({100*count/errors:.1f}%).')
        paired_best=repair["splits"]["test"]["arms"]["sequence only"]
        p(f'The strongest macro-F1 edit changes exact labeled span F1 from {paired_best["baseline_spans"]["exact_labeled_span_f1"]:.5f} to {paired_best["spans"]["exact_labeled_span_f1"]:.5f}. An application optimized for exact spans should use the separate span profile.')
        before=paired_best["baseline"]["per_role_f1"]
        after=paired_best["labels"]["per_role_f1"]
        table(("Role","Base F1","Edited F1","Change in points"),[
            (role.replace("_"," "),f'{before[role]:.5f}',f'{score:.5f}',f'{100*(score-before[role]):+.2f}')
            for role,score in after.items()])
        cli=result("sentence-pairwise-cli-qa")
        if cli:
            p(f'The direct and consensus CPU commands independently matched every benchmark label and span on a {cli["sentences"]}-sentence judgment.')
        if pair_replay:
            p(f'The clipped-passage replay scores {pair_replay["cases"]} passages across {pair_replay["candidate_units"]} units. All methods use the same passage text, candidate unit offsets and gold character spans.')
            table(("Source","Exact span F1","Token accuracy"),[(key,
                  f'{arm["scores"]["exact_labeled_spans"]["f1"]:.5f}',pct(arm["scores"]["token_accuracy"]))
                  for key,arm in pair_replay["profiles"].items()])
        links(("Hastie and Tibshirani on pairwise classification","https://proceedings.neurips.cc/paper_files/paper/1997/file/70feb62b69f16e0238f741fab228fec2-Paper.pdf"))
    joint_pairs=result("sentence-joint-pairwise")
    joint_pair_plan=result("sentence-joint-pairwise-frozen")
    if joint_pairs and joint_pair_plan and "test" in joint_pairs["splits"]:
        h(2,"Adding role-pair evidence before sequence decoding")
        p(f'This experiment writes new score and transition coefficients around the frozen 21 pair readouts. Development uses {len(joint_pair_plan["selection_cases"])} judgments and selects separate objectives for sentence macro-F1 and exact span F1. The control retunes the decoder’s transition weights on the same judgments. The direct, consensus and within-run pooling alternatives use identical encoders and input text.')
        rows=[]
        for group,record in joint_pairs["splits"].items():
            for name,value in record["profiles"].items():
                lo,hi=value["paired_spans"]["exact_labeled_span_f1"]
                rows.append((group,record["cases"],name,f'{value["labels"]["macro_f1"]:.5f}',
                             f'{100*value["labels"]["accuracy"]:.2f}%',f'{value["spans"]["exact_labeled_span_f1"]:.5f}',
                             f'{100*lo:+.2f} to {100*hi:+.2f}'))
        table(("Cohort","Judgments","Profile","Sentence macro-F1","Sentence accuracy","Exact span F1",
               "Paired exact span F1 change from original in points"),rows)
        check=joint_pairs["splits"]["test"]["profiles"]["spans"]
        lo,hi=check["paired_spans_to_no_pair"]["exact_labeled_span_f1"]
        p(f'The span profile’s official exact-span change versus the tuned control is {100*lo:+.2f} to {100*hi:+.2f} points at 95% confidence. The difference is uncertain on the official test.')
        if joint_pair_archive:
            lo,hi=joint_pair_archive["paired_change_from_public"][0]
            p(f'On identical sentence cuts, the paired 95% exact span gain over the strongest public author projection is {100*lo:+.2f} to {100*hi:+.2f} points. The verification checked all {joint_pair_archive["sentences"]:,} sentences and confirmed agreement between sentence-index and character-offset scoring.')
        for key,description in (("test","raw full judgments"),("cloud","matched Gemini passages")):
            raw_joint_pair=result("sentence-joint-pairwise-raw-"+key)
            if raw_joint_pair:
                h(3,f'Transfer to {description}')
                p(f'This replay covers {raw_joint_pair["judgments"]} judgments and {raw_joint_pair["units"]:,} units. The weights and score settings were frozen on supplied sentences before this evaluation.')
                table(("Profile","Exact span F1","Token accuracy"),[
                    (k,f'{v["scores"]["exact_labeled_spans"]["f1"]:.5f}',f'{100*v["scores"]["token_accuracy"]:.2f}%')
                    for k,v in raw_joint_pair["profiles"].items()])
        joint_cli=result("sentence-joint-pairwise-cli-qa")
        if joint_cli:
            p(f'The CPU command reproduced the labels and spans for both profiles on a {joint_cli["sentences"]}-sentence judgment and matched benchmark scoring on a raw passage.')
        links(("Lafferty, McCallum and Pereira on joint sequence labeling","https://www.cis.upenn.edu/~pereira/papers/crf.pdf"))
    referral_coverage=result("pairwise-referral-coverage")
    referral_policy=result("pairwise-referral-coverage-frozen")
    if referral_coverage and referral_policy and "test" in referral_coverage["splits"]:
        h(2,"How much can sentence referrals improve the classifier?")
        p("This audit gives each router the true label at every sentence it selects. The resulting score is an upper bound for any model restricted to those positions. Every other sentence keeps the frozen pairwise classifier’s label.")
        p(f'Development chose the {referral_policy["chosen"]} ranking. At matched cost, it uses the same number of referrals as the existing router within each judgment.')
        table(("Cohort","Policy","Sentences referred","Errors reachable","Error coverage","Oracle macro-F1"),[
            (split,name,f'{q["referred"]:,}',f'{q["errors_reachable"]:,}',
             f'{100*q["errors_reachable"]/q["total_errors"]:.2f}%',f'{q["ceiling"]["macro_f1"]:.5f}')
            for split,trial in referral_coverage["splits"].items() for name,q in trial["coverage"].items()])
        p("The 20% referral setting is the first tested budget to clear 0.770 in the official-test oracle experiment. The existing router and the new ranking at the same budget both remain below that score even with perfect corrections.")
    mixedbread_trial=result("mixedbread-referral")
    mixedbread_plan=result("mixedbread-referral-frozen")
    if mixedbread_trial and mixedbread_plan and "confirm" in mixedbread_trial["splits"]:
        h(2,"A 24.1M-parameter encoder for referred sentences")
        p(f'The Mixedbread study uses the same referred positions as InLegalBERT. Up to two preceding units share a {mixedbread_plan["window"]}-token window with the target. A closed-form head reads the mean of the target’s contextual token vectors. Neither encoder receives gradient updates.')
        table(("Cohort","Judgments","Referred sentences","Structure only macro-F1","Mixedbread macro-F1","InLegalBERT macro-F1"),[
            (cohort,q["cases"],f'{q["referrals"]:,}',f'{q["heads"]["structure only"]["labels"]["macro_f1"]:.5f}',
             f'{q["heads"]["Mixedbread target pooled"]["labels"]["macro_f1"]:.5f}',
             f'{q["InLegalBERT referral"]["macro_f1"]:.5f}') for cohort,q in mixedbread_trial["splits"].items()])
        if "test" not in mixedbread_trial["splits"]:
            p("The table reflects the completed cohorts so far. Official test scoring is pending.")
        mixed_context=result("mixedbread-referral-context")
        if mixed_context and "test" in mixed_context["cohorts"]:
            q=mixed_context["cohorts"]["test"]
            p(f'Among the {q["units"]:,} official target windows, the median contained {q["median_packed_tokens"]:g} tokens, {q["full_windows"]:,} reached the 96-token cap and {q["target_truncated"]:,} truncated the target sentence itself.')
        if mixedbread_overlay and "confirm" in mixedbread_overlay["splits"]:
            p("A separate experiment adds either encoder on top of the stronger direct pairwise classifier. The combined head uses both frozen embeddings. It applies the same development-selected override rule on confirmation and later cohorts.")
            table(("Cohort","Edit","Parent macro-F1","Edited macro-F1","Edited accuracy","Paired macro-F1 change in points"),[
                (cohort,name,f'{q["baseline"]["macro_f1"]:.5f}',f'{v["labels"]["macro_f1"]:.5f}',
                 f'{100*v["labels"]["accuracy"]:.2f}%',' to '.join(f'{100*c:+.2f}' for c in v["paired_baseline"]["macro_f1"]))
                for cohort,q in mixedbread_overlay["splits"].items() for name,v in q["arms"].items()])
            rows=[]
            for cohort,q in mixedbread_overlay["splits"].items():
                limits=q.get("paired original Mixedbread vs BERT")
                if limits:
                    rows.append((cohort,' to '.join(f'{100*x:+.2f}' for x in limits["macro_f1"]),
                                 ' to '.join(f'{100*x:+.2f}' for x in limits["accuracy"])))
            if rows:
                table(("Cohort","Mixedbread minus InLegalBERT macro-F1, 95% interval in points",
                       "Mixedbread minus InLegalBERT accuracy, 95% interval in points"),rows)
        links(("Pinned Mixedbread encoder","https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/tree/e6ac24e5d6efb8782b59de1647b3ececb4ece94e"),
              ("Ting and Witten on combining classifier confidence","https://arxiv.org/abs/1105.5466"))
    kernel=result("sentence-kernel-weight-write")
    kernel_plan=result("sentence-kernel-weight-write-frozen")
    if kernel and kernel_plan and "test" in kernel["splits"]:
        h(2,"Kernel readout over the frozen judgment features")
        p(f'The write approximates a mixture of Gaussian kernels with random Fourier features. A closed-form ridge solve uses {kernel_plan["training_rows"]:,} labeled sentences. The selection checks whether the written readout should replace low-confidence decisions or should cover every sentence.')
        table(("Cohort","Judgments","Parent macro-F1","Edited macro-F1","Edited accuracy","Changed labels","Paired macro-F1 change in points"),[
            (s,q["cases"],f'{q["baseline"]["macro_f1"]:.5f}',f'{q["edited"]["macro_f1"]:.5f}',
             f'{100*q["edited"]["accuracy"]:.2f}%',f'{q["changed"]:,}',
             ' to '.join(f'{100*v:+.2f}' for v in q["paired"]["macro_f1"])) for s,q in kernel["splits"].items()])
        links(("Rahimi and Recht on random kernel features","https://papers.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html"))
    phase_readout=result("sentence-phase-readout")
    phase_bert=result("sentence-phase-readout-bert")
    if phase_readout and "test" in phase_readout["splits"]:
        h(2,"Different readouts for different parts of a judgment")
        p("The experiment writes five position-weighted readouts for the seven roles. One variant also conditions their weights on confidence. All readouts share the frozen encoder, and every sentence is eligible for correction. The global readout is the control.")
        table(("Cohort","Method","Direct model macro-F1","Edited macro-F1","Edited accuracy","Changed labels"),[
            (s,method,f'{q["baseline"]["macro_f1"]:.5f}',f'{v["labels"]["macro_f1"]:.5f}',
             f'{100*v["labels"]["accuracy"]:.2f}%',f'{v["changed"]:,}')
            for s,q in phase_readout["splits"].items() for method,v in q["arms"].items()])
        if phase_bert and "test" in phase_bert["splits"]:
            q=phase_bert["splits"]["test"]
            lo,hi=q["paired_base"]["macro_f1"]
            p(f'The position edit chosen on development changes the official InLegalBERT overlay from {q["base"]["macro_f1"]:.5f} to {q["candidate"]["macro_f1"]:.5f} macro-F1. The paired difference is {100*lo:+.3f} to {100*hi:+.3f} macro-F1 points at 95% confidence.')
        links(("Jacobs and colleagues on mixtures of local experts","https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf"))
    corpus_lexical=result("sentence-full-corpus-lexical")
    corpus_memory=result("sentence-full-corpus-lexical-memory-frozen")
    corpus_bert=result("sentence-full-corpus-lexical-bert")
    if corpus_lexical and corpus_memory and "test" in corpus_lexical["splits"]:
        h(2,"A lexical memory written from the larger training corpus")
        p(f'The sparse memory tallies word unigrams, bigrams and trigrams in {corpus_memory["training_rows"]:,} sentences from {len(corpus_memory["training_cases"]):,} training judgments. The {len(corpus_memory["calibration_cases_excluded"])} calibration cases are excluded from that count. The fitted head reads lexical scores from the target and surrounding sentences, and it can alter any sentence.')
        cells=4*7*corpus_memory["feature_slots"]
        p(f'The written lexical tables contain {cells:,} float32 values across one global table and three position-specific tables. The encoder stays frozen.')
        table(("Cohort","Features","Direct model macro-F1","Edited macro-F1","Edited accuracy",
               "Paired macro-F1 change in points","Changed labels"),[
            (s,method,f'{q["baseline"]["macro_f1"]:.5f}',f'{v["labels"]["macro_f1"]:.5f}',
             f'{100*v["labels"]["accuracy"]:.2f}%',' to '.join(f'{100*y:+.2f}' for y in v["paired"]["macro_f1"]),f'{v["changed"]:,}')
            for s,q in corpus_lexical["splits"].items() for method,v in q["arms"].items()])
        if corpus_bert and "test" in corpus_bert["splits"]:
            table(("Cohort","InLegalBERT overlay macro-F1","With lexical memory","Edited accuracy","Changed labels"),[
                (s,f'{q["base"]["macro_f1"]:.5f}',f'{q["candidate"]["macro_f1"]:.5f}',
                 f'{100*q["candidate"]["accuracy"]:.2f}%',f'{q["changed"]:,}') for s,q in corpus_bert["splits"].items()])
        links(("Wang and Manning on word sequences and log-count ratios","https://aclanthology.org/P12-2018/"))
    role_trigrams=result("sentence-trigram-transition")
    trigram_plan=result("sentence-trigram-transition-frozen")
    if role_trigrams and trigram_plan and "test" in role_trigrams["splits"]:
        h(2,"Decoding over triples of rhetorical roles")
        p(f'The second-order decoder writes a smoothed table from {trigram_plan["labeled_triples"]:,} adjacent role triples. It tests global and position-conditioned tables, with separate development objectives for sentence macro-F1 and exact span F1. The implementation matched exhaustive search for paths up to five sentences and reproduced the first-order predictions at zero triple weight on every cohort.')
        table(("Cohort","Profile","Sentence macro-F1","Baseline exact span F1","Edited exact span F1",
               "Paired exact span F1 gain in points","Changed labels"),[
            (s,method,f'{v["labels"]["macro_f1"]:.5f}',f'{q["baseline_spans"]["exact_labeled_span_f1"]:.5f}',
             f'{v["spans"]["exact_labeled_span_f1"]:.5f}',
             ' to '.join(f'{100*y:+.2f}' for y in v["paired_spans"]["exact_labeled_span_f1"]),f'{v["changed"]:,}')
            for s,q in role_trigrams["splits"].items() for method,v in q["profiles"].items()])
        links(("Ye and colleagues on higher-order sequence features","https://proceedings.neurips.cc/paper/2009/hash/94f6d7e04a4d452035300f18b984988c-Abstract.html"))
    minilm_whitening=result("sentence-minilm-whitened")
    minilm_whitening_plan=result("sentence-minilm-whitened-frozen")
    if minilm_whitening and minilm_whitening_plan and "test" in minilm_whitening["splits"]:
        h(2,"Reusing MiniLM across a wider referral budget")
        p(f'The frozen 22.7M-parameter MiniLM encoder contributes 384-dimensional vectors for roughly 20% of each judgment’s sentences. The experiment writes direct linear readouts from {minilm_whitening_plan["fit_rows"]:,} referred training sentences. It compares document cues alone, MiniLM, a 96-dimensional whitened projection, and a projection with semantic context from nearby referred positions.')
        p("The encoder run caps input at 96 tokens including special tokens and reads each referred sentence separately. The legacy cache has no per-block text fingerprints. A replay of the original router verified the cached sentence indices against every judgment. The benchmark reuses the archived embeddings from their original quantized batches.")
        table(("Cohort","Judgments","Referred sentences","Readout","Macro-F1","Accuracy","Exact span F1","Edited labels","Paired macro-F1 gain in points"),[
            (s,q["cases"],f'{q["referrals"]:,}',name,f'{v["scores"]["macro_f1"]:.5f}',f'{100*v["scores"]["accuracy"]:.2f}%',
             f'{v["spans"]["exact_labeled_span_f1"]:.5f}',f'{v["changed"]:,}',
             ' to '.join(f'{100*y:+.2f}' for y in v["paired"]["macro_f1"])) for s,q in minilm_whitening["splits"].items()
            for name,v in q["arms"].items()])
        choice=minilm_whitening["development_selected_arm"]
        q=minilm_whitening["splits"]["test"]
        if all(not v["settings"] for v in minilm_whitening_plan["arms"].values()):
            p("Development kept the existing classifier for all four readouts. A separate diagnostic forces nonzero edits and freezes each readout’s settings on development before evaluating the other cohorts.")
        else:
            p(f'Development selected the {choice} readout.')
        p(f'The official-test router makes {q["errors_reachable"]:,} errors reachable and has a perfect-correction ceiling of {q["oracle_macro_f1"]:.5f} macro-F1.')
        if q.get("composed_with_bert"):
            table(("Cohort","InLegalBERT review macro-F1","With the chosen MiniLM study head","Edited labels","Paired macro-F1 gain in points"),[
                (s,f'{v["base"]["macro_f1"]:.5f}',f'{v["scores"]["macro_f1"]:.5f}',f'{v["changed"]:,}',
                 ' to '.join(f'{100*y:+.2f}' for y in v["paired"]["macro_f1"])) for s,d in minilm_whitening["splits"].items()
                for v in ([d["composed_with_bert"]] if "composed_with_bert" in d else [])])
        active_minilm=result("sentence-minilm-whitened-active")
        active_mini_plan=result("sentence-minilm-whitened-active-frozen")
        if active_minilm and active_mini_plan and "test" in active_minilm["splits"]:
            h(3,"Diagnostic with active corrections")
            p(f'Each readout had to change at least {active_mini_plan["minimum_development_revisions"]} development labels. The diagnostic includes candidates selected below the baseline, which allows comparison of actual revisions on the held-out judgments.')
            table(("Cohort","Readout","Base macro-F1","Edited macro-F1","Edited accuracy","Changed labels","Paired macro-F1 gain in points"),[
                (s,n,f'{q["base"]["macro_f1"]:.5f}',f'{r["scores"]["macro_f1"]:.5f}',f'{100*r["scores"]["accuracy"]:.2f}%',
                 f'{r["changed"]:,}',' to '.join(f'{100*v:+.2f}' for v in r["paired"]["macro_f1"]))
                for s,q in active_minilm["splits"].items() for n,r in q["arms"].items()])
            p(f'The active diagnostic selected the {active_minilm["development_selected_arm"]} head on development.')
            table(("Cohort","InLegalBERT review macro-F1","With the active diagnostic head","Changed labels"),[
                (s,f'{v["base"]["macro_f1"]:.5f}',f'{v["scores"]["macro_f1"]:.5f}',f'{v["changed"]:,}')
                for s,q in active_minilm["splits"].items() for v in ([q["composed_with_bert"]] if "composed_with_bert" in q else [])])
        links(("Pinned all-MiniLM-L6-v2 model","https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
              ("Su and colleagues on whitening sentence representations","https://arxiv.org/abs/2103.15316"))
    referral_utility=result("sentence-referral-utility")
    utility_plan=result("sentence-referral-utility-frozen")
    if referral_utility and utility_plan and "test" in referral_utility["splits"]:
        h(2,"Ranking sentences by the value of a correction")
        p(f'A closed-form readout estimates error probability and the approximate macro-F1 benefit of correcting a label. It uses {utility_plan["fit_rows"]:,} sentences from {len(utility_plan["train_cases"])} training judgments outside the pairwise editor’s fit set. Development chooses the ranking at a 20% review budget. Both rankings refer the same number of sentences within each judgment.')
        table(("Cohort","Budget","Referred","Confidence errors reached","Utility errors reached","Confidence ceiling","Utility ceiling"),[
            (s,b,f'{a["referred"]:,}',f'{a["errors_reachable"]:,}',f'{c["errors_reachable"]:,}',
             f'{a["ceiling"]["macro_f1"]:.5f}',f'{c["ceiling"]["macro_f1"]:.5f}')
            for s,q in referral_utility["splits"].items() for b,a in q["baseline"].items() for c in [q["edited"][b]]])
        p("The ceilings assume every selected label is corrected perfectly. The evaluation measures routing headroom alongside the frozen classifier.")
        baseline=referral_utility["splits"]["test"]["baseline"]["20%"]
        utility=referral_utility["splits"]["test"]["edited"]["20%"]
        table(("Gold role","Errors reached by confidence","Errors reached by utility"),[
            (r,a,b) for r,a,b in zip(("none","facts","issue","petitioner argument","respondent argument","reasoning","decision"),
                                     baseline["gold_errors_reachable"],utility["gold_errors_reachable"])])
        p(f'The utility router reaches fewer errors overall at the 20% budget. Its perfect-correction accuracy is {100*utility["ceiling"]["accuracy"]:.2f}% beside {100*baseline["ceiling"]["accuracy"]:.2f}% for confidence routing. The change favours macro-F1, which gives each role equal weight.')
        links(("Mozannar and Sontag on learning where to defer","https://proceedings.mlr.press/v119/mozannar20b.html"))
    replay=result("sentence-referral-actual-replay-archive")
    if replay:
        h(2,"Testing the routing budget with actual reviewer labels")
        p(f'This replay uses the same {replay["sentences"]:,} sentences from {replay["cases"]} official test judgments. The two rankings use identical review budgets within each judgment. Every referred label comes from the same pre-existing reviewer file within each comparison. Each author file is aligned to the original gold sequence and checked against its published archive score.')
        p("The previous-predicted-role reviewer uses the authors’ saved predictions. The released checkpoint inventory has no separately named weights for that variant. This replay measures the composition on the official test and cannot run on new judgments from the prediction file alone.")
        table(("Author reviewer","Budget","Ranking","Macro-F1","Accuracy","Exact span F1","Corrected labels","Broken labels"),[
            (name,b,route,f'{v["labels"]["macro_f1"]:.5f}',f'{100*v["labels"]["accuracy"]:.2f}%',
             f'{v["spans"]["exact_labeled_span_f1"]:.5f}',f'{v["fixed"]:,}',f'{v["broken"]:,}')
            for name,q in replay["reviewers"].items() for b,trial in q["rates"].items() for route,v in trial["routes"].items()])
        compare=replay["reviewers"]["previous_predicted_role"]["rates"]["20%"]
        p(f'The best archived-reviewer replay reaches {compare["routes"]["utility"]["labels"]["macro_f1"]:.5f} macro-F1 with the previous-predicted-role reviewer. Substitution lowers exact span F1 compared with the local span-oriented classifier.')
        lo,hi=compare["paired_utility_minus_confidence"]["macro_f1"]
        p(f'The official 20% macro-F1 difference for utility versus confidence routing is {100*lo:+.2f} to {100*hi:+.2f} points at 95% confidence with the previous-predicted-role reviewer.')
        if "local_overlay_protected_rows" in replay:
            h(3,"Preserving the existing local InLegalBERT corrections")
            p(f'This fixed comparison protects {replay["local_overlay_protected_rows"]:,} accepted local BERT changes and asks the archived reviewer for the other selected labels within the same 20% budget.')
            table(("Author reviewer","Ranking","Macro-F1","Accuracy","Exact span F1"),[
                (name,route,f'{v["labels"]["macro_f1"]:.5f}',f'{100*v["labels"]["accuracy"]:.2f}%',
                 f'{v["spans"]["exact_labeled_span_f1"]:.5f}') for name,q in replay["reviewers"].items()
                for route,v in q["protect_local_overlay_at_20_percent"]["routes"].items()])
        links(("Authors’ InLegalBERT predictions","https://huggingface.co/datasets/L-NLProc/LegalSeg_InLegalBERT_Predictions"),
              ("Authors’ released checkpoint inventory","https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53"))
    mini_replay=result("sentence-referral-actual-replay-minilm")
    if mini_replay:
        h(3,"Matched budgets within the archived MiniLM coverage")
        p("Both rankings draw from the identical cached MiniLM positions within each judgment. The diagnostic uses the same frozen MiniLM head and applies its existing override rule before comparison.")
        table(("Cohort","Budget","Same budget per ranking","Confidence macro-F1","Utility macro-F1","Paired difference in points"),[
            (s,b,f'{routes["confidence"]["referred"]:,}',f'{routes["confidence"]["labels"]["macro_f1"]:.5f}',
             f'{routes["utility"]["labels"]["macro_f1"]:.5f}',
             ' to '.join(f'{100*x:+.2f}' for x in v["paired_utility_minus_confidence"]["macro_f1"]))
            for s,q in mini_replay["splits"].items() for b,v in q["rates"].items() for routes in [v["routes"]]])
    if wide_live:
        h(3,"Wider review with the pinned local InLegalBERT checkpoint")
        p("Each route receives a 20% review budget per judgment. The benchmark encodes the union of both routes with the same frozen reviewer and acceptance rule. The table lists completed cohorts.")
        table(("Cohort","Judgments","Referred per route","Base macro-F1","Confidence macro-F1","Utility macro-F1","Confidence accuracy","Paired utility minus confidence in points"),[
            (s,q["cases"],f'{r["confidence"]["referred"]:,}',f'{q["baseline"]["macro_f1"]:.5f}',
             f'{r["confidence"]["labels"]["macro_f1"]:.5f}',f'{r["utility"]["labels"]["macro_f1"]:.5f}',
             f'{100*r["confidence"]["labels"]["accuracy"]:.2f}%',
             ' to '.join(f'{100*v:+.2f}' for v in q["comparison"]["paired_utility_minus_confidence"]["macro_f1"]))
            for s,q in wide_live["splits"].items() for r in [q["comparison"]["routes"]]])
        pairs=((0,5,"None and reasoning"),(1,2,"Facts and issue"),(3,4,"Petitioner and respondent arguments"))
        def error_count(x,a,b):
            return x["confusion"][a][b]+x["confusion"][b][a]
        table(("Cohort","Role pair","Parent errors","Confidence review errors","Utility review errors"),[
            (split,label,f'{error_count(q["baseline"],a,b):,}',f'{error_count(arms["confidence"]["labels"],a,b):,}',
             f'{error_count(arms["utility"]["labels"],a,b):,}')
            for split,q in wide_live["splits"].items() for arms in [q["comparison"]["routes"]]
            for a,b,label in pairs])
        p("Pair counts include errors in both directions. A change in the total can include sentences moved to or from other incorrect roles.")
        if wide_official:
            conf=wide_official["comparison"]["routes"]["confidence"]
            lo,hi=conf["paired_base"]["macro_f1"]
            p(f'On the official test, confidence review corrected {conf["fixed"]:,} labels and introduced {conf["broken"]:,} errors compared with the sequence classifier. The paired 95% macro-F1 gain is {100*lo:+.3f} to {100*hi:+.3f} points. This comparison uses the sequence classifier as its reference.')
        if wide_cli:
            checks=wide_cli["cases"]
            p(f'The CPU command reproduced every supplied-sentence label, reviewed position and combined span on {len(checks)} confirmation judgments with accepted reviewer changes. Those judgments cover {sum(q["sentences"] for q in checks):,} sentences and {sum(q["reviewed"] for q in checks):,} reviewer windows. An additional raw-text check verified ordered character spans.')
    multiclass=result("sentence-wide-bert-multiclass")
    multiclass_plan=result("sentence-wide-bert-multiclass-frozen")
    if multiclass and multiclass_plan:
        h(3,"Writing a seven-role correction for the contextual reviewer")
        p(f'The write uses {multiclass_plan["fit_rows"]:,} original training referrals from {len(multiclass_plan["fit_cases"])} judgments. BERT outputs are cross-fitted in {multiclass_plan["folds"]} folds by judgment. The candidate matrices combine both models’ role scores with local sentence features and can score all seven roles. The same confidence routing budget supplies the reference and the candidate.')
        rule=multiclass_plan["rule"]
        if rule:
            _,(strength,threshold,protect,alternatives)=rule
            p(f'Development selected write strength {strength:g}, a score threshold of {threshold:g}, and {"preservation" if protect else "reconsideration"} of earlier reviewer overrides. The selected policy could propose {"any of the seven roles" if alternatives=="any role" else "the document model’s label, the reviewer’s top role or the incumbent label"}.')
        else:
            p("The unchanged wide review scored best under the development criteria.")
        table(("Cohort","Judgments","Reviewed","Parent macro-F1","Edited macro-F1","Errors corrected","Errors introduced","Paired macro-F1 change in points"),[
            (s,q["cases"],f'{q["reviewed"]:,}',f'{q["parent"]["macro_f1"]:.5f}',f'{q["candidate"]["labels"]["macro_f1"]:.5f}',
             q["candidate"]["fixed"],q["candidate"]["broken"],
             ' to '.join(f'{100*x:+.2f}' for x in q["paired"]["macro_f1"])) for s,q in multiclass["splits"].items()])
        links(("van der Laan et al. on combining predictions using cross-validation","https://pubmed.ncbi.nlm.nih.gov/17910531/"))
    wide_template=result("sentence-wide-bert-template")
    if wide_template:
        h(3,"Combining wider review with exact sentence memory")
        p("The original exact-template policy supplies extra corrections at both confidence budgets. Accepted BERT corrections take precedence. This composition keeps both frozen policies.")
        table(("Cohort","Budget","Judgments","Review windows","Review-only macro-F1","With memory","Corrected","Introduced","Paired macro-F1 change in points"),[
            (s,f'{float(rate):.0%}',q["cases"],f'{t["reviewed"]:,}',f'{t["base"]["macro_f1"]:.5f}',
             f'{t["composed"]["labels"]["macro_f1"]:.5f}',t["composed"]["fixed"],t["composed"]["broken"],
             ' to '.join(f'{100*x:+.2f}' for x in t["paired"]["macro_f1"]))
            for s,q in wide_template["splits"].items() for rate,t in q["rates"].items()])
        official_template=wide_template["splits"].get("test",{})
        if "previous_end_to_end" in official_template:
            p(f'The prior runnable InLegalBERT and exact-template combination reproduced its {official_template["previous_end_to_end"]["macro_f1"]:.5f} official macro-F1 score.')
            table(("Official review budget","Macro-F1 of wider combination","Paired change from prior runnable model in points"),[
                (f'{float(rate):.0%}',f'{q["composed"]["labels"]["macro_f1"]:.5f}',
                 ' to '.join(f'{100*x:+.2f}' for x in q["paired_to_previous_end_to_end"]["macro_f1"]))
                for rate,q in official_template["rates"].items()])
    prevalence=result("sentence-role-prevalence")
    prevalence_plan=result("sentence-role-prevalence-frozen")
    if prevalence and prevalence_plan:
        h(3,"Writing judgment-level role frequency corrections")
        p(f'The edit uses {prevalence_plan["fit_rows"]:,} sentences from {len(prevalence_plan["fit_cases"]):,} LegalSeg training judgments. It summarizes predicted roles over the whole judgment and in five relative-position blocks. A closed-form matrix estimates each block’s role frequency errors. The edit adds a bounded bias to the frozen per-sentence scores and uses zero gradient steps.')
        share,strength,threshold,preserve=prevalence_plan["choice"]
        if strength:
            p(f'Development selected a blend of {share:g} whole-judgment weight and {1-share:g} block weight, with bias strength {strength:g} and a minimum score gain of {threshold:g}. Issue protection was {"enabled" if preserve else "disabled"}.')
        else:
            p("Development selected the zero-strength control under the macro-F1 and accuracy criteria.")
        table(("Cohort","Judgments","Parent macro-F1","Edited macro-F1","Parent accuracy","Edited accuracy","Labels changed","Paired macro-F1 change in points"),[
            (s,q["cases"],f'{b["labels"]["macro_f1"]:.5f}',f'{r["labels"]["macro_f1"]:.5f}',
             pct(b["labels"]["accuracy"]),pct(r["labels"]["accuracy"]),f'{r["changed"]:,}',
             ' to '.join(f'{100*x:+.2f}' for x in q["paired_to_base"]["macro_f1"]))
            for s,q in prevalence["splits"].items() for b,r in [(q["baseline"],q["candidate"])]])
        p("Global prototype modulation and role-boundary modeling motivate making document-level information explicit. The tested edit estimates frequency corrections from the current model’s own outputs.")
        links(("Belfathi et al. on local context and global prototypes","https://aclanthology.org/2026.eacl-long.137/"),
              ("Malik et al. on role-boundary modeling","https://aclanthology.org/2022.nllp-1.13/"))
    budget=result("sentence-wide-bert-budget")
    headroom=result("sentence-role-pair-headroom")
    if headroom:
        h(3,"Which role confusions limit the official score?")
        p(f'The {headroom["rows"]:,}-sentence test has {headroom["total_errors"]:,} errors in the runnable model that scores {headroom["macro_f1"]:.5f} macro-F1. The table uses the human test labels to correct every error in a role pair and then recalculates the seven-role metric. These are ceilings for perfect corrections within the specified pairs.')
        table(("Confused roles","Errors in both directions","Macro-F1 if this pair is fixed","Macro-F1 if this and higher-ranked pairs are fixed"),[
            (" and ".join(r["roles"]).replace("_"," "),f'{r["all_pair_errors"]:,}',f'{r["macro_f1_if_pair_fixed"]:.5f}',
             f'{r["cumulative_macro_f1"]:.5f}') for r in headroom["ranked"][:7]])
        p(f'Perfect correction of the two highest-impact pairs reaches {headroom["ranked"][1]["cumulative_macro_f1"]:.5f}, just below the published {headroom["published_target"]:.3f} target. Errors between the two parties’ arguments have the next-largest isolated impact.')
    fuzzy=result("sentence-fuzzy-pair-memory")
    fuzzy_plan=result("sentence-fuzzy-pair-memory-frozen")
    if fuzzy and fuzzy_plan:
        h(3,"Writing a near-match memory for none and reasoning")
        memory=fuzzy_plan["memory"]
        p(f'This sparse edit stores {memory["shape"][0]:,} official training examples with {memory["nonzero_coefficients"]:,} nonzero coefficients. The official error audit motivated the choice of the none and reasoning pair. The model tests lexical near matches when the existing scores plausibly support either role. Hashed character n-grams give a cosine score. A near-match vote requires distinct examples and distinct source judgments. The search excludes exact matches after masking numbers.')
        if fuzzy_plan["choice"]:
            similar,neighbors,agreement,margin=fuzzy_plan["choice"]
            p(f'Development chose minimum similarity {similar:g}, at least {neighbors} agreeing neighbors, vote agreement {agreement:g}, and a score margin of at most {margin:g}.')
        else:
            p("Development selected the baseline from the near-match alternatives.")
        table(("Cohort","Judgments","Queried sentences","Parent macro-F1","Edited macro-F1","Labels changed","Errors corrected","Errors introduced","Paired change in points"),[
            (s,q["cases"],f'{q["queried"]:,}',f'{q["baseline"]["macro_f1"]:.5f}',
             f'{q["edited"]["labels"]["macro_f1"]:.5f}',q["edited"]["changed"],q["edited"]["fixed"],q["edited"]["broken"],
             ' to '.join(f'{100*x:+.2f}' for x in q["paired"]["macro_f1"]))
            for s,q in fuzzy["splits"].items()])
        links(("Ok and Lee on sparse associative memories","https://www.alphaxiv.org/abs/2611.weight-space-programming-associative-memory"),
              ("SeqTNS on similarity in rhetorical role labeling","https://aclanthology.org/2025.findings-ijcnlp.50/"))
    if budget:
        h(3,"How much review does each ranking need?")
        p("The budget replay uses the exact FP32 features and frozen acceptance rule from the wider benchmark. It tries six nested budgets. On development, it chooses the smallest budget that retains at least 90% of that ranking’s 20% macro-F1 gain, with an accuracy decrease of at most 0.1 percentage points from the baseline. Those choices carry forward unchanged.")
        table(("Cohort","Ranking","Selected budget","Reviewed at selected budget","Reviewed at 20%","Base macro-F1","Selected macro-F1","20% macro-F1","Paired change from base in points","Paired change from 20% in points"),[
            (s,route,f'{100*rate:g}%',f'{selected["referred"]:,}',f'{curve["0.2"]["referred"]:,}',
             f'{q["baseline"]["macro_f1"]:.5f}',f'{selected["labels"]["macro_f1"]:.5f}',
             f'{curve["0.2"]["labels"]["macro_f1"]:.5f}',
             ' to '.join(f'{100*x:+.2f}' for x in selected["paired_to_base"]["macro_f1"]),
             ' to '.join(f'{100*x:+.2f}' for x in selected["paired_to_full"]["macro_f1"]))
            for s,q in budget["splits"].items() for route,curve in q["routes"].items()
            for rate in [budget["development_chosen_rates"][route]] if rate is not None
            for selected in [curve[str(rate)]]])
        table(("Cohort","Ranking","3%","5%","7.5%","10%","15%","20%"),[
            (s,route,*(f'{curve[str(rate)]["labels"]["macro_f1"]:.5f}' for rate in (.03,.05,.075,.10,.15,.20)))
            for s,q in budget["splits"].items() for route,curve in q["routes"].items()])
    quantized=result("sentence-bert-int8-review")
    if quantized:
        h(3,"Dynamic INT8 for the CPU reviewer")
        graph=quantized["graph"]
        p(f'The pooled InLegalBERT encoder uses per-channel unsigned INT8 weights for matrix multiplications, producing a {graph["quantized_bytes"]/1e6:.1f} MB ONNX graph from {graph["fp32_bytes"]/1e6:.1f} MB. The paired comparison uses the deployed referral positions and acceptance rule. CPU timings cover encoder windows after warm-up with {quantized["thread_count"]} threads. The pinned reference uses saved FP32 GPU pooled features.')
        table(("Cohort","Judgments","Windows","CPU encoder","Windows per second","Macro-F1","Accuracy","Labels different from pinned FP32","Paired macro-F1 change in points"),[
            (s,q["cases"],f'{q["referrals"]:,}',name,f'{m["windows_per_second"]:.2f}',
             f'{m["labels"]["macro_f1"]:.5f}',f'{100*m["labels"]["accuracy"]:.2f}%',
             m["label_disagreements"],' to '.join(f'{100*x:+.2f}' for x in m["paired_to_pinned"]["macro_f1"]))
            for s,q in quantized["splits"].items() for name,m in q["methods"].items()])
        precision=result("sentence-bert-int8-precision")
        if precision:
            p(f'A separate diagnostic compares weight types and ranges on the same {len(precision["positions"])} deterministic confirmation windows. It measures numerical error in the pooled features. The task score above uses the entire referral set.')
            table(("Weights and range","Graph size","Mean absolute feature change","Peak feature change"),[
                (name.replace("-"," "),f'{q["graph_size_bytes"]/1e6:.1f} MB',
                 f'{q["mean_abs_pooler_error"]:.4f}',f'{q["peak_pooler_error"]:.4f}')
                for name,q in precision["results"].items()])
        links(("ONNX Runtime quantization documentation","https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html"))
    template=result("sentence-template-memory")
    template_source=result("sentence-template-memory-source")
    if template and template_source and "test" in template["splits"]:
        h(2,"Retrieving recurring sentences from the training corpus")
        p(f'The write stores {sum(template_source["stored_keys"].values()):,} recurrent keys across exact sentences, masked numbers and adjacent sentence pairs. Each training judgment contributes at most one unambiguous vote per key. It excludes {len(template_source["excluded_exact_duplicate_judgments"])} training judgments whose full sentence sequence exactly matches a validation or test judgment after whitespace and case normalization.')
        table(("Cohort","Base macro-F1","With template memory","Edited accuracy","Corrected labels","Broken labels","Paired macro-F1 change in points"),[
            (s,f'{q["baseline"]["macro_f1"]:.5f}',f'{q["candidate"]["macro_f1"]:.5f}',f'{100*q["candidate"]["accuracy"]:.2f}%',
             q["fixed"],q["broken"], ' to '.join(f'{100*x:+.2f}' for x in q["paired"]["macro_f1"]))
            for s,q in template["splits"].items()])
        if template_overlay and "test" in template_overlay["splits"]:
            table(("Cohort","Existing local BERT review macro-F1","With template memory","Corrected labels","Broken labels"),[
                (s,f'{q["baseline"]["macro_f1"]:.5f}',f'{q["candidate"]["macro_f1"]:.5f}',q["fixed"],q["broken"])
                for s,q in template_overlay["splits"].items()])
        template_cli=result("sentence-template-memory-cli-qa")
        if template_cli:
            p(f'The CPU command reproduced every label and combined span on a {template_cli["sentences"]}-sentence confirmation judgment with an accepted template revision.')
            if template_cli.get("composed_bert_labels_and_spans_exact_parity"):
                p(f'The same CPU check verified the composition with {template_cli["bert_revisions"]} accepted InLegalBERT revisions and {template_cli["template_revisions_after_bert"]} additional template revisions.')
    expert_gate=result("sentence-bert-reviewer-gain")
    expert_gate_plan=result("sentence-bert-reviewer-gain-frozen")
    if expert_gate and expert_gate_plan and "test" in expert_gate["splits"]:
        h(2,"Estimating the benefit of a real InLegalBERT correction")
        p(f'The gate uses {expert_gate_plan["train_referrals"]:,} contextual reviewer outputs calculated with four-fold cross-fitting by judgment. It writes two outputs for expected accuracy and macro-F1 gain. Test inference uses the existing frozen reviewer and its exact referral positions.')
        table(("Cohort","Referrals","Preceding macro-F1","Edited macro-F1","Edited accuracy","Changed labels","Paired macro-F1 change in points"),[
            (s,f'{q["referred"]:,}',f'{q["baseline"]["macro_f1"]:.5f}',f'{q["scores"]["macro_f1"]:.5f}',f'{100*q["scores"]["accuracy"]:.2f}%',
             q["changed"], ' to '.join(f'{100*x:+.2f}' for x in q["paired"]["macro_f1"])) for s,q in expert_gate["splits"].items()])
    conditioned=result("sentence-conditioned-edges")
    conditioned_freeze=result("sentence-conditioned-edges-frozen")
    if conditioned and conditioned_freeze and "test" in conditioned["splits"]:
        h(2,"Conditioning sentence transitions on neighboring evidence")
        p(f'The encoder stays frozen. Smoothed counts from {len(conditioned_freeze["train_cases"])} training judgments write tables for role transitions. The four signals are adjacent sentence representation similarity, emission probability overlap, lexical support for staying in the same role, and relative position. Each of five bins can change the preference for keeping a role or the full set of 49 transition scores.')
        mode,source,strength=conditioned["choice"]
        if strength:
            feature=conditioned_freeze["features"][source] if source<len(conditioned_freeze["features"]) else "all features averaged"
            feature={"cosine":"representation similarity","agreement":"emission overlap",
                     "margin":"lexical support for a repeated role","position":"relative position"}.get(feature,feature)
            method="same-role preference" if mode=="stay" else "full conditional transition score"
            p(f'Validation selected the {method} using {feature} with weight {strength:g}.')
        else:
            p("Validation kept the existing sentence-joint weights.")
        table(("Cohort","Cases","Sentences","Edited macro-F1","Sentence-joint macro-F1","Edited accuracy","Changed labels"),[
            (group,q["cases"],f'{q["sentences"]:,}',f'{q["candidate"]["macro_f1"]:.5f}',
             f'{q["parent"]["macro_f1"]:.5f}',pct(q["candidate"]["accuracy"]),f'{q["changed"]:,}')
            for group,q in conditioned["splits"].items()])
        lo,hi=conditioned["splits"]["test"]["paired"]["macro_f1"]
        p(f'The test case-paired 95% interval for macro-F1 change is {100*lo:+.2f} to {100*hi:+.2f} points.')
        if lo<=0<=hi:
            p("The release keeps sentence-joint as the supported supplied-sentence profile. The interval includes zero and the edited profile classified five fewer sentences correctly.")
        qa=result("sentence-conditioned-edges-qa")
        if qa and qa["candidate"]==conditioned["splits"]["test"]["candidate"]:
            lo,hi=qa["paired_to_aligned_archive"]["macro_f1"]
            p(f'An independent NumPy decoder reproduced the results. The paired 95% interval versus the best aligned author archive is {100*lo:+.2f} to {100*hi:+.2f} macro-F1 points.')
        links(("Conditional Random Fields","https://mlanthology.org/icml/2001/lafferty2001icml-conditional/"),
              ("Mind Your Neighbours","https://aclanthology.org/2024.lrec-main.987/"))
    joint=result("sentence-joint-edit")
    joint_frozen=result("sentence-joint-frozen")
    if joint and joint_frozen and "test" in joint["splits"]:
        h(3,"Joint correction of supplied sentence labels")
        p(f'The experiment uses {len(joint_frozen["training_cases"]):,} training judgments to write role transition scores and closed-form readouts. It tests a local estimate of the change in macro-F1 from assigning a role, together with a score for retaining the current label. Joint decoding can revise several adjacent sentences in one pass.')
        kind,style,ridge,macro,retention,rate,strength=joint["choice"]
        p(f'Validation selected the {kind} evidence, a nominal {rate:.0%} routing budget, macro-F1 weight {macro:g}, retention score {retention:g} and transition strength {strength:g}.')
        table(("Cohort","Cases","Macro-F1","Sentence-uncertainty F1","Accuracy","Eligible sentences","Label changes","95% interval for F1 change"),[
            (name,r["cases"],f'{r["candidate"]["macro_f1"]:.4f}',f'{r["preceding"]["macro_f1"]:.4f}',
             pct(r["candidate"]["accuracy"]),f'{r["eligible_sentences"]:,}',f'{r["modified"]:,}',
             f'{100*r["paired_to_preceding"]["macro_f1"][0]:+.2f} to {100*r["paired_to_preceding"]["macro_f1"][1]:+.2f} points')
            for name,r in joint["splits"].items()])
        qa=result("sentence-joint-qa")
        portable=result("sentence-joint-portable-qa")
        if qa and qa["scores"]==joint["splits"]["test"]["candidate"]:
            a,b,c,archive=(qa[k] for k in ("scores","preceding","without_transition_scores","aligned_archive"))
            table(("Matched test profile","Macro-F1","Accuracy","Reasoning F1","Decision F1"),[
                (name,f'{q["macro_f1"]:.5f}',f'{100*q["accuracy"]:.2f}%',f'{q["per_role_f1"]["reasoning"]:.4f}',
                 f'{q["per_role_f1"]["decision"]:.4f}')
                for name,q in (("Joint correction",a),("Same correction with zero transition scores",c),
                               ("Sentence-uncertainty",b),("Strongest aligned author archive",archive))])
            lo,hi=qa["paired_to_aligned_archive"]["macro_f1"]
            p(f'The paired 95% interval versus the strongest aligned archive is {100*lo:+.2f} to {100*hi:+.2f} macro-F1 points. The new scorer made {qa["change_counts"]["multi_sentence_blocks"]:,} groups of adjacent changes. The independent NumPy decoder reproduced all official test scores.')
            if portable and qa["paired_to_preceding"]["macro_f1"][0]>0:
                p(f'The CPU sentence-joint profile reproduced individual labels on {len(portable["cases"])} live judgments and uses the existing hierarchy checkpoint.')
        latency=result("sentence-joint-cpu-latency")
        if latency:
            table(("Supplied sentences","Input words","Uncertainty median","Joint median"),[
                (q["candidate_units"],f'{q["whitespace_tokens"]:,}',f'{q["medians"]["sentence-uncertainty"]:.2f} s',
                 f'{q["medians"]["sentence-joint"]:.2f} s') for q in latency["cases"]])
        links(("Bridging Viterbi and posterior decoding","https://jmlr.org/papers/v15/lember14a.html"),
              ("Thresholding classifiers to maximize F1","https://arxiv.org/abs/1402.1892"))
    if archive_rows:
        h(3,"Archived predictions on identical sentence rows")
        table(("Author prediction archive","Nonblank rows","Macro-F1","Accuracy"),[
            (k.replace("_"," "),f'{v["scores"]["rows"]:,}',f'{v["scores"]["macro_f1"]:.4f}',pct(v["scores"]["accuracy"]))
            for k,v in sorted(archive_rows["archives"].items(),key=lambda x:-x[1]["scores"]["macro_f1"])])
        p("The eight files contain numerically aligned labels for every official test row. Scores exclude the same 13 blank sentences. The paper's higher headline uses its published run, which remains a separate reference.")
        links(("The leading InLegalBERT prediction archive","https://huggingface.co/datasets/L-NLProc/LegalSeg_InLegalBERT_Predictions"))
        p("SeqTNS reports 0.83 macro-F1 for sentence classification on the Bhattacharya et al. corpus. That seven-role taxonomy includes statutes and precedents. The LegalSeg benchmarks here use the seven categories and judgments defined in LegalSeg.")
        links(("SeqTNS benchmark","https://aclanthology.org/2025.findings-ijcnlp.50/"))
    archive=result("archived-sentence-span-references")
    named=result("named-hierarchy-span-oracle")
    second_replay=result("independent-archive-replay")
    if archive or named:
        h(2,"Published labels projected onto known spans")
        comparison=[]
        if archive:
            for arm, x in archive["scored_release_files"].items():
                o=x["scores"]
                comparison.append([arm.replace("_"," "),span(o),pct(o["token_accuracy"]),"Original LegalSeg sentence cuts supplied"])
        if second_replay:
            for k,display in (("mtl","MTL, matched by gold case sequence"),("role_aware_ci","Role-Aware CI archive")):
                z=second_replay[k]["scores"]
                comparison.append([display,span(z),pct(z["token_accuracy"]),"Original LegalSeg sentence cuts supplied"])
        if named:
            for arm,o in named["scores"].items():
                comparison.append([arm.replace("_"," "),span(o),pct(o["token_accuracy"]),"Original LegalSeg sentence cuts supplied"])
        table(("Sentence output source","Exact span F1","Token accuracy","Input information"),comparison)
        if archive and named:
            ranked=[(k,v["scores"]) for k,v in archive["scored_release_files"].items()]
            ranked.append(("named hierarchy",named["scores"]["named_hierarchy"]))
            if second_replay:
                ranked.extend((k,second_replay[k]["scores"]) for k in ("mtl","role_aware_ci"))
            top,maxm=max(ranked,key=lambda x:x[1]["exact_labeled_spans"]["f1"])
            if trade:
                v=trade["tasks"]["whole"]["scores"]["test"]["candidate"]
                if discourse and "whole" in discourse["tasks"]:
                    v=max((v,*[o["scores"]["test"]["candidate"] for o in discourse["tasks"]["whole"]["profiles"].values()]),key=lambda o:o["exact_labeled_spans"]["f1"])
                if role_edge_test:
                    v=max((v,*(j["candidate"] for j in role_edge_test.values())),key=lambda o:o["exact_labeled_spans"]["f1"])
                p(f'The new raw-text profile reaches {span(v)}, compared with {span(maxm)} for the highest of the {len(ranked)} published runs re-scored here ({top.replace("_"," ")}). This is the highest exact-span score in this comparison. The benchmark in the LegalSeg paper measures seven-role macro-F1 at supplied sentence boundaries.')
        paired=result("archive-span-paired-comparison")
        if paired:
            h(3,"Paired comparison against the strongest archived projection")
            table(("Raw-text profile","Exact span F1","95% interval for F1 gain","Token accuracy","95% interval for token accuracy gain"),[
              ({"plain":"Original edit","exact":"Role-specific exact spans","balanced":"Token-balanced edit","discourse_exact":"Discourse exact","discourse_balanced":"Discourse balanced"}[k],
               span(x["scores"]), f'{100*x["paired_gap_95pct"][0][0]:+.1f} to {100*x["paired_gap_95pct"][0][1]:+.1f} points',
               pct(x["scores"]["token_accuracy"]), f'{100*x["paired_gap_95pct"][1][0]:+.1f} to {100*x["paired_gap_95pct"][1][1]:+.1f} points')
               for k,x in paired["raw_text_candidates"].items()])
        p("These references convert sentence labels from the same LegalSeg test set into contiguous role spans. The independent raw-input task has to find its own candidate offsets. They are a strong diagnostic and their scores were recomputed from released predictions.")
        provenance=result("independent-archive-provenance")
        if provenance:
            q=provenance["hierarchy"]
            p(f'The two public hierarchical files contain the same {q["rows_in_both_prediction_streams"]:,} predictions. The flat file pairs those predictions with the first labels of the unfiltered test stream. Correcting the case and row alignment changes {q["different_gold_positions_vs_filtered"]:,} reference labels and raises its sentence macro-F1 from 0.270 to 0.598. The Role-Aware label-definitions file matches the official validation labels in order.')
        if second_replay:
            ll=second_replay["rhetoricllama_first_line"]
            lo,hi=ll["exact_span_f1_bounds"]
            p(f'A separate RhetoricLLaMA replay matches all official texts and case IDs. The conservative first-line parser resolves 149,155 rows. Allowing every possible role on the remaining 726 gives an outer exact-span F1 range of {lo:.3f} to {hi:.3f}. Those are parsing sensitivity bounds. The paper does not publish this measure.')
        links(("Official LegalSeg predictions","https://huggingface.co/collections/L-NLProc/legalseg-models"),("LegalSeg paper","https://aclanthology.org/2025.findings-naacl.63/"))
    matched=result("focused-span-gemini-replay")
    if matched:
        h(2,"Replay against Gemini on identical passages")
        ref=matched["frozen_gemini_span_reference"]
        table(("Model","Exact span F1","Token accuracy"),[["Gemini Decisions",span(ref),pct(ref["token_accuracy"])]]+
          [[k.replace("_"," "),span(v["scores"]),pct(v["scores"]["token_accuracy"])] for k,v in matched["candidates"].items()])
        p("The held-out API set contains 12 previously screened passages and 64 units. All profiles use the recorded original requests and same candidate boundaries. New weight choices were selected on local validation and compared with the archived Gemini answers.")
        best_name,best_api=max(matched["candidates"].items(),key=lambda item:item[1]["scores"]["exact_labeled_spans"]["f1"])
        lo,hi=best_api["span_f1_token_accuracy_gap_ci95"][0]
        p(f'The strongest measured local span profile on this small set is {best_name.replace("_"," ")}. Its paired exact-span F1 gap to Gemini is {100*lo:+.1f} to {100*hi:+.1f} points at 95% confidence. The whole-judgment discourse profiles lose more accuracy on the clipped inputs. Gemini still needs a matched full-judgment evaluation before a parity claim.')
        links(("Gemini Decisions API launch and pricing","https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/"))
        if role_edge_published:
            replay=role_edge_published["matched_gemini"]
            table(("Additional role-pair replay","Exact span F1","Token accuracy"),[
              (name,span(q["scores"]),pct(q["scores"]["token_accuracy"])) for name,q in replay["candidates"].items()])
    corpus_memory=result("corpus-residual-memory")
    if corpus_memory:
        h(2,"Targeted keys from official training data")
        table(("Profile","20-case exact span F1","Parent"),[
            (label,span(profile["scores"]["test_screen"]["candidate"]),span(profile["scores"]["test_screen"]["incumbent"]))
              for label,profile in corpus_memory["profiles"].items()])
        p("This trial writes a sparse associative memory of selected official training activations. It stores true-role corrections over an unchanged encoder.")
    quote=result("quotation-span-edit")
    if quote:
        h(2,"Quotation and statute cues")
        table(("Profile","712-case exact span F1","Candidate token accuracy","Parent token accuracy","95% interval for F1 gain"),[
            (label,f'{test["candidate"]["exact_labeled_spans"]["f1"]:.5f}',pct(test["candidate"]["token_accuracy"]),
             pct(test["incumbent"]["token_accuracy"]),f'{100*low:+.3f} to {100*high:+.3f} points')
              for label,profile in quote["profiles"].items() for test in [profile["scores"]["test"]]
              for low,high in [test["span_and_token_gain_ci95"][0]]])
        p("Quotation and statute cues gave a small token accuracy gain. Exact-profile span F1 was effectively unchanged and balanced-profile span F1 declined. The released weights retain the previous cues.")
        links(("LegalSeg annotation examples","https://aclanthology.org/2025.findings-naacl.63.pdf"))
    duration=result("empirical-duration-write")
    if duration:
        h(2,"Role duration estimated from training examples")
        table(("Profile","Cohort","Cases","Exact span F1","Parent F1","Token accuracy","Parent accuracy","Selected curve"),[
            (name,cohort,q["cases"],f'{q["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
             f'{q["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(q["candidate"]["token_accuracy"]),
             pct(q["parent"]["token_accuracy"]),f'{profile["selected"][0]}, gain {profile["selected"][1]:g}')
            for name,profile in duration["profiles"].items() for cohort,q in profile["scores"].items()])
        p("The decoder offsets the scores for role runs of up to 32 candidate units. The written offsets come from observed frequencies in the original seven-role training cases. Validation selects the curve and weight before the test set is re-scored.")
        p("The exact profile selected a zero duration correction. The balanced profile selected pooled frequencies and lost exact span F1 on the test cases. The release keeps both parent profiles.")
    long=result("length-adaptive-span-write")
    if long:
        h(2,"On long judgments")
        values=[]
        for k,long_profile in long["profiles"].items():
            for slice2 in ("confirm_long","test_long","test_all"):
                m=long_profile["scores"][slice2]
                values.append([k,slice2.replace("_"," "),m["cases"],span(m["candidate"]),span(m["incumbent"]),pct(m["candidate"]["token_accuracy"]),
                               f'{long_profile["width"]} units, {long_profile["local_mix"]:g} local weight'])
        table(("Profile","Cohort","Judgments","Exact span F1","Control","Token accuracy","Validation choice"),values)
        p("This edit combines the existing whole-document state and a locally recomputed state. A validation-only selection decides the direct weight write. The edit applies to inputs with at least 8,000 whitespace tokens.")
    fusion=result("role-specialized-encoder-fusion")
    if fusion:
        h(2,"Role-specific writes from several encoders")
        rows=[]
        for name,t in fusion["tasks"].items():
            for k,v in t["profiles"].items():
                x=v["scores"]["test"]
                rows.append([name,k,x["cases"],span(x["candidate"]),span(x["incumbent"]),str(v["spec"])])
        table(("Task","Profile","Cases","Exact span F1","Base","Validation choice"),rows)
        p("The edit physically joins centered encoder heads in one block matrix. Selection can assign zero contribution from the new encoders.")
    boundary_write=result("contextual-boundary-write")
    if boundary_write:
        h(2,"Input-dependent transition weights")
        rows=[]
        for k,head in boundary_write["profiles"].items():
            q=head["scores"]["test"]
            rows.append([k,q["cases"],span(q["candidate"]),span(q["incumbent"]),span(q["shuffled_boundary_control"]),str(head["strength"])])
        table(("Profile","Cases","Exact span F1","Base","Shuffled signal","Edit gain"),rows)
        p("A closed-form discriminant maps surrounding emission changes into a bias applied only when the decoder changes role. Within-document shuffling keeps the score distribution and tests whether the specific change location matters.")
    independent=result("contextual-boundary-independent-validation")
    if independent:
        p(f'The frozen boundary profiles were checked on {independent["cases"]} separate validation judgments without reselecting their gains.')
    hard=result("hard-preservation-emitter")
    if hard:
        h(2,"Direct emitter edits with a protected subspace")
        rows=[]
        for name,x in hard["profiles"].items():
            for k,protected_profile in x["choices"].items():
                t=protected_profile["scores"]["screen"]
                rows.append([name,k,span(t["candidate"]),span(t["incumbent"]),str(protected_profile["edit"])])
        table(("Profile","Variant","20-case exact span F1","Same-family control","Validation choice"),rows)
        p("The readout is fit in closed form using official training annotations. The edit is orthogonal to a chosen subspace of correctly classified activations. A random subspace and the unchanged parent serve as controls.")
    new_keys=result("encoder-source-keys-granite")
    if new_keys:
        h(2,"Granite with annotated training keys")
        table(("Cases","Profile","Exact span F1","Sequence control","Selected head"),[
            (x["scores"]["test"]["cases"],n+" "+label,span(x["scores"]["test"]["candidate"]),span(x["scores"]["test"]["hierarchy"]),x["edit"])
            for n,a in new_keys["tasks"].items() for label,x in (("blend",a["best"]),("encoder-only emissions",a["standalone_encoder"]))])
        p("This is a data-derived closed-form classifier fit with 128 training keys per role. The pretrained encoder remains fixed. The comparison measures whether true source style improves on authored examples.")
    contextual_deberta=result("deberta-span-write")
    deberta=contextual_deberta or result("deberta-isolated-screen")
    if deberta:
        h(2,"DeBERTa-v3-xsmall on whole judgments")
        table(("Profile","Test cases","Exact span F1","Token accuracy","Readout and context selected"),[
            (label.replace("_"," "),q["cases"],span(q["candidate"]),pct(q["candidate"]["token_accuracy"]),
             t["edit"]+", "+t["mode"]) for label,t in deberta["validation_selected"]["whole"].items()
             if label != "grid" for q in [t["scores"]["test"]]]+
            [("Released discourse hierarchy",20,span(deberta["current_discourse_control"]["whole"]["test"]),
              pct(deberta["current_discourse_control"]["whole"]["test"]["token_accuracy"]),"Full judgment")])
        p(f'The {deberta["encoder_parameters"]:,}-parameter bidirectional encoder supplies activations to seven written role scores. The classifier write uses either authored examples or keys from the official training split. The original DeBERTa backbone stays fixed. All entries are measured on the same screening judgments.')
        if not contextual_deberta:
            p("This table covers the completed isolated-unit screen. The separate contextual-window screen is still computing activations.")
        links(("Pinned DeBERTa checkpoint","https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall/tree/a150876415327c80daeff35ca6f68f5ed8cf5c24"),
              ("DeBERTaV3 paper","https://arxiv.org/abs/2111.09543"))
    recurrent=result("lstm-input-subspace-write")
    if recurrent:
        h(2,"Direct edits to the recurrent encoder")
        table(("Cohort","Cases","Edited exact span F1","Parent F1","Edited token accuracy","Parent token accuracy"),[
            (cohort,q["cases"],span(q["candidate"]),span(q["parent"]),pct(q["candidate"]["token_accuracy"]),
             pct(q["parent"]["token_accuracy"])) for cohort,q in recurrent["scores"].items()])
        p(f'The edit writes both directions of the existing document LSTM input matrices and biases. It increases or decreases sensitivity to role-contrast directions while preserving the gate input at the training mean. The selection was {recurrent["selected"][0]} with gain {recurrent["selected"][1]:g}.')
    margins=result("raw-conditioned-emitter-full") or result("raw-conditioned-emitter-screen")
    raw_freeze=result("raw-conditioned-emitter-frozen")
    if margins or raw_freeze:
        h(2,"Direct classifier corrections on raw input")
        profiles=margins["profiles"] if margins else raw_freeze
        table(("Profile","Cohort","Cases","Edited exact span F1","Parent F1","Edited token accuracy","Parent token accuracy"),[
            (name,k,q["cases"],span(q["candidate"]),span(q["parent"]),pct(q["candidate"]["token_accuracy"]),
             pct(q["parent"]["token_accuracy"])) for name,item in profiles.items() for k,q in item["scores"].items()])
        p("The edit uses the inference splitter on official training cases. It solves for a direct change to the seven emitter rows using local CRF margins at parent-model errors. High-purity units contribute to the estimate. One candidate upweights units that touch a reachable role boundary.")
    relabel=result("role-relabel-write")
    if relabel:
        h(2,"Correction at unchanged span boundaries")
        table(("Cohort","Cases","Edited exact span F1","Parent F1","Edited token accuracy","Parent token accuracy"),[
            (k,q["cases"],span(q["candidate"]),span(q["parent"]),pct(q["candidate"]["token_accuracy"]),
             pct(q["parent"]["token_accuracy"])) for k,q in relabel["scores"].items()])
        p("A written linear discriminator examines segments proposed by the existing decoder. It can change a None span to petitioner argument or a facts span to reasoning. Its evidence includes the existing model scores, nearby predictions, and discourse cues. The original character offsets are preserved. A conflicting neighboring role blocks the correction.")
    nli=result("nli-minilm-span-edit")
    if nli:
        h(2,"Entailment readout on the clipped-passage screen")
        table(("Profile","Held-out cases","Exact span F1","Incumbent"),[(k,x["scores"]["test"]["cases"],
           span(x["scores"]["test"]["candidate"]),span(x["scores"]["test"]["incumbent"])) for k,x in nli["model_profiles"].items()])
        p("A native three-output entailment classifier scores each role definition. The patch gives each role a weighted combination of contradiction, neutral and entailment evidence, with an optional fixed null-input bias.")
        rep=result("nli-minilm-full-confirmation")
        if rep:
            h(3,"Frozen follow-up across 120 passages")
            table(("Passages","Cases","Written NLI blend","Original hierarchy","Discourse edit"),
                  [[k.replace("_"," "),v["cases"],span(v["metrics"]["selected"]),span(v["metrics"]["base"]),span(v["metrics"]["discourse"])]
                   for k,v in rep["cohorts"].items()])
        links(("NLI model card","https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768"),
              ("Text classification as entailment","https://aclanthology.org/D19-1404/"))
    qwen=result("qwen-sixhundred-engineering")
    if qwen:
        h(2,"Qwen3-0.6B readout pilot")
        highlights=[("native_runtime","Native Qwen vocabulary scores"),("bias_control","Written bias"),("associative","Written associative readout"),("gemini_archive","Archived Gemini Decisions")]
        table(("System","Token accuracy","Exact span F1"),[(label,pct(qwen["results"][k]["token_accuracy"]),span(qwen["results"][k])) for k,label in highlights])
        p("A gradient-free associative edit writes 7,175 values across the output matrix and bias. The pilot contains two judgments, ten evaluated units and 270 tokens. Qwen receives one unit per input. The archived Gemini request also had passage context. The full twelve-passage comparison and 120-case test are unmeasured.")
        links(("Pinned Qwen GGUF","https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a"))
    julia=result("julia-choice-span-write")
    if julia:
        h(2,"Julia 1 typed-choice encoder")
        table(("Cohort","Cases","Profile","Exact span F1","Token accuracy"),[
            (k.replace("_"," "),v["cases"],label.replace("_"," "),span(q),pct(q["token_accuracy"]))
            for k,v in julia["scores"].items() for label,q in v["scores"].items()])
        p("Julia 1 is a 144.3M-parameter bidirectional encoder with an option-conditioned head. It receives seven written role definitions and a target unit with immediate neighboring text. A closed-form write calibrates the seven output scores and keeps the encoder fixed. This experiment uses passages and tests the same input offsets as the archived Gemini subset.")
        links(("Julia 1 model card","https://huggingface.co/SupersonicLabs/Julia-1"),
              ("Pinned Julia 1 ONNX","https://huggingface.co/SupersonicLabs/Julia-1-ONNX/tree/82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8"))
    h(2,"Boundary coverage before classification")
    coverage=result("boundary-span-coverage")
    if coverage:
        table(("Input","Gold spans","Endpoints covered today","Endpoints with added sentence cuts","Endpoints with all proposed cuts"),[(label,value["current"]["gold_spans"],pct(value["current"]["reachable_gold_span_recall"]),pct(value["sentence"]["reachable_gold_span_recall"]),pct(value["all"]["reachable_gold_span_recall"])) for key,label in (("whole","712 complete judgments"),("fragment","120 clipped passages"),("matched","12 Gemini-matched passages")) for value in [coverage[key]]])
    oracle=result("span-boundary-oracle")
    if oracle:
        o=oracle["tasks"]["whole"];p(f'With an oracle supplying the roles, the current candidate cuts could reach {o["current"]["exact_span_f1_attainable"]:.3f} exact span F1 across all 712 judgments. The broader sentence proposal raises this to {o["sentence"]["exact_span_f1_attainable"]:.3f}. These diagnostic scores include the unmatched spans needed to cover the entire text. They quantify the room for models to improve without changing the splitter.')
    p("If either endpoint is absent, the decoder cannot emit the exact span. Coverage measures the fraction of gold spans with both candidate endpoints, even when a real decoder would make role errors. The exact-span experiment selects a boundary policy on validation and measures false positives on the test set.")
    links(("Legal sentence boundary detection","https://aclanthology.org/W19-2204/"))
    cut_study=result("optional-boundary-lattice")
    join_study=result("optional-merge-lattice")
    relocation=result("boundary-relocation")
    calibrated_cut=result("calibrated-boundary-lattice")
    if cut_study or join_study or relocation or calibrated_cut:
        h(2,"Letting the decoder choose between different text boundaries")
        rows=[]
        for label,trial in (("Optional split",cut_study),("Optional join",join_study),
                            ("Boundary relocation",relocation),("Calibrated split",calibrated_cut)):
            if not trial:
                continue
            for name,s in trial["splits"].items():
                lo,hi=s["gap_ci95"]["exact_span_f1"]
                rows.append([label,name,s["cases"],f'{s["candidate"]["exact_labeled_spans"]["f1"]:.4f}',
                    f'{s["parent"]["exact_labeled_spans"]["f1"]:.4f}',pct(s["candidate"]["token_accuracy"]),
                    pct(s["parent"]["token_accuracy"]),f'{100*lo:+.2f} to {100*hi:+.2f} points',
                    f'{s.get("opened",s.get("joint_windows",s.get("moves",0))):,}'])
        table(("Method","Cohort","Cases","Exact span F1","Parent F1","Token accuracy","Parent accuracy","95% interval for F1 gain","Choices opened"),rows)
        p("The optional split keeps each original unit available as one branch. Other branches split at a proposed character offset and assign distinct roles to the two pieces, using frozen local embeddings and a written boundary prior. The optional join lets adjacent short units share a role, using their combined embedding to update that role score. Each decoder matched exhaustive path scoring on small synthetic examples. Both select weights before official test replay.")
        if relocation:
            p("The relocation decoder can shift a predicted transition up to one original unit left or right, while preserving the sequence of roles. The split proposal, prior and local logits are reused. Ordered dynamic programming prevents crossing or collapsing neighboring transitions.")
        if calibrated_cut:
            fit=result("calibrated-boundary-train")
            if fit:
                p(f'A regularized discriminant uses {fit["proposals"]:,} proposals from {len(fit["cases"]):,} training judgments, with {fit["positives"]:,} real role changes. It combines the lexical prior with left and right frozen embedding scores and neighboring emissions. The coefficients are computed from class means and covariance in one linear solve.')
            ranking=result("calibrated-boundary-ranking")
            if ranking:
                table(("Boundary ranking on confirmation","Average precision","Precision among top 5%","Recall among top 5%"),[
                    (label,f'{r["average_precision"]:.3f}',pct(r["top_fraction"]["0.05"]["precision"]),
                     pct(r["top_fraction"]["0.05"]["recall"]))
                    for key,label in (("lexical_only","Lexical prior"),("closed_form","Closed-form discriminant"))
                    for r in [ranking[key]]])
        if cut_study:
            test=cut_study["splits"].get("test")
            train=result("optional-boundary-train")
            if train:
                p(f'In the {train["cases"]:,} training judgments, {train["gold_role_changes"]:,} of {train["options"]:,} proposed cuts coincide with an annotated change of role.')
            if test:
                p(f'The 712 judgments contain {test["options"]:,} additional candidate offsets. They include {test["reachable_extra_gold_changes"]:,} annotated role changes. The selected decoder opens {test["opened"]:,} cuts, including {test["opened_at_gold_change"]:,} annotated changes.')
        audit=result("optional-boundary-reachability")
        if audit:
            base,pool=[audit["sets"][key] for key in ("original","optional")]
            p(f'With exact roles and unrestricted use of those offsets, {pool["reachable"]:,} reference spans have both ends available, up from {base["reachable"]:,}. The corresponding oracle F1 changes from {base["oracle_exact_span_f1"]:.3f} to {pool["oracle_exact_span_f1"]:.3f}. This diagnostic permits more than one new cut inside an original unit.')
            table(("Role","Gold spans","Reachable before","Additional reachable spans"),[
                (role.replace("_"," "),f'{n:,}',f'{base["reachable_by_role"][role]:,}',
                 f'{pool["reachable_by_role"][role]-base["reachable_by_role"][role]:,}')
                for role,n in audit["gold_roles"].items()])
        links(("Semi-Markov conditional random fields","https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html"),
              ("Joint span segmentation and rhetorical role labeling","https://arxiv.org/abs/2302.06448"))
    h(2,"Open encoders under one billion parameters")
    table(("Checkpoint","Parameters","Context design in this experiment","Edited tensors"),[
          ("IBM Granite Embedding Small English R2","47M","Bidirectional ModernBERT. Compare each unit alone with both-sided token context in 384-token windows.","Seven-way projection, CRF and final normalization trial"),
          ("all-MiniLM-L6-v2","23M","Six-layer bidirectional encoder with isolated and contextual token pooling under the same window scheme.","Seven-way projection and CRF"),
          ("Alibaba GTE ModernBERT Base","149M","Bidirectional 8,192-token architecture. Use common 384-token windows for an initial size-controlled comparison.","Seven-way projection, CRF and final normalization trial"),
          ("LegalSeg hierarchy","16.82M","Bidirectional word and document LSTMs see the sequence of candidate units.","Seven-role transition weights and authored segment-length costs")])
    p("The pinned ModernBERT cards list an 8,192-token native limit. The common 384-token window experiment does not test that limit. For a long input, disjoint portions contribute once to unit pooling and overlapping context supplies neighboring text. No gold sentence cuts enter inference.")
    links(("Granite model collection","https://github.com/ibm-granite/granite-embedding-models"),("MiniLM model card","https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"),("GTE model card","https://huggingface.co/Alibaba-NLP/gte-modernbert-base"),("ModernBERT paper","https://arxiv.org/abs/2412.13663"))
    h(2,"How the editing literature applies")
    p("Humza and Omar Nusrat audited 433- to 4,801-parameter networks with planted errors and tested each proposed repair against known rules. They reported a median of three changed parameters across successful repairs. This is direct evidence that sparse writes can fix identifiable behavior in small systems. It motivates preserving the largely intact checkpoint and checking unrelated behavior independently.")
    links(("Clinical weight-space audit","https://arxiv.org/abs/2611.05341"))
    p("Ok and Lee write sparse associative tables from examples. We give each role an authored set of representative utterances and write a seven-row readout from their activations. We compare simple role means with a regularized correction for each key.")
    links(("Weight-Space Programming Doesn't Need Transformers","https://arxiv.org/abs/2611.22055"),
          ("Associative memory code and weights","https://github.com/effl-lab/Weight-Space-Programming"))
    p("ROME writes a rank-one update in a particular MLP layer to change a factual association. MEMIT and EMMET generalize to batches of edits. The earlier final-MLP InLegalBERT trial used a related low-rank write to a pretrained projection. It helped on some sentence samples while losing weighted validation accuracy. The new encoder readout starts from class prototypes over frozen hidden states. The normalization trial changes existing pretrained coefficients and verifies those changes in an actual forward pass.")
    links(("ROME","https://arxiv.org/abs/2202.05262"),("MEMIT","https://arxiv.org/abs/2210.07229"),("EMMET","https://arxiv.org/abs/2403.14236"))
    p("AlphaEdit and CrispEdit preserve behavior by limiting the directions an update may modify. This experiment adapts the idea by estimating activation directions from 192 unlabeled LegalSeg training excerpts. One candidate protects none of those directions. Others project the correction away from the leading 16 or 48 directions. The resulting edit is a covariance-based approximation and it supplies no general capability guarantee. CrispEdit itself uses a different curvature estimate and projected optimization.")
    links(("AlphaEdit","https://arxiv.org/abs/2410.02355"),("CrispEdit","https://proceedings.mlr.press/v306/ikram26a.html"))
    p("The segment-duration trial adds a role-dependent weight for a span lasting one, two, three or at least four proposed units. A zero duration matrix reproduces the ordinary CRF decoder. Legal speaker minimal pairs keep the legal proposition fixed and vary who is saying it and why. They provide keys for a separate write to the existing classifier rows.")
    links(("Semi-Markov CRFs","https://proceedings.neurips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html"),
          ("Attribution-guided model rectification","https://arxiv.org/abs/2603.15656"))
    p("A separate analytic residual edit estimates a correction from misclassified LegalSeg training rows and penalizes changes on rows the parent already labels correctly. This uses training annotations to fit a closed-form change to existing classifier weights. It has no optimizer or gradient updates and is tracked separately from edits written entirely from authored cues.")
    p("Task arithmetic edits several tensors together using differences between checkpoints that share a base. Our earlier compatible-weight interpolation lost to keeping its stronger parent. Blending output scores from different model families tests complementary representations without requiring their hidden weights to align.")
    links(("Editing Models with Task Arithmetic","https://arxiv.org/abs/2212.04089"))
    p("Santurkar and colleagues edit existing decision rules through low-rank weight writes keyed by activations. Kim and Do derive class-specific representation changes from fine-tuning and show how to map them into classifier weights. Our recurrent gate experiment instead uses role contrasts measured directly on frozen features. It applies a covariance-aware change to existing weights with a center-preserving bias correction. LEACE provides a related example of using covariance structure for an exact linear representation intervention, with a different objective.")
    links(("Rewriting classifier prediction rules","https://arxiv.org/abs/2112.01008"),
          ("Class vectors for classifier editing","https://proceedings.neurips.cc/paper_files/paper/2025/hash/4f0484db17ae7d2edb3c2031b885ba76-Abstract-Conference.html"),
          ("LEACE","https://arxiv.org/abs/2306.03819"))
    h(2,"Research benchmark and evidence standard")
    p("LegalSeg Table 3 reports 0.770 macro-F1 for the hierarchical BiLSTM-CRF using sentence boundaries and document context. The aligned archive scores include 0.598 for named hierarchy outputs, 0.621 for TransformerOverInLegalBERT and 0.633 for the previous-role InLegalBERT release. Our previous edited hierarchy scored 0.567 in this setting.")
    p("HiCoBERT reports 0.5044 macro-F1 and 0.5047 span F1 on 679 LegalSeg test judgments after mapping the human labels to its four-section taxonomy and building section-consistent segments. It merges reasoning and both arguments, absorbs the none role into neighboring sections, and excludes 33 documents above its 118-segment cap. This comparison measures cross-jurisdiction transfer under that protocol.")
    links(("HiCoBERT cross-dataset evaluation and Table 16","https://pmc.ncbi.nlm.nih.gov/articles/PMC13056923/"))
    p("SeqTNS reports 0.83 macro-F1 on the seven-role Bhattacharya corpus of 50 Supreme Court judgments, using a taxonomy that includes statute and precedent. Its similarity-based sequence approach is relevant to model design. Its reported result uses a different corpus and label set.")
    links(("SeqTNS Section 2 and Table 1","https://aclanthology.org/2025.findings-ijcnlp.50/"),
          ("Coupling local context and global prototypes for rhetorical role labeling","https://arxiv.org/abs/2603.03856"))
    paper_figure=result("legalseg-paper-figure-recheck")
    if paper_figure:
        f=paper_figure["figure_3"]
        p(f'Figure 3 contains {f["rows"]:,} predictions under the hierarchical BiLSTM-CRF label. Recomputing all seven classes yields {f["macro_f1"]:.4f} macro-F1 and {f["accuracy"]*100:.2f}% accuracy. All 49 counts match the author’s Hier_BiLSTM_predictions.csv. The run behind Table 3 remains unverified from the release. The TDM-Gauntlet preprint independently reports that it could reproduce ToInLegalBERT and could not recover the 0.770 headline from the released predictions.')
        links(("LegalSeg Figure 3",paper_figure["paper"]),("TDM-Gauntlet preprint",paper_figure["independent_reproduction_note"]))
    p("TDM-Gauntlet reports 0.520 macro-F1 for zero-shot Jev on 149,875 LegalSeg sentence decisions. The repository currently lists a README and license.")
    links(("TDM-Gauntlet repository","https://github.com/NaiveNeuron/TDM-Gauntlet"))
    links(("LegalSeg","https://aclanthology.org/2025.findings-naacl.63/"),("Jev typed decision benchmark","https://arxiv.org/abs/2611.20191"))
    frozen_rows=result("role-edge-given-sentences")
    if frozen_rows:
        table(("Role-pair profile with known boundaries","Sentences","Edited macro-F1","Parent macro-F1","Edited accuracy","Parent accuracy"),[
            (name,f'{r["candidate"]["rows"]:,}',f'{r["candidate"]["macro_f1"]:.3f}',f'{r["parent"]["macro_f1"]:.3f}',
             pct(r["candidate"]["accuracy"]),pct(r["parent"]["accuracy"])) for name,r in frozen_rows["profiles"].items()])
        p("The same weights selected for raw spans were applied at supplied boundaries across the original seven-role test. The comparison to the parent uses the same rows and document context. The published headline uses the paper's own run.")
    p("A claim for the published sentence-label task requires the same seven labels, document inputs and macro-F1 protocol. The raw span comparisons use identical reconstructed text for each official case and score exact offsets and roles. Candidate weights are selected on validation. Aggregate test results from the same 712 judgments have informed the broader sequence of experiments, so fresh cases would provide a stronger final replication.")
    template="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AttoFlow | Judgment span weight editing</title>
    <style>body{margin:0;background:#f4f6f3;color:#202a2e;font:16px/1.55 system-ui,sans-serif}main{box-sizing:border-box;max-width:1080px;margin:auto;padding:44px 5vw 70px;background:#fff}h1,h2{font-family:Georgia,serif;line-height:1.16;font-weight:400}h1{font-size:3rem}h2{margin-top:2.2rem;font-size:1.85rem}header{border-bottom:2px solid #10665b;color:#10665b;font-size:1.08rem;font-weight:700;padding-bottom:12px}p{max-width:83ch}a{color:#0c675d}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.93rem}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dbe4df;padding:10px}th{background:#edf3ef;color:#0e564c}.links{font-size:.89rem;margin-top:0}</style></head><body><main><header>AttoFlow · Research notes</header>"""
    OUT.parent.mkdir(exist_ok=True,parents=True)
    tmp=OUT.with_suffix(".part")
    tmp.write_text(template+"\n"+"\n".join(html)+"</main></body></html>",encoding="utf-8")
    tmp.replace(OUT)

if __name__=="__main__":main()
