"""Build a common source of verified claims and tables for the PDF and offline HTML brief."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL={
 "Gemini choice API":"https://aistudio.google.com/api-reference/decisions",
 "Gemini rate limits":"https://aistudio.google.com/docs/rate-limits#gemini-decision",
 "LegalSeg paper":"https://aclanthology.org/2025.findings-naacl.63/",
 "Official splits":"https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV/tree/31217aa845866bbd18a8ad567a785e6f5485435a",
 "Published weights":"https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT",
 "Paragraph dataset and license":"https://data.mendeley.com/datasets/97khn9nswp/1",
 "MARRO release":"https://huggingface.co/datasets/subinay612/MARRO_Rhetorical-Role-Labeling",
 "Independent Jev evaluation":"https://arxiv.org/abs/2611.20191",
 "Horizon Labs":"https://horizon-labs.ai/",
 "Clinical weight-space audit":"https://arxiv.org/abs/2611.05341",
 "Associative memory paper":"https://arxiv.org/abs/2611.22055",
 "Associative memory code":"https://github.com/effl-lab/Weight-Space-Programming",
 "Gemma 4 model card":"https://huggingface.co/google/gemma-4-E2B-it",
 "Quantized Gemma weights":"https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/tree/2a8538c2e2d4687953a86c422c7c623cfd022880",
 "Content-free calibration":"https://proceedings.mlr.press/v139/zhao21c.html"}
NAMES={"gemini":"Gemini Decision 1","original":"Released BERT A","direct":"A with centroid edit","associative_direct":"A with associative edit",
       "context_parent":"Released BERT B","associative_context":"B with associative edit","instruction_native":"Gemma native letter scores",
       "instruction_unchanged":"Gemma copied readout","instruction_edited":"Gemma written readout"}
ROLE_NAMES=["None","Facts","Issue","Petitioner argument","Respondent argument","Reasoning","Decision"]
def rd(name):
    f=ROOT/"results"/(name+".json")
    return json.loads(f.read_text()) if f.is_file() else None
def f1(m):return f'{m:.3f}'
def pct(m):return f'{100*m:.1f}%'
def gap(v):return f'{100*v:+.1f}'
def ci(v):return f'{gap(v[0])} to {gap(v[1])}'
def acc(row,total):return f'{row["correct"]}/{total} · {pct(row["accuracy"])}'
def txt(s):return ("text",s)
def hdr(s):return ("heading",s)
def tbl(cols,rows,w):return ("table",cols,rows,w)
def refs(*r):return ("refs",[(k,URL[k]) for k in r])
def page(title,sub,*items):return {"title":title,"subtitle":sub,"items":list(items)}
def briefs():
    paired=rd("gemini-seven-pilot");sen=paired["results"]["sentence"]
    para=rd("gemini-seven-paragraphs")["results"]["multi_sentence"]
    joinedrec=rd("gemini-seven-homogeneous")
    joined=joinedrec["results"]["homogeneous"] if joinedrec else None
    docrec=rd("document-segmentation");doccloud=rd("gemini-seven-spans")
    extended=rd("gemini-seven-confirmation");novel=rd("instruction-projection-validation")
    first=rd("program-validation")["confirmation"];second=rd("associative-direct-validation");ctx=rd("associative-context-validation")
    old=rd("direct-interim-test");full=rd("direct-official-test");paper=rd("published-baselines")
    ceiling=rd("identical-inputs-ceiling")["test"]
    synthetic=rd("gemini-paired-synthetic")
    # The compact reference is frozen independently of these provider results.
    structural="associative_direct"
    mainkeys=["gemini",structural,"associative_context"]
    if novel and "instruction_edited" in sen["metrics"] and "instruction_edited" in para["metrics"]:mainkeys+=["instruction_unchanged","instruction_edited"]
    models=[]
    for k in mainkeys:
        models.append([NAMES[k],acc(sen["metrics"][k],sen["rows"]),f1(sen["metrics"][k]["macro_f1"]),
                       acc(para["metrics"][k],para["rows"]),f1(para["metrics"][k]["macro_f1"])])
    pages=[page("Can a local edit match Gemini?",
        "A developer submits one English sentence or paragraph from an Indian judgment. The classifier returns one dominant role from a seven-class taxonomy.",
        hdr("Matched human-gold pilot"),tbl(["System","Sentences correct","F1 (7)","Multi-sentence correct","F1 (7)"],models,(.26,.21,.10,.27,.16)),
        txt("Each system received the same excerpt bytes and answered with one of seven LegalSeg roles. The 112 single-sentence inputs and 21 multi-statement inputs have existing source labels. The latter are intact source units with monetary amounts or dates masked when present. The CSV gives no original paragraph-layout boundaries."),
        txt(f'The compact associative edit and Gemini agreed on {sen["paired_with_gemini"][structural]["seven_label_agreement_count"]}/112 sentence labels. Its accuracy gap was {gap(sen["paired_with_gemini"][structural]["accuracy_gap"])} percentage points, with a 95% case-bootstrap interval of {ci(sen["paired_with_gemini"][structural]["accuracy_gap_95pct"])}. On the 21 multi-statement source units the gap was {gap(para["paired_with_gemini"][structural]["accuracy_gap"])} points, with an interval of {ci(para["paired_with_gemini"][structural]["accuracy_gap_95pct"])}.'),
        hdr("What 100% would require"),
        txt(f'{ceiling["raw_inputs_with_incompatible_labels"]:,} distinct nonblank inputs recur verbatim with conflicting official test roles. Even an oracle that chooses the most frequent recorded role for each distinct excerpt misses at least {ceiling["minimum_errors_on_nonblank_rows"]:,} of {ceiling["nonblank_rows"]:,} scored rows. The resulting ceiling is {100*ceiling["empirical_maximum_accuracy"]:.2f}%. Another {ceiling["blank_rows"]} rows contain no classifiable text. A system that reads surrounding sentences receives additional evidence and has a different ceiling.'),
        refs("LegalSeg paper","Official splits","Gemini choice API"))]
    if extended:
        v=extended["results"]["sentence"];records=[]
        for k in mainkeys:
            if k not in v["metrics"]:continue
            m=v["metrics"][k];g=v["paired_with_gemini"].get(k)
            records.append([NAMES[k],acc(m,v["rows"]),f1(m["macro_f1"]),ci(g["accuracy_gap_95pct"]) if g else "Reference"])
        pg=page("An independent sentence comparison",
           "One anonymous eligible sentence per judgment, sampled without a class quota. All 196 cases are distinct from the 98 pilot cases.",
           tbl(["System","Correct / 196","Macro-F1","95% interval for accuracy gap, pp"],records,(.31,.24,.15,.30)),
           txt("The sampling frame contained 611 eligible test judgments. No issue-labeled sentence happened to be drawn. The fixed seven-class macro-F1 assigns zero F1 to that unsupported class. We excluded personal identifiers and sensitive case details before querying the API. LegalSeg’s gold label stayed hidden from the API. Always predicting the training-majority role, none, gives 65/196 (33.2%). The results describe the screened passages and their case weighting."),
           hdr("Where the answers diverge"),tbl(["Local readout","Both correct","Local only","Gemini only","Both wrong","Same label"],
             [[NAMES[k],p["both_correct"],p["only_local_correct"],p["only_gemini_correct"],p["both_wrong"],p["seven_label_agreement_count"]] for k,p in v["paired_with_gemini"].items() if k in mainkeys],
             (.30,.14,.13,.15,.14,.14)),
           txt("Accuracy counts matches to the human gold label. Agreement counts identical model responses, including shared errors. We resampled whole cases together for both systems 4,000 times. Positive gaps favor the local model."),
           txt("Gold support by role in this draw was "+", ".join(f'{k.replace("_"," ")} {vv}' for k,vv in v["gold_support"].items())+"."),
           refs("Official splits","Gemini choice API"))
        pages.append(pg)
    rolecols=["Human role","Gemini","A, edited","B, edited"]
    rolekeys=["gemini",structural,"associative_context"]
    if novel and "instruction_edited" in sen["metrics"]:rolecols+=["Gemma edit"];rolekeys+=["instruction_edited"]
    items=[]
    for j,label in enumerate(sen["labels"]):
        items.append([ROLE_NAMES[j]]+[f'{int(sen["metrics"][k]["confusion"][j][j])}/16' for k in rolekeys])
    pages.append(page("Which roles are being missed?",
        "Each pilot sentence role has 16 source annotations. These are correct counts within each role, so rare categories remain visible.",
        tbl(rolecols,items,[.34]+[.66/len(rolekeys)]*len(rolekeys)),
        txt(f'The sentence pilot spans {sen["cases"]} judgments. Generic transitions, cited legal propositions and submissions with an unstated speaker can support different readings in isolation. The real-corpus scores measure agreement with the source annotations.'),
        hdr("Controlled fictional examples"),
        txt(f'Gemini matched the intended role on 167/168 authored passages. The original A centroid edit matched on 111/168 and agreed with Gemini on {synthetic["models"]["direct"]["agrees_with_gemini"]}/168. The examples used two writing patterns and 12 scenarios. Their explicit speaker cues make them a test of basic behavior.'),
        txt("The real pilot exposes a different task. A sentence such as “The appeal was dismissed” can recount an earlier order or give the present disposition. Annotating the role of an isolated excerpt and annotating it while reading the entire case can produce different gold labels."),
        refs("LegalSeg paper","Gemini choice API","Gemini rate limits")))
    valrows=[]
    for k,a in (("A, public-file parent",first["base"]),("A, centroid edit",first["selected"]),("A, associative edit",second["confirmation"]),
                ("B, published parent",ctx["confirmation_parent"]),("B, associative edit",ctx["confirmation"])):
        valrows.append([k,pct(a["accuracy"]),f1(a["macro_f1"])])
    pages.append(page("What we wrote into the weights",
        "The two BERT parents have 109,487,623 parameters each. A complete FP32 checkpoint occupies 437,974,028 bytes and runs on a laptop CPU.",
        hdr("Direct changes to the classification head"),
        txt("For parent A, six authored examples per role define a contrast direction. We add a scaled direction to each output row and adjust its intercept using published class frequencies. The associative version adds a regularized, minimum-norm correction specified by 84 authored key–role pairs. Each final head contains 5,376 weights and seven biases."),
        txt("We also wrote a 50/50 midpoint of two published encoders. Comparing the endpoints and midpoint on validation favored original parent B. B was originally trained with surrounding sentences. Here B sees only the submitted excerpt. Its associative edit was selected for development accuracy, subject to preserving development macro-F1."),
        hdr("Disjoint validation judgments"),tbl(["Checkpoint","Accuracy","Macro-F1"],valrows,(.55,.22,.23)),
        txt("Each evaluation scores the same 1,120 held-out validation sentences from 371 cases, with class counts weighted to their original prevalence. B’s associative edit decreased confirmation accuracy despite a small gain on the selection split. The larger difference between A and B comes from the donor weights."),
        refs("Published weights","Clinical weight-space audit","Associative memory code")))
    if novel:
        spec=novel["frozen"];m=novel["confirmation"]
        pages.append(page("A seven-row instruction-model head",
            "Gemma 4 E2B runs locally. Its Q8 text backbone occupies 4,967,491,520 bytes. Seven rows of width 1,536 and seven biases add 10,759 task-specific parameters.",
            txt("The frozen instruction model reads the role definitions, then the user’s excerpt. At its final prompt token we take the 1,536-dimensional hidden state. Seven existing A–G vocabulary rows provide a control. Replacing the language-model output with the programmed matrix produces seven scores directly."),
            txt(f'The written head uses content-free bias coefficient {spec["selected_actual_edit"]["bias_coefficient"]} and associative damping {spec["selected_actual_edit"]["damping"]}. Five content-free inputs specify the bias. Where enabled, 21 authored keys specify a closed-form correction. The search used 56 development sentences. An additional 56 sentences come from a disjoint validation fold.'),
            tbl(["Readout on the same frozen decoder","Confirmation accuracy","Macro-F1"],[[k,pct(m[n]["accuracy"]),f1(m[n]["macro_f1"])] for k,n in (("Native quantized option logits","native"),("Copied A–G readout","unchanged"),("Programmed seven-class readout","edited"))],(.50,.30,.20)),
            txt(f'The copied float32 readout and the native quantized option scores chose the same role on {novel["matched_parity"]["same_native_argmax"]}/{novel["matched_parity"]["rows"]} pilot excerpts. Native matrix multiplication also quantizes activations. The paired-readout comparison isolates the effect of writing weights from the advantage of the larger instruction model.'),
            txt("The readout uses the same source text as Gemini and a fixed local rubric. The model runs without gradient steps or Gemini outputs. The separate 438 MB BERT developer path offers a smaller local option."),
            refs("Gemma 4 model card","Quantized Gemma weights","Content-free calibration","Associative memory paper")))
    pcases=para["cases"]
    pa=[]
    for unit,rec in (("Intact multi-statement row",para),("Joined same-role rows",joined)):
        if not rec:continue
        for k in mainkeys:
            if k in rec["metrics"]:
                m=rec["metrics"][k];pa.append([unit,NAMES[k],acc(m,rec["rows"]),f1(m["macro_f1"])])
    pages.append(page("One label for a paragraph",
        "Both paragraph tests use the seven original LegalSeg roles. The two evidence types have different annotation histories.",
        tbl(["Input and reference","System","Correct","Macro-F1"],pa,(.30,.30,.24,.16)),
        txt(f"The first set contains 21 source-labeled units from {pcases} judgments, three per role. Each contains multiple statements. Many qualifying rows concern sums of money or contain court enumeration, which limits generalization. We screened and masked identifying details before either model saw them."),
        txt("The second set contains two or three consecutive source rows from one judgment. Every constituent row has the same human role. Joining them provides a derived paragraph label. These easier, same-role passages cannot establish accuracy on mixed-role paragraphs with one adjudicated dominant label."),
        txt("At inference, long submissions use 510-token BERT windows with nominal overlap 128. The final window ends at the last token. Voting weights partition the content across windows so overlaps contribute once. On all 149,881 official test rows, 18 exceed one window."),
        refs("Official splits","LegalSeg paper")))
    if docrec:
        g=docrec["groups"]["test"];program=docrec["selection"]
        dr=[]
        for title,m in [*g["all_base_input_methods"].items(),("Selected sequence edit",g["programmed"]),
                        ("Selected edit, oracle row boundaries",g["programmed_with_reference_row_boundaries"])]:
            dr.append([title,pct(m["token_accuracy"]),f1(m["exact_role_boundaries"]["f1"]),f1(m["exact_labeled_spans"]["f1"])])
        pages.append(page("Where does each role begin and end?",
            "The span interface returns half-open character offsets and a role for every run of text. Adjacent units with the same predicted role form one span.",
            tbl(["Raw-passage method","Token accuracy","Transition F1","Exact labeled-span F1"],dr,(.42,.17,.17,.24)),
            txt(f'The held-out evaluation contains {g["cases"]} complete test judgments, {g["source_rows"]} labeled source rows and {g["programmed"]["tokens"]:,} whitespace-delimited tokens. Rejoining source rows with one ordinary space removes explicit row boundaries. PDF layout cannot be reconstructed from these CSVs.'),
            txt(f'The programmed decoder used {program["method"]}, diagonal transition weight {program["sticky"]}, directed-order weight {program["order"]} and neighboring-logit coefficient {program["neighbor"]}. The program wrote a 7 × 7 transition matrix and a three-value neighbor kernel. We selected among 48 prespecified combinations on ten development judgments, confirmed on ten separate judgments, then scored twenty untouched test judgments.'),
            txt("Exact labeled-span F1 requires the beginning, end and role to match. Boundary F1 scores interior role-change offsets without a role label. Token accuracy weights each non-whitespace token equally. The oracle row-boundary comparison isolates boundary-detection losses."),
            refs("LegalSeg paper","Official splits","Published weights")))
    if doccloud:
        t=doccloud;items=[]
        for k,title in (("local","Programmed local decoder"),("gemini","Gemini Decision 1")):
            m=t[k];items.append([title,pct(m["token_accuracy"]),f1(m["exact_role_boundaries"]["f1"]),f1(m["exact_labeled_spans"]["f1"])])
        pages.append(page("Gemini on the same unmarked passages",
            "A separate public passage check uses the same raw input bytes and the same automatically proposed units for both systems.",
            tbl(["System","Token accuracy","Transition F1","Exact labeled-span F1"],items,(.36,.20,.19,.25)),
            txt(f'The sample contains {t["cases"]} distinct judgments and {t["source_rows"]} consecutive gold source rows. The automatic splitter produced {t["proposed_units"]} units. Gemini and the local decoder chose the same role for {t["agreement_units"]} units. The shared raw text and gold-blind offsets are identical for the two systems.'),
            txt("The API takes typed choices, so each candidate unit receives a seven-role question. The input is the entire passage. The local sequence decoder scores those units and joins adjoining equal labels. Both are constrained by the set of proposed boundaries."),
            txt("Exact labeled-span F1 gap, local minus Gemini, has a 95% case-bootstrap interval of "+ci(t["paired_document_gap_95pct"][1])+" percentage points. The pilot contains passages with at least one gold role transition and cannot estimate prevalence across all judgments."),
            refs("Gemini choice API","Official splits")))
    classic=[["A author prediction file, all test rows","149,881","53.2%","0.491"],
             ["A author predictions on uniform test sample","1,792",pct(old["published_release"]["accuracy"]),f1(old["published_release"]["macro_f1"])],
             ["A reloaded on the same sample","1,792",pct(old["original_reloaded"]["accuracy"]),f1(old["original_reloaded"]["macro_f1"])],
             ["A centroid edit on the same sample","1,792",pct(old["edited"]["accuracy"]),f1(old["edited"]["macro_f1"])]]
    if full:
        measures=full["frozen_model"].get("product_full_excerpt",{})
        for k,n in (("original","A parent, all test rows"),("direct","A centroid edit, all test rows"),("associative_direct","A associative edit, all test rows")):
            if k in measures:
                m=measures[k]["metrics"];classic.append([n,"149,881",pct(m["accuracy"]),f1(m["macro_f1"])])
    pages.append(page("Where the official benchmark fits",
        "Accuracy measures exact gold-label matches. Macro-F1 averages class-wise F1 and gives rare roles equal influence. They can move in opposite directions.",
        tbl(["Standalone system","Rows","Accuracy","Macro-F1"],classic,(.46,.16,.20,.18)),
        txt("On the uniform test sample the centroid edit improved macro-F1 versus its loaded parent while lowering accuracy. The Hub assigns identical weight bytes to the standalone filename and the previous-gold-label filename. The reloaded model matched 68.5% of the authors’ separately published predictions in that sample."),
        txt("LegalSeg reports 0.77 macro-F1 for a hierarchical document model in Table 3. The named released predictions give 0.598 on 149,825 aligned rows. TransformerOverInLegalBERT gives 0.622, and an independent document-context Jev evaluation reports 0.520. Those methods receive surrounding material unavailable in a one-excerpt request."),
        txt("The source CSV and released standalone predictions align on all 149,881 test rows. The inference package records dataset revisions, parent and target hashes, model parameters, paired case counts and the public-file identity discrepancy."),
        refs("LegalSeg paper","Published weights","Official splits","Independent Jev evaluation")))
    pages.append(page("What the team can verify next",
        "The commercial decision is whether developers will trust a standalone label enough to use it in a search, navigation or document-review workflow.",
        hdr("The observable contract"),
        txt("Input is one nonempty English string. Output is one label among none, facts, issue, petitioner_argument, respondent_argument, reasoning and decision. Consider the paragraph “Notice was served on 3 May. We therefore allow the appeal.” The reference centroid checkpoint returns decision."),
        hdr("What would establish parity"),
        txt("Agree in advance on a tolerable accuracy gap, then compare the frozen model and Gemini against the same independently annotated excerpts. Preserve case boundaries. Report sentence and paragraph results separately, per-role recall, paired uncertainty, and local latency on a named laptop."),
        txt("Seven-role evidence comes from the official source annotations, from explicit runs of those annotations, and from separately marked synthetic diagnostics. For mixed-role text, retain the exact role ranges as the reference. The corpus rarely provides an independent human dominant label for an entire mixed paragraph."),
        hdr("What can be shared"),
        txt("The accompanying portable archive contains local inference code, programmed head weights, checksum-pinned reconstruction, source links and redacted benchmark predictions. It excludes raw LegalSeg CSVs. Developers can reproduce the local comparisons using their authorized copies."),
        txt("Gemini Decision 1 answers a typed choice question. The documented free tier lists 10,000 requests per project per day. The live comparisons used de-identified public passages or fictional text. Gemini supplies comparison labels. Every active weight edit was written and selected independently of those API outputs."),
        refs("Gemini choice API","Gemini rate limits","Horizon Labs","Associative memory paper")))
    from extra_brief import add
    from stocktake_report import assemble
    return assemble(add(pages))
