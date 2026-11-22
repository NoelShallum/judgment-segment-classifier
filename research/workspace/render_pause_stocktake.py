"""Render the AttoFlow research pause stocktake from completed evidence."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[1]/"outputs/judgment-role/AttoFlow-research-stocktake.pdf"

def read(stem):
    return json.loads((ROOT/"results"/(stem+".json")).read_text())

wide=read("sentence-wide-bert-benchmark")["splits"]["test"]
budget=read("sentence-wide-bert-budget")
assert wide["cases"]==712 and wide["baseline"]["rows"]==149868
assert "test" in budget["splits"]
confidence=wide["comparison"]["routes"]["confidence"]
local=read("sentence-template-memory-bert")["splits"]["test"]["candidate"]
archive=read("sentence-joint-pairwise-span-archive")
raw=read("segment-weight-qa")["scores"]
qa=read("sentence-wide-bert-cli-qa")
hold_blocks=sorted((ROOT/"cache/sentence-wide-bert-benchmark").glob("holdout-*/*.npy"))
future_blocks=sorted((ROOT/"cache/sentence-bert-future-context").glob("fit-*/*.npy"))
import numpy as np
hold_count=sum(np.load(f,mmap_mode="r").shape[0] for f in hold_blocks)
future_count=sum(np.load(f,mmap_mode="r").shape[0] for f in future_blocks)

fontdir=Path("/usr/share/fonts/truetype/ibm-plex")
for name,file in (("Plex","IBMPlexSans-Regular.ttf"),("PlexBold","IBMPlexSans-SemiBold.ttf"),
                  ("PlexSerif","IBMPlexSerif-Regular.ttf")):
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily("Plex",normal="Plex",bold="PlexBold")
ink=colors.HexColor("#182B34")
mute=colors.HexColor("#4A6065")
green=colors.HexColor("#106558")
line=colors.HexColor("#D8E4DF")
pale=colors.HexColor("#EEF5F0")
styles={
    "title":ParagraphStyle("title",fontName="PlexSerif",fontSize=31,leading=35,textColor=ink,spaceAfter=11),
    "page":ParagraphStyle("page",fontName="PlexSerif",fontSize=25,leading=29,textColor=ink,spaceAfter=9),
    "deck":ParagraphStyle("deck",fontName="Plex",fontSize=11.2,leading=15.4,textColor=mute,spaceAfter=13),
    "h":ParagraphStyle("h",fontName="PlexBold",fontSize=13.0,leading=16,textColor=green,spaceBefore=12,spaceAfter=5),
    "body":ParagraphStyle("body",fontName="Plex",fontSize=10.35,leading=14.35,textColor=ink,spaceAfter=8),
    "small":ParagraphStyle("small",fontName="Plex",fontSize=8.7,leading=11.8,textColor=mute,spaceAfter=7),
    "cell":ParagraphStyle("cell",fontName="Plex",fontSize=9.15,leading=12.05,textColor=ink),
    "head":ParagraphStyle("head",fontName="PlexBold",fontSize=9,leading=11.4,textColor=green),
    "metric":ParagraphStyle("metric",fontName="PlexSerif",fontSize=22,leading=25,textColor=green),
}
story=[]
W=485
def p(text,style="body"):
    story.append(Paragraph(text,styles[style]))
def heading(text):
    p(text,"h")
def page(title,deck=None):
    story.append(PageBreak())
    p(title,"page")
    if deck:
        p(deck,"deck")
def link(label,url):
    return f'<link href="{escape(url)}" color="#106558">{escape(label)}</link>'
def sources(*items):
    p("  ·  ".join(link(*x) for x in items),"small")
def table(headers,rows,widths):
    data=[[Paragraph(escape(str(x)),styles["head"]) for x in headers]]
    data.extend([Paragraph(escape(str(x)),styles["cell"]) for x in row] for row in rows)
    t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),pale),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LINEBELOW",(0,0),(-1,-1),.45,line)]))
    story.extend([t,Spacer(1,8)])

legal=("LegalSeg paper and evaluation protocol","https://aclanthology.org/2025.findings-naacl.63/")
preds=("LegalSeg released predictions","https://huggingface.co/collections/L-NLProc/legalseg-models")
gemini=("Google Decisions launch","https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/")
terms=("Gemini API terms","https://ai.google.dev/gemini-api/terms")

p("Open model research<br/>Progress at the pause","title")
p("AttoFlow • 22 November 2026","deck")
p("The best-supported legal use case is <b>labeling sentences with the full judgment available</b>. The strongest matched improvement is exact span reconstruction with the same supplied sentence cuts. The official sentence-label benchmark and Gemini parity remain separate open targets.")
cards=Table([[Paragraph(a,styles["metric"]) for a in ("0.64392","0.26931","0.770")],
             [Paragraph(x,styles["cell"]) for x in ("Runnable sentence macro-F1","Exact span F1 with supplied cuts","Published LegalSeg macro-F1")]],colWidths=[W/3]*3)
cards.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),pale),("VALIGN",(0,0),(-1,-1),"TOP"),
    ("LEFTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,0),10),
    ("BOTTOMPADDING",(0,1),(-1,1),11)]))
story.extend([cards,Spacer(1,11)])
heading("What the final official test established")
table(("Profile on 712 judgments","Macro-F1","Accuracy","Reviewer windows"),[
    ("Sequence model with role-pair weights","0.64270","65.86%","—"),
    ("Earlier InLegalBERT plus exact-template memory",f'{local["macro_f1"]:.5f}',f'{100*local["accuracy"]:.2f}%',"7,482"),
    ("Confidence route, 20% budget",f'{confidence["labels"]["macro_f1"]:.5f}',f'{100*confidence["labels"]["accuracy"]:.2f}%',"30,252"),
    ("Confidence route, 15% development choice","0.64391","65.97%","22,818"),
    ("Utility route, 20% budget","0.64287","65.94%","30,252"),
],(235,72,71,107))
p("The 20% confidence route gains 0.121 macro-F1 points over the sequence model. Its paired 95% interval is +0.007 to +0.247 points. Its point estimate sits just 0.0029 points above the earlier, lower-cost model. The official confidence replay at 15% uses 7,434 fewer review windows, with a paired difference from 20% of -0.042 to +0.040 points.")
p("Scope: each score uses the same 149,868 nonblank test sentences. The 15% policy was selected on development. The 10% replay also scored 0.64393, an exploratory point estimate across tested budgets. The wider review was measured with FP32 GPU features. CPU parity covers the checked judgments on page 2.","small")
sources(legal)

page("Inputs, outputs and deployment","Seven mutually exclusive roles apply whenever the caller requests a single label.")
p("<b>Taxonomy.</b> None, facts, issue, petitioner argument, respondent argument, reasoning and decision. For a single sentence or paragraph, the contract returns one role from the full taxonomy. For a judgment or mixed passage, the segmentation interface returns adjacent spans with one role per span.")
table(("Input","Output","Measured result"),[
    ("Full judgment with supplied sentences","One role per sentence","0.64392 macro-F1 and 66.00% accuracy on 712 official judgments."),
    ("Full judgment with supplied sentences","Contiguous runs with sentence offsets","0.26931 exact span F1 versus 0.23371 for the strongest aligned author projection. Paired gain +2.79 to +4.37 points."),
    ("Raw full judgment","Role and half-open Unicode character offsets","0.30401 exact span F1 and 51.52% token accuracy for the segment-exact profile. The raw-joint profile reaches 63.75% token accuracy and 0.2409 exact span F1."),
    ("Clipped passages","One role per labeled span","MiniLM contextualized role-key trial reached 0.261 exact span F1 on 120 passages. The original edited profile scored 0.235."),
    ("Isolated sentence","One role","The isolated hierarchy experiment reached 0.40580 macro-F1 versus 0.49112 for the aligned author sentence-only predictions."),
    ("Standalone paragraph","One role","The local and Gemini systems each got 8 of 21 correct on the constructed paragraph subset. The official CSV supplies sentence rows without original paragraph boundaries."),
],(128,127,230))
heading("What fits on a laptop")
p("The document hierarchy has <b>16.82M parameters</b>. It reads sentence vectors as a sequence, with practical limits set by runtime and memory. The frozen <b>109.49M-parameter InLegalBERT</b> reviewer supports 512 WordPiece positions, including special tokens. The benchmark uses 96 per review window, covering the target and up to two previous sentences.")
p("The FP32 CPU command matched every label, span and referral position across three selected confirmation judgments with 569 sentences and 115 review windows. A raw-text fixture also verified ordered character spans. The two INT8 checks changed 148 final predictions in total while compressing the graph from 438 MB to 182 MB. FP32 remains the parity reference.")
sources(("LegalSeg CSV releases","https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV"),preds)

page("What we edited","The legal experiments wrote weights, calibration matrices or lookup values without gradient-based fine-tuning.")
table(("Experiment family","What was tried","What the evidence says"),[
    ("Sequence and span decoding","49 transition weights, role and length costs, neighboring-role transitions, boundary lattices, path reranking and joint role decisions.","The original transition edit moved full-judgment exact span F1 from 0.198 to 0.257. Segment-exact reached 0.30401. Prioritizing exact spans trades off token accuracy."),
    ("Document readouts and internal weights","Closed-form lexical and semantic heads, pair-specific boundaries, graph features, kernels, recurrent-gate and normalization edits, protected subspaces and refitted heads.","Direct role-pair evidence and sequence decoding produced the strongest broadly reproducible document base. Recurrent and normalization edits showed mixed or narrow gains."),
    ("Associative and retrieval memory","Authored role keys, class prototypes, sparse training-example memory, exact templates and 32,768 fuzzy character n-gram examples.","Exact templates gave small gains in combinations. Fuzzy none/reasoning corrected three development labels. Confirmation corrected one and introduced two."),
    ("Whole-judgment calibration","Role frequency corrections over the whole judgment and five relative-position blocks. Seven-output fusion used four-fold BERT predictions from 144 training judgments.","The development policy retained the existing model for both trials. Wider confidence routing modestly improved the sequence baseline."),
    ("Alternative backbones","MiniLM (~23M), LEGAL-BERT Small (~35M), Granite R2 Small (47M), DeBERTa-v3-xsmall (~71M), Julia 1 (~144M), GTE ModernBERT Base (149M), EmbeddingGemma (270M) and Qwen3-0.6B.","Most were pilots or referral comparisons. None displaced the leading full-judgment profile in a matched official evaluation. MiniLM was useful on clipped passages."),
],(111,190,184))
heading("Research precedents")
p("Horizon Labs supplied the product hypothesis of small models with specific, measurable uses. Ok and Lee motivated sparse, directly writable memory. Nusrat and Nusrat supplied evidence for localized repairs in small networks. Legal rhetorical-role studies motivated global context and explicit role boundaries.")
sources(("Horizon report","https://horizon-labs.ai/"),("Ok and Lee","https://www.alphaxiv.org/abs/2611.weight-space-programming-associative-memory"),
        ("Nusrat and Nusrat","https://arxiv.org/abs/2611.05341"),("Belfathi et al.","https://aclanthology.org/2026.eacl-long.137/"))

page("Where the evidence stops","Official sentence labels, exact spans and hosted API comparisons answer different questions.")
heading("Published and released comparisons")
table(("Reference","Result","Interpretation"),[
    ("LegalSeg Table 3","0.770 macro-F1","Published hierarchical BiLSTM-CRF sentence result. The public files we aligned did not reproduce the headline."),
    ("Strongest aligned LegalSeg author archive","0.6329 macro-F1","Previous predicted role. The matched exact-span projection scores 0.23371."),
    ("AttoFlow replay using author predictions","0.65035 macro-F1","Composition with saved predictions. The corresponding reviewer checkpoint was unavailable as a distinct public download."),
    ("SeqTNS and HiCoBERT","0.83 and 0.5044 macro-F1","SeqTNS used a different seven-role corpus and taxonomy. HiCoBERT mapped LegalSeg to four roles and evaluated 679 judgments."),
],(163,104,218))
p("The public files named for standalone and previous-true-label InLegalBERT were byte-identical. The provenance discrepancy remains open. Repeated rounds of research have inspected aggregate results from the same 712 judgments. A fresh labeled cohort would give a stronger replication of tiny improvements.")
sources(legal,preds,("SeqTNS","https://aclanthology.org/2025.findings-ijcnlp.50/"),("HiCoBERT","https://pmc.ncbi.nlm.nih.gov/articles/PMC13056923/"))
heading("What limits the current score")
p("On the earlier runnable model, <b>none and reasoning account for 25,507 of 51,051 errors</b>. A perfect correction of those confusions would yield 0.72448 macro-F1. Adding every facts/issue correction would yield 0.76630. Argument-role confusions rank next by isolated impact. These are calculations using gold labels, with every other prediction held fixed.")
heading("Gemini Decision 1")
p("On the same 12 screened passages and 64 candidate units, Gemini scored 0.160 exact span F1 and 61.9% token accuracy. The strongest local matched profile scored 0.125 and 46.9%. The paired span interval crosses zero. Full-judgment parity and broad standalone paragraph parity remain unmeasured.")
p("Gemini can serve as a hosted fixed-choice benchmark for the exact seven-role rubric. Google lists 10,000 requests per project per day on the free tier, with 200 requests and 50,000 input tokens per minute. Google’s terms restrict competitive model development and extraction. Distillation for release remains a separate question for the applicable agreement.")
sources(gemini,("Google rate limits","https://ai.google.dev/gemini-api/docs/rate-limits"),terms)

page("Portfolio and pause checkpoint","The portfolio goal is an open model developers adopt, with public or synthetic data and practical local inference.")
table(("Candidate","Input → output","Status at the pause"),[
    ("Catalog value normalizer","Attribute, raw value, product context and allowed set → canonical ID or unknown.","Leading non-legal candidate. A cross-merchant alias baseline scored 59.1% on 3,488 WDC-PAVE training-file examples under merchant exclusions. The official test remained untouched."),
    ("Judgment rhetorical roles","Sentence, paragraph or entire judgment → one role per requested unit, or labeled spans.","Most extensively benchmarked candidate. Strongest evidence is for full-judgment context and supplied sentence cuts."),
    ("Repository issue labeler","Issue title/body, repository labels and descriptions → zero or more label IDs.","Kept in the product comparison. Existing regex actions and personalized labelers define its baseline. This stocktake records no new trained model."),
],(129,152,204))
p("Isaacus was the legal API and model reference. Microsoft-originated Presidio was the privacy and redaction reference. Trace integration was optional, and the first release could remain independent. The prior portfolio report also covers privacy, support, finance and timekeeping opportunities.")
sources(("Isaacus classification","https://isaacus.com/products/classification"),("Microsoft Presidio","https://github.com/microsoft/presidio"))
heading("Saved at the stop")
table(("Workstream","Checkpoint"),[
    ("Official wider FP32 and budget replay","Complete across all 712 judgments. Receipts are included in the experiment bundle."),
    ("384-judgment wider FP32 holdout",f"Paused with {hold_count:,} of 17,934 new windows cached. No holdout score was issued."),
    ("Following-context reviewer",f"Paused with {future_count:,} of 1,476 fit windows cached. No selection or confirmation score was issued."),
    ("Wide review plus exact templates","Development and confirmation complete. The official composition job was canceled while waiting for the budget replay."),
],(165,320))
heading("Companion files")
p("<b>weight-editing-research.html</b> contains the full method and benchmark log. <b>AttoFlow-judgment-span-edits.zip</b> contains the pinned receipts, patches and reproduction scripts. <b>AttoFlow-open-model-research.pdf</b> and <b>AttoFlow-model-catalog.xlsx</b> contain the broader product comparison.","small")
p("The research goal is paused. The official LegalSeg headline, broad Gemini parity and a release decision remain open.","small")

def chrome(canvas,doc):
    aw,ah=A4
    canvas.setStrokeColor(line)
    canvas.line(55,ah-50,aw-55,ah-50)
    canvas.line(55,43,aw-55,43)
    canvas.setFillColor(green)
    canvas.setFont("PlexBold",10)
    canvas.drawString(55,ah-39,"ATTOFLOW")
    canvas.setFillColor(mute)
    canvas.setFont("Plex",8.6)
    canvas.drawRightString(aw-55,ah-39,"RESEARCH STOCKTAKE")
    canvas.drawString(55,29,"Open models and weight-space programming")
    canvas.drawRightString(aw-55,29,str(doc.page))

doc=SimpleDocTemplate(str(DEST),pagesize=A4,leftMargin=55,rightMargin=55,topMargin=67,bottomMargin=56,
                      title="AttoFlow research stocktake",author="AttoFlow research")
doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
