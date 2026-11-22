"""Find original seven-role LegalSeg test rows with internal sentence boundaries."""
import csv
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
ROOT=Path(__file__).resolve().parent
DEST=ROOT/"gemini-seven-paragraphs"
ROLES={"None":"none","Facts":"facts","Issue":"issue","Arguments of Petitioner":"petitioner_argument",
       "Arguments of Respondent":"respondent_argument","Reasoning":"reasoning","Decision":"decision"}
BOUNDARY=re.compile(r'(?<=[.?!])\s+(?=[“"‘(]?[A-Z])')
ABBR=set("vs v ltd co inc corpn anr ors dr sri shri smt mr mrs ms no nos sr jr ltd ors anr coram hon pvt st honble honbl etc viz dept s para paras fig cl art sec secs reg adv ex exh rs cr supp edn ed pp vol jl jj j ca ia cp w p aa scc lab atc itr air scr jr slp govt prof pn ro llp misc wp cri wpc cm crl sc hc ors".split())
SENSITIVE=re.compile(r"\b(?:wife|husband|divorc\w*|daughter|son|mother|father|marriage|murder\w*|molest\w*|rap(?:e|ist)\w*|illness|injur\w*|convict\w*|sex\w*|assault\w*|shoot\w*|kill\w*|blood\w*|physio\w*|illtreat\w*|famili\w*|hospital|suici\w*|weapon\w*|child\w*|stab\w*|diagnos\w*|bail|pregnan\w*|kidnap\w*)\b",re.I)
SAFE_CAPS=set("""A An The This That These Those It Its In On At As For From To Of By With If Where When Whether What Why How Which Who Whose While Although No Not Nor Before After Under Upon Since Further Besides Moreover Hence Thus Therefore Accordingly However Consequently Firstly Secondly Finally We Our There They He She His Her Their One Two Having Being Learned Counsel So Invoking Perusal Here Such Suchlike Admittedly Indeed Now Once Section Sections Article Articles Act Rules Rule Order Orders Civil Code Constitution Parliament Government Central State Union India Indian High Supreme Court Courts Bench Division Single Trial Tribunal Appellate Authority Assessing Officer Government Appeal Appellant Appellants Respondent Respondents Petitioner Petitioners Plaintiff Plaintiffs Defendant Defendants Application Petition Judgment Judgement Registrar Registry Advocate General Counsel Senior Commissioner Board Collector Revenue Arbitration Contract Evidence Procedure Limitation Rent Tax Sales Customs Income Service Industrial Labour Land Acquisition Motor Vehicles Sessions Sub Judge District Magistrate Bar Council First Second Issue Issues Decision Facts Ruling Discussion Analysis Arguments Attracting According Both Attention Thereafter Even Another Then All Only Next Apart Nevertheless Aforesaid Since Under Reading Submissions Solicitor Additional Sessions Re Special Leave Civil Appeal Appeals Petition Petitions Other Others Reserved Coram J JJ OR AND THE OF IN COURT SUPREME INDIA CIVIL APPELLATE JURISDICTION ORDER REPORTABLE NON ARBITRATION CONSTITUTION Question Proceedings Condonation Dismissed Allowed Against Consequential Considering Despite Irrespective Normally Ordinarily Strictly Regarding Sometimes Taking Turning Reverting Understood Nothing Suffice Undoubtedly Yet No Ltd v Vs Pvt AIR SCC SCR No Nos No Ltd Company M/s Taxes Manager Deputy Secretary Services Municipal Public Writ Interim Board Taxation Bombay Delhi Madras Calcutta Corporation Companies Chapter Act Part CPC CrPC Consumer Chief Gazette CPC LLP Rules Bank Ltd Limited Regional Authority Director Schedule Regulation Regulations Development Excise Constitution India Mumbai Kerala Gujarat Karnataka Maharashtra Bihar Pradesh Pradesh Uttar West Bengal Punjab Chennai Tahsildar Even Quoting First Schedule Learned Merely Electricity Arbitration Honble Resolutions Finance Committee Public Works Department India Central Central State Schools University Lokayukta Provident Constitution Parliament Constitution Amendment Income Tax Section Criminal Criminal Procedure Code Supreme AIR SCC SCR""".split())
def boundaries(text):
    good=[]
    for m in BOUNDARY.finditer(text):
        left=text[:m.start()];last=re.search(r"([A-Za-z.]+)[.?!]$",left)
        if last and (last.group(1).lower() in ABBR or len(last.group(1))<2 or '.' in last.group(1)):continue
        if len(left.split())<4 or len(text[m.end():].split())<4:continue
        good.append(m.start())
    return good
def dump(p,obj):
    tmp=p.with_suffix(".partial");tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def main():
    DEST.mkdir(exist_ok=True)
    prior=set()
    for folder in ("gemini-benchmark","gemini-confirmation"):
        prior|={r["doc"] for r in json.loads((ROOT/folder/"approved.json").read_text()) if r["source"]=="sentence"}
    pool=defaultdict(list);eligible=Counter()
    with (ROOT/"references/official/test.csv").open(encoding="utf-8-sig",newline="") as f:
        for index,r in enumerate(csv.DictReader(f)):
            t=" ".join((r["Text"] or "").split());k=ROLES[r["Label"]]
            if not (12<=len(t.split())<=240) or not boundaries(t):continue
            eligible[k]+=1
            if SENSITIVE.search(t) or re.search(r"[@\\]|https?\b|www\.",t,re.I):continue
            unknown=[v for v in re.findall(r"\b[A-Z][A-Za-z'-]*",t) if v not in SAFE_CAPS]
            pool[k].append({"id":"multisentence-"+str(index),"source":"multi_sentence","doc":r["Index"],"text":t,
                "gold":k,"unknown":len(set(unknown)),"shared_case":r["Index"] in prior})
    rng=random.Random(20261122031);ordered={}
    for k,group in pool.items():
        rng.shuffle(group);group.sort(key=lambda x:(x["unknown"],x["shared_case"]))
        ordered[k]=group
    dump(DEST/"candidates.json",{"boundary_candidates":eligible,"ranked":ordered})
    print('source units with candidate boundaries',eligible)
    print('after sensitivity screen by role',[(k,len(a),sum(r['unknown']==0 for r in a)) for k,a in ordered.items()])
if __name__=="__main__":main()
