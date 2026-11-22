"""Prepare candidates for a local privacy review before a paired, held-out Gemini benchmark."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from paragraph_transfer import read
from program_trial import TO_ID,save_json
from weight_edit import LABELS,ROOT


DEST=ROOT/"gemini-benchmark"
ORDINAL=re.compile(r"^\s*(?:\(?\d{1,3}[.)]\s*|[ivxlcdm]+[.)]\s+)",re.I)
TITLE=re.compile(r"\b(?:Mr|Mrs|Ms|Messrs|Shri|Smt|Sir|Dr|Justice|village|address|telephone|email|named|resident|aged|diagnos\w*|abuse|sexual|medical|murder|rape|suicide|deceased|minor child|witness named|birth|hospital|criminal offence|blood|disease)\b",re.I)
CAPS=set("""A An The This That These Those It Its In On At As For From To Of By With If Where When Whether What Why How Which Who Whose While Although No Not Nor Before After Under Upon Since Further Besides Moreover Hence Thus Therefore Accordingly However Consequently Firstly Secondly Finally Mridha We Our There They He She His Her Their One Two Having Being Learned Counsel So Invoking Perusal Here Such Suchlike Admittedly Indeed Now Once Section Sections Article Articles Act Rules Rule Order Orders Civil Code Constitution Parliament Government Central State Union India Indian High Supreme Court Courts Bench Division Single Trial Tribunal Appellate Authority Assessing Officer Government Appeal Appellant Appellants Respondent Respondents Petitioner Petitioners Plaintiff Plaintiffs Defendant Defendants Application Petition Judgment Judgement Registrar Registry Advocate General Counsel Senior Commissioner Board Collector Revenue Arbitration Contract Evidence Procedure Limitation Rent Tax Sales Customs Income Service Industrial Labour Land Acquisition Motor Vehicles Sessions Sub Judge District Magistrate Bar Council First Second Issue Issues Decision Facts Ruling Discussion Analysis Arguments Attracting According Both Attention Thereafter Even Another Then All Only Next Apart Nevertheless Aforesaid Since Under Reading Submissions Solicitor Additional Sessions Re Special Leave Civil Appeal Appeals Petition Petitions Other Others Reserved Coram J JJ OR AND THE OF IN COURT SUPREME INDIA CIVIL APPELLATE JURISDICTION ORDER REPORTABLE NON ARBITRATION CONSTITUTION Question Proceedings Condonation Dismissed Allowed Against Consequential Considering Despite Irrespective Normally Ordinarily Strictly Regarding Sometimes Taking Turning Reverting Understood Nothing Suffice Undoubtedly Yet""".split())
CAPS.discard("Mridha")


def allow(text,paragraph=False):
    cleaned=ORDINAL.sub("",text.strip())
    cleaned=" ".join(cleaned.split())
    words=cleaned.split()
    if not (8 if not paragraph else 12)<=len(words)<=(62 if not paragraph else 210):
        return None
    if re.search(r"[0-9@/\\]|https?\b",cleaned,re.I) or TITLE.search(cleaned):
        return None
    capitals=re.findall(r"\b[A-Z][A-Za-z'-]*",cleaned)
    if any(token not in CAPS for token in capitals):
        return None
    return cleaned


def item(source,idx,doc,text,role):
    return {"id":source+"-"+str(idx),"source":source,"doc":doc,"text":text,"gold":role,
            "sha256":hashlib.sha256(text.encode()).hexdigest()}


def main():
    DEST.mkdir(exist_ok=True)
    per=defaultdict(list);raw={}
    with (ROOT/"references/official/test.csv").open(encoding="utf-8-sig",newline="") as f:
        for i,row in enumerate(csv.DictReader(f)):
            t=allow(row["Text"])
            if t:per[LABELS[TO_ID[row["Label"]]]].append(item("sentence",i,row["Index"],t,LABELS[TO_ID[row["Label"]]]))
    rng=random.Random(2026112193);audit=[];seen=set()
    for role,group in sorted(per.items()):
        rng.shuffle(group);used=set();entries=[]
        for row in group:
            if row["doc"] in used or row["sha256"] in seen:continue
            entries.append(row);used.add(row["doc"]);seen.add(row["sha256"])
            if len(entries)==28:break
        raw[role]=len(group);audit.extend(entries)
    source=read("PRR.zip")
    matched=json.loads((ROOT/"results/prr-paragraph-transfer.json").read_text())["cases_with_matching_spans"]
    paragraphs=defaultdict(list)
    for case,idx,text,label in source:
        if case in matched:continue
        t=allow(text,True)
        if t:paragraphs[label].append(item("paragraph",f"{case}-{idx}",case,t,label))
    for role,group in sorted(paragraphs.items()):
        rng.shuffle(group);audit.extend(group[:20]);raw["PRR "+role]=len(group)
    save_json(DEST/"candidates.json",{"eligible":raw,"candidates":audit,
              "policy":"Manually review privacy and text integrity before transmitting to the official Gemini endpoint. Gold labels never appear in model inputs. Selection is independent of both models' predictions. All sentence texts come from the test split; paragraph candidates come from cases with no exact substantive training match."})
    print("Eligibility and review pool",raw,Counter((r['source'],r['gold']) for r in audit),flush=True)


if __name__=="__main__":main()
