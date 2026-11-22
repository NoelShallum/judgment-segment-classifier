"""Query the matched, privacy-reviewed holdout with a fixed choice question."""
import argparse
import datetime
import getpass
import hashlib
import json
import os
import time
import requests
from seven_paths import FOLDERS,folder_for
from seven_paragraphs import dump as save_json
from synthetic_review import DEFINITIONS,request_review,role_prompt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--live",action="store_true",required=True)
    ap.add_argument("--prompt-key",action="store_true")
    ap.add_argument("--corpus",choices=tuple(FOLDERS),default="pilot")
    args=ap.parse_args()
    folder=folder_for(args.corpus)
    rows=json.loads((folder/"approved.json").read_text())
    schema={"model":"gemini-decision-1","instructions":role_prompt(False),"criteria":DEFINITIONS,"type":"choice"}
    signature=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    save_json(folder/"request-schema.json",{"sha256":signature,**schema})
    key=getpass.getpass("Google project key: ") if args.prompt_key else os.environ["GEMINI_API_KEY"]
    dest=folder/"answers.json"
    answers=json.loads(dest.read_text()) if dest.is_file() else {}
    session=requests.Session()
    for i,r in enumerate(rows):
        if r["id"] in answers:
            old=answers[r["id"]]
            if old["input_sha256"]!=r["sha256"] or old["question_sha256"]!=signature:raise ValueError("Input or instruction changed")
            continue
        assert hashlib.sha256(r["text"].encode()).hexdigest()==r["sha256"]
        pred=request_review(session,key,r["text"],fictional=False)
        answers[r["id"]]={"input_sha256":r["sha256"],"question_sha256":signature,"model":schema["model"],"choice":pred,
            "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save_json(dest,answers)
        if (i+1)%12==0 or i+1==len(rows):print("Completed",i+1,"of",len(rows),flush=True)
        time.sleep(.32)
if __name__=="__main__":main()
