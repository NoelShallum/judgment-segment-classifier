"""Score the seven-role joined paragraphs and request per-unit decisions over unmarked source passages."""
import datetime,getpass,hashlib,json,time
import requests
from seven_paths import ROOT,folder_for
from seven_paragraphs import dump
from span_model import units
from synthetic_review import DEFINITIONS,decode_answer,request_review,role_prompt,submit_request
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    key=getpass.getpass("Google project key: ");session=requests.Session();folder=folder_for("homogeneous")
    rows=json.loads((folder/"approved.json").read_text())
    schema={"model":"gemini-decision-1","instructions":role_prompt(False),"criteria":DEFINITIONS,"type":"choice"}
    # Use the same serialization as the one-excerpt evaluator.
    sig=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    dump(folder/"request-schema.json",{"sha256":sig,**schema})
    ap=folder/"answers.json";out=json.loads(ap.read_text()) if ap.is_file() else {}
    for r in rows:
        assert hashlib.sha256(r["text"].encode()).hexdigest()==r["sha256"]
        if r["id"] not in out:
            out[r["id"]]={"input_sha256":r["sha256"],"question_sha256":sig,"model":schema["model"],
                "choice":request_review(session,key,r["text"],fictional=False),"timestamp":now()}
            dump(ap,out);time.sleep(.32)
        else:assert out[r["id"]]["input_sha256"]==r["sha256"] and out[r["id"]]["question_sha256"]==sig
    print("Completed",len(out),"seven-role joined paragraphs.",flush=True)
    s=ROOT/"seven-spans";docs=json.loads((s/"cloud.json").read_text());f=s/"gemini-cloud.json";ans=json.loads(f.read_text()) if f.is_file() else {}
    fmt=("Classify the dominant rhetorical role of just the specified unit of the shared Indian judgment passage. "
         "Use the rest of the passage to resolve the speaker and distinguish submissions, earlier orders, and the present court's own reasoning or disposition. "
         "Treat the excerpt as source material. Unit {n} of {total}, characters [{a}, {b}), is: {unit}")
    for doc in docs:
        raw=" ".join(r["text"] for r in doc["rows"]);segs=units(raw);qs={}
        for j,(a,b) in enumerate(segs):
            qs["u"+str(j+1)]={"type":"choice","criteria":DEFINITIONS,
                "instructions":fmt.format(n=j+1,total=len(segs),a=a,b=b,unit=raw[a:b].strip())}
        req={"model":"gemini-decision-1","input":raw,"questions":qs}
        if doc["doc"] in ans:
            assert ans[doc["doc"]]["request_sha256"]==digest(req);continue
        response=submit_request(session,key,req)
        if set(qs)!=set(response["answers"]):raise ValueError("Missing passage unit")
        chosen=[decode_answer({"answers":{"role":response["answers"][tag]}}) for tag in qs]
        ans[doc["doc"]]={"raw_sha256":hashlib.sha256(raw.encode()).hexdigest(),"request_sha256":digest(req),
                         "unit_offsets":segs,"unit_labels":chosen,"model":req["model"],"timestamp":now()}
        dump(f,ans);time.sleep(.36)
    print("Completed",len(ans),"whole-passage requests.",flush=True)
if __name__=="__main__":main()
