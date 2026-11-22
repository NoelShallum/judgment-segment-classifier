"""Run the local seven-row Gemma readout through a local llama.cpp embeddings server."""
import argparse,hashlib,json,sys
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import requests
from safetensors.numpy import load_file
from windowing import content_windows
class InstructionRoleClassifier:
    def __init__(self,profile_dir=None,server="http://127.0.0.1:8080"):
        self.root=Path(profile_dir) if profile_dir else Path(__file__).resolve().parent
        if urlparse(server).hostname not in ("127.0.0.1","localhost","::1"):raise ValueError("Use a loopback llama.cpp embeddings server.")
        self.url=server.rstrip("/");self.s=requests.Session();self.spec=json.loads((self.root/"profile.json").read_text())
        w=self.root/"edited.safetensors"
        if hashlib.sha256(w.read_bytes()).hexdigest()!=self.spec["head_sha256"]:raise ValueError("Unexpected head checksum.")
        self.head=load_file(str(w));self.pre=self._tokens((self.root/"prefix.txt").read_text(),True,True);self.post=self._tokens((self.root/"suffix.txt").read_text(),True,False)
    def _tokens(self,txt,special,bos):
        resp=self.s.post(self.url+"/tokenize",json={"content":txt,"parse_special":special,"add_special":bos},timeout=25);resp.raise_for_status();return resp.json()["tokens"]
    def predict(self,text):
        if not isinstance(text,str) or not text.strip():raise ValueError("Supply a nonempty English sentence or paragraph.")
        ids=self._tokens(text,False,False);cap=self.spec["context_length"]-len(self.pre)-len(self.post)
        score=np.zeros(len(self.spec["labels"]),dtype=np.float64)
        for a,b,mass in content_windows(len(ids),cap):
            req={"content":self.pre+ids[a:b]+self.post,"embd_normalize":-1}
            resp=self.s.post(self.url+"/embeddings",json=req,timeout=300);resp.raise_for_status()
            x=np.asarray(resp.json()[0]["embedding"],dtype=np.float32).reshape(-1)
            score+=mass*(x@self.head["classifier.weight"].T+self.head["classifier.bias"])
        return {"label":self.spec["labels"][int(score.argmax())]}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("text",nargs="?");ap.add_argument("--server",default="http://127.0.0.1:8080")
    ap.add_argument("--profile-dir",type=Path,default=Path(__file__).resolve().parent);a=ap.parse_args()
    print(json.dumps(InstructionRoleClassifier(a.profile_dir,a.server).predict(a.text if a.text is not None else sys.stdin.read())))
if __name__=="__main__":main()
