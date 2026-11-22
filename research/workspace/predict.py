"""Return one dominant seven-way role or the labeled character spans of a longer judgment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from windowing import token_windows

class JudgmentRoleClassifier:
    def __init__(self,model_dir=None,device="cpu"):
        self.folder=Path(model_dir) if model_dir else Path(__file__).resolve().parent
        self.tokenizer=AutoTokenizer.from_pretrained(self.folder,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(self.folder,local_files_only=True).to(device).eval()
        self.device=device
        self.token_limit=min(self.model.config.max_position_embeddings,self.tokenizer.model_max_length)
        self._sequence=None

    @staticmethod
    def _validated(text):
        if not isinstance(text,str) or not text.strip():raise ValueError("Supply a nonempty English judgment excerpt.")
        return text

    @torch.inference_mode()
    def _scores(self,texts):
        # Each submitted excerpt has its own complete coverage, even when batched with others.
        encoded=self.tokenizer(list(texts),add_special_tokens=False,truncation=False,return_attention_mask=False,
            return_token_type_ids=False,verbose=False)["input_ids"]
        entries=[(i,ids,weight) for i,tokens in enumerate(encoded)
                 for ids,weight in token_windows(self.tokenizer,tokens,self.token_limit)]
        entries.sort(key=lambda v:len(v[1]))
        total=np.zeros((len(texts),self.model.config.num_labels),dtype=np.float64)
        mass=np.zeros(len(texts),dtype=np.float64)
        start=0
        while start<len(entries):
            size=min(16,max(1,1000//len(entries[start][1])))
            group=entries[start:start+size]
            inputs=self.tokenizer.pad([{"input_ids":ids} for _,ids,_ in group],padding=True,return_tensors="pt").to(self.device)
            logits=self.model(**inputs).logits.detach().to(torch.float32).cpu().numpy()
            for (i,_,weight),row in zip(group,logits):
                total[i]+=row.astype(np.float64)*weight
                mass[i]+=weight
            start+=len(group)
        return (total/mass[:,None]).astype(np.float32)

    def predict(self,text):
        role=int(self._scores([self._validated(text)])[0].argmax())
        return {"label":self.model.config.id2label[role]}

    def segment(self,text):
        from safetensors.numpy import load_file
        from span_model import context_inputs,decode,merge,units
        text=self._validated(text)
        if self._sequence is None:
            spec=self.folder/"sequence.json";weights=self.folder/"sequence.safetensors"
            if not spec.is_file() or not weights.is_file():
                raise ValueError("This checkpoint needs the accompanying sequence.json and sequence.safetensors to return spans.")
            info=json.loads(spec.read_text());names=[self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
            if info["labels"]!=names:raise ValueError("Sequence labels differ from checkpoint labels.")
            params=load_file(str(weights))
            self._sequence=(int(info["context_left"]),params["transition"],params["neighbor_kernel"])
        left,transition,kernel=self._sequence;offsets=units(text)
        inputs=context_inputs(text,offsets,left)
        emissions=np.concatenate([self._scores(inputs[i:i+64]) for i in range(0,len(inputs),64)])
        tracked=decode(emissions,transition,kernel)
        return {"spans":merge(offsets,tracked,self.model.config.id2label)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text",nargs="?",help="A sentence, paragraph or judgment. Omit to read standard input.")
    parser.add_argument("--model-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--spans",action="store_true",help="Return half-open Unicode character ranges with their roles.")
    args=parser.parse_args();text=args.text if args.text is not None else sys.stdin.read()
    if not text.strip():parser.error("Supply a nonempty sentence, paragraph or judgment.")
    if args.threads<1:parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    classifier=JudgmentRoleClassifier(args.model_dir)
    print(json.dumps(classifier.segment(text) if args.spans else classifier.predict(text),ensure_ascii=False))
if __name__=="__main__":main()
