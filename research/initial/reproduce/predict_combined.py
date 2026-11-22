"""Run the directly assembled BERT plus recurrent readout on a sentence or paragraph."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from hier_model import HierModel,LABELS
from windowing import token_windows

class CombinedClassifier:
    def __init__(self,bert_dir,hier_dir,head):
        self.bert=AutoModelForSequenceClassification.from_pretrained(bert_dir,local_files_only=True).eval().bert
        self.tok=AutoTokenizer.from_pretrained(bert_dir,local_files_only=True)
        self.hier=HierModel(hier_dir)
        self.head=load_file(str(head))
    @torch.inference_mode()
    def predict(self,text):
        if not text.strip():raise ValueError('Supply a nonempty excerpt')
        ids=self.tok(text,add_special_tokens=False,verbose=False)['input_ids'];parts=token_windows(self.tok,ids,512)
        vals=[];mass=[]
        for tokens,m in parts:
            vals.append(self.bert(input_ids=torch.tensor([tokens]),attention_mask=torch.ones(1,len(tokens),dtype=torch.long)).pooler_output[0].numpy());mass.append(m)
        x=np.average(vals,axis=0,weights=mass).astype(np.float32)
        h=self.hier.contextualize(self.hier.words([text]),independent=True)[0]
        y=(self.head['classifier.weight']@np.concatenate((x,h))+self.head['classifier.bias']).argmax()
        return {'label':LABELS[y]}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('text');ap.add_argument('--bert-dir',type=Path,required=True)
    ap.add_argument('--hier-dir',type=Path,default=Path(__file__).resolve().parent/'hierarchical')
    ap.add_argument('--head',type=Path,default=Path(__file__).resolve().parent/'combined.safetensors');args=ap.parse_args();torch.set_num_threads(2)
    print(json.dumps(CombinedClassifier(args.bert_dir,args.hier_dir,args.head).predict(args.text)))
