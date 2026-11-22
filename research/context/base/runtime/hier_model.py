"""CPU inference on the public seven-role hierarchical encoder and physical CRF edits."""
import argparse
import json
from pathlib import Path
import string
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from safetensors.torch import load_file
from span_model import decode, merge, units

LABELS = ["none","facts","issue","petitioner_argument","respondent_argument","reasoning","decision"]
PUNCTUATION = str.maketrans(string.punctuation, " " * len(string.punctuation))
EMITTER = "emitter.hidden2tag."
CRF = "crf.transitions"

def path_labels(emissions, transitions):
    """Viterbi including the donor start and end scores. Roles occupy rows 3–9."""
    scores = np.asarray(emissions, dtype=np.float32).copy()
    scores[0] += transitions[1,3:]
    scores[-1] += transitions[3:,2]
    return decode(scores, transitions[3:,3:], (0.,1.,0.))

class HierModel:
    def __init__(self, directory, profile=None):
        self.directory = Path(directory)
        self.vocab = json.loads((self.directory/"word2idx.json").read_text())
        state = load_file(str(self.directory/"hier.safetensors"), device="cpu")
        if profile is not None:
            patch = load_file(str(profile))
            if set(patch) != {EMITTER+"weight", EMITTER+"bias", CRF}:
                raise ValueError("Unexpected edit tensors")
            for k,v in patch.items():
                if state[k].shape != v.shape:
                    raise ValueError(f"Incompatible shape for {k}")
                state[k] = v
        self.word = state["sent_encoder.emb.weight"]
        self.sent = self.lstm(state, "sent_encoder.lstm", self.word.shape[1])
        self.doc = self.lstm(state, "emitter.lstm", 2*self.sent.hidden_size)
        self.weight = state[EMITTER+"weight"].numpy()
        self.bias = state[EMITTER+"bias"].numpy()
        self.transitions = state[CRF].numpy()
        self.parameters = sum(v.numel() for v in state.values())

    @staticmethod
    def lstm(state, prefix, input_size):
        size = state[prefix+".weight_hh_l0"].shape[1]
        recurrent = nn.LSTM(input_size, size, batch_first=True, bidirectional=True)
        recurrent.load_state_dict({k.removeprefix(prefix+"."):v for k,v in state.items() if k.startswith(prefix+".")})
        return recurrent.eval()

    def tokenize(self, text):
        tokens = text.lower().translate(PUNCTUATION).split()
        return [self.vocab.get(s,1) for s in tokens] or [1]

    @torch.inference_mode()
    def words(self, texts, batch_size=64):
        tokens = [torch.tensor(self.tokenize(t),dtype=torch.long) for t in texts]
        order = sorted(range(len(tokens)),key=lambda i:len(tokens[i]))
        result = torch.empty(len(tokens), self.sent.hidden_size*2)
        for start in range(0,len(order),batch_size):
            idx = order[start:start+batch_size]
            encoded = [tokens[i] for i in idx]
            lens = [len(item) for item in encoded]
            feats = self.word[pad_sequence(encoded,batch_first=True)]
            _,(hidden,_) = self.sent(pack_padded_sequence(feats,lens,batch_first=True,enforce_sorted=False))
            result[idx] = hidden.transpose(0,1).reshape(len(idx),-1)
        return result

    @torch.inference_mode()
    def contextualize(self, words, independent=False):
        if independent:
            out,_ = self.doc(words.unsqueeze(1))
            return out[:,0].numpy()
        out,_ = self.doc(words.unsqueeze(0))
        return out[0].numpy()

    def logits(self, features):
        return features @ self.weight[3:].T + self.bias[3:]

    def classify(self, text):
        if not text.strip():
            raise ValueError("Supply a nonblank sentence or paragraph")
        x=self.contextualize(self.words([text]),independent=True)
        y=path_labels(self.logits(x),self.transitions)
        return {"label":LABELS[y[0]]}

    def segment_fragment(self, text, program):
        offsets=units(text)
        words=self.words([text[a:b].strip() for a,b in offsets])
        x=np.column_stack((self.contextualize(words),self.contextualize(words,independent=True)))
        from safetensors.numpy import load_file as load_program
        head=load_program(str(program))
        y=path_labels(x@head["classifier.weight"].T+head["classifier.bias"],head["transition"])
        return {"spans":merge(offsets,y,LABELS)}

    def segment(self, text):
        offsets=units(text)
        x=self.contextualize(self.words([text[a:b].strip() for a,b in offsets]))
        y=path_labels(self.logits(x),self.transitions)
        return {"spans":merge(offsets,y,LABELS)}

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text",nargs="?");ap.add_argument("--text-file",type=Path);mode=ap.add_mutually_exclusive_group();mode.add_argument("--spans",action="store_true");mode.add_argument("--fragment",action="store_true",help="A passage cut from inside a judgment, returning character spans")
    ap.add_argument("--model-dir",type=Path,default=Path(__file__).resolve().parent);ap.add_argument("--unedited",action="store_true")
    ap.add_argument("--profile",type=Path,help="Use a supplied three-tensor weights patch for a whole judgment or a single excerpt")
    ap.add_argument("--threads",type=int,default=2);args=ap.parse_args()
    if bool(args.text is not None) == bool(args.text_file):
        ap.error("Supply either a text argument or --text-file")
    text=args.text_file.read_text() if args.text_file else args.text
    torch.set_num_threads(args.threads)
    if args.profile and (args.unedited or args.fragment):ap.error("--profile is for --spans or one dominant label")
    profile=None if args.unedited or args.fragment else args.profile or args.model_dir/(("document" if args.spans else "excerpt")+".safetensors")
    model=HierModel(args.model_dir,profile)
    if args.fragment and args.unedited:
        ap.error("Use --spans --unedited for the donor CRF")
    result=model.segment_fragment(text,args.model_dir/"fragment.safetensors") if args.fragment else model.segment(text) if args.spans else model.classify(text)
    print(json.dumps(result,ensure_ascii=False))
