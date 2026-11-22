"""Convert the pinned PyTorch GTE encoder using the same attention implementation as evaluation."""
from argparse import ArgumentParser
import openvino as ov
import torch

from encoder_span_edit import Encoder
from seven_paths import ROOT

def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument("--encoder",default="gte",choices=("gte","granite"))
    args=ap.parse_args()
    torch.set_num_threads(1)
    export=ROOT/"runtime"/(args.encoder+"-cpu-equivalent")
    export.mkdir(parents=True,exist_ok=True)
    source=Encoder(args.encoder)
    class Wrapper(torch.nn.Module):
        def __init__(self,base):
            super().__init__()
            self.base=base
        def forward(self,input_ids,attention_mask):
            return self.base(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)[0]
    model=Wrapper(source.model).eval()
    mock=source.tok(["The facts are admitted.","We allow the appeal."],padding=True,return_tensors="pt")
    with torch.inference_mode():
        graph=ov.convert_model(model,example_input=(mock["input_ids"],mock["attention_mask"]))
    for p,name in zip(graph.get_parameters(),("input_ids","attention_mask")):
        p.output(0).get_tensor().set_names({name})
    graph.reshape({name:ov.PartialShape([-1,-1]) for name in ("input_ids","attention_mask")})
    ov.save_model(graph,export/"encoder.xml",compress_to_fp16=False)
    print("Exported",export,flush=True)
if __name__=="__main__":main()
