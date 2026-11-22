"""Export the exact pinned three-label NLI model for local FP32 acceleration."""
import openvino as ov
import torch
from nli_span_edit import Entailment,ROOT

def main():
    torch.set_num_threads(1)
    target=ROOT/"runtime/nli-span-minilm"
    target.mkdir(parents=True,exist_ok=True)
    e=Entailment()
    class Logits(torch.nn.Module):
        def __init__(self,model):
            super().__init__()
            self.model=model
        def forward(self,input_ids,attention_mask):
            return self.model(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)[0]
    data=e.tokenizer(["The appeal is dismissed.","The notice was served."],["The court rules.","The facts are recounted."],
                     padding=True,return_tensors="pt")
    with torch.inference_mode():
        graph=ov.convert_model(Logits(e.model).eval(),example_input=(data["input_ids"],data["attention_mask"]))
    for p,name in zip(graph.get_parameters(),("input_ids","attention_mask")):
        p.output(0).get_tensor().set_names({name})
    graph.reshape({name:ov.PartialShape([-1,-1]) for name in ("input_ids","attention_mask")})
    ov.save_model(graph,target/"model.xml",compress_to_fp16=False)
    print("Exported",target,flush=True)
if __name__=="__main__":
    main()
