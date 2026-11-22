"""Download a revision-pinned public alternative encoder and verify CPU inference."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from safetensors.torch import save_file
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

NAME="nlpaueb/legal-bert-small-uncased"
REV="0e23f7a9a39f59768ea7e09766d8ee308580fb17"
WEIGHTS="28ab0bfad928a678df394a5f6a8ec666a530b6b23a21f750778f453ea127b47d"
DEST=Path(__file__).resolve().parent/"references/legal-bert-small"
FILES=("config.json","tokenizer_config.json","vocab.txt","pytorch_model.bin")


def sha(path):
    with open(path,"rb") as f:
        return file_digest(f,"sha256").hexdigest()


def fetch(name):
    dest=DEST/name
    url=f"https://huggingface.co/{NAME}/resolve/{REV}/{name}"
    if not dest.exists():
        part=dest.with_suffix(".part")
        with urlopen(Request(url,headers={"User-Agent":"AttoFlow research reproducibility"}),timeout=120) as stream, part.open("wb") as out:
            shutil.copyfileobj(stream,out)
        if name=="pytorch_model.bin":
            assert sha(part)==WEIGHTS
        part.replace(dest)
    digest=sha(dest)
    if name=="pytorch_model.bin":
        assert digest==WEIGHTS
    return {"name":name,"sha256":digest,"bytes":dest.stat().st_size,"source":url}


def main():
    DEST.mkdir(exist_ok=True,parents=True)
    with ThreadPoolExecutor(4) as pool:
        retrieved=list(pool.map(fetch,FILES))
    torch.set_num_threads(2)
    raw=torch.load(DEST/"pytorch_model.bin",weights_only=True,map_location="cpu")
    encoder={k.removeprefix("bert."):t.contiguous()
             for k,t in raw.items() if k.startswith("bert.") and not k.endswith(("position_ids","token_type_ids"))}
    config=BertConfig.from_json_file(str(DEST/"config.json"))
    model=BertModel(config).eval()
    keys=model.load_state_dict(encoder,strict=False)
    assert not keys.missing_keys and not keys.unexpected_keys,keys
    physical=DEST/"encoder.safetensors"
    save_file(encoder,str(physical))
    tokenizer=BertTokenizerFast.from_pretrained(DEST,local_files_only=True)
    inputs=tokenizer(["Counsel for the respondent objects to the appeal.","The court granted the petition."],padding=True,return_tensors="pt")
    with torch.inference_mode():
        out=model(**inputs).last_hidden_state
    assert tuple(out.shape[:1])==(2,) and out.shape[-1]==config.hidden_size and torch.isfinite(out).all()
    evidence={"model":NAME,"revision":REV,"license":"CC BY-SA 4.0","sources":retrieved,
              "encoder_sha256":sha(physical),"encoder_parameters":sum(p.numel() for p in model.parameters()),
              "layers":config.num_hidden_layers,"context_limit":config.max_position_embeddings,"cpu_probe_shape":list(out.shape)}
    (DEST/"provenance.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(evidence,flush=True)


if __name__=="__main__":
    main()
