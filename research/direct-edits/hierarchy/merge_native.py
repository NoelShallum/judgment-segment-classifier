"""Assemble the frozen 16,816,110-parameter hierarchy as an ordinary safetensors checkpoint."""
import argparse
from pathlib import Path
import shutil
from safetensors.numpy import load_file,save_file
from lex_counts import HERE,ROOT,file_hash

SOURCE="b76ebc10abdfa248a88916b049e45b53fa18bcbb4e5307c3828c95342fe563ff"
PATCH="51fbf4e0773718d7e6e593278fa59bca3eefe6fa286e9b35d1ed97134e85022d"

def assemble(donor,patch,output):
    if file_hash(donor/"hier.safetensors")!=SOURCE or file_hash(patch)!=PATCH:
        raise ValueError("Native checkpoint identity differs from the frozen benchmark")
    output.mkdir(exist_ok=True,parents=True)
    target=output/"hier.safetensors"
    if target.exists():raise FileExistsError(target)
    values=load_file(str(donor/"hier.safetensors"));updates=load_file(str(patch))
    for k,v in updates.items():
        if k not in values or values[k].shape!=v.shape or values[k].dtype!=v.dtype:
            raise ValueError(f"Tensor is incompatible: {k}")
        values[k]=v
    for name in ("word2idx.json","tag2idx.json"):
        origin=donor/name;dest=output/name
        if dest.exists() and file_hash(origin)!=file_hash(dest):raise FileExistsError(dest)
        if not dest.exists():shutil.copyfile(origin,dest)
    temp=target.with_suffix(".part")
    save_file(values,str(temp))
    temp.replace(target)
    print(str(target),file_hash(target),sum(v.size for v in values.values()))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path)
    parser.add_argument("--donor",type=Path,default=ROOT/"references/hier");parser.add_argument("--patch",type=Path,default=HERE/"uniform-retention-plus-head.safetensors")
    p=parser.parse_args();assemble(p.donor,p.patch,p.output)
