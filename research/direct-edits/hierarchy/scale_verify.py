"""Check class-conditional streaming algebra and native Viterbi against the independent reference functions."""
import numpy as np
from native_memory import HERE,DONOR,HierModel,fit_head,decode,dump,file_hash
from scale_program import FOLDER,empty,update,solution,compiler
def main():
    base=HierModel(DONOR);h0=np.vstack((base.weight[3:].T,base.bias[3:])).astype(np.float64)
    f=np.load(HERE/"native-base-fit-240.npz")
    end=f["ends"][:25];n=end[-1];x=f["states"][:n];y=f["gold"][:n];s=empty()
    for a,b in zip(end,end[1:]):update(s,[x[a:b],x[a:b]],y[a:b],h0)
    check=[];cpp=compiler()
    for p in (0,.28,.6,1.):
        w=solution(s["G"][0],s["B"][0],s["count"],p,.03,.5,h0)
        target=(h0+.5*(fit_head(x.astype(float),y,h0,p,.03)-h0)).astype(np.float32)
        maxerr=float(np.max(np.abs(w-target)));assert maxerr<0.000002
        z=x@w[:-1]+w[-1];one=decode(z,end,base.transitions);two=cpp(z,end,base.transitions)
        diff=int((one!=two).sum());assert diff==0
        check.append(dict(class_power=p,max_weight_difference=maxerr,cpp_vs_reference_label_changes=diff))
    dump(FOLDER/"streamed-qc.json",dict(cases=24,rows=int(n),independent_reference="native_memory.fit_head plus original Python path_labels",checks=check,program_sha256=file_hash(HERE/"scale_program.py"),decoder_sha256=file_hash(FOLDER/"viterbi.cpp")))
    print(check,flush=True)
if __name__=="__main__":main()
