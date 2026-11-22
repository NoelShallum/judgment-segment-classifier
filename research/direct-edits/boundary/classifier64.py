"""Independent, literal publisher classify_text baseline on fixed isolated sentences."""
import json,hashlib
from collect import P,PROTO,S,dataset,load_model,np,torch
from score import OUT,confusion,scores_cm

def run():
    ids=PROTO['interim'];cases=dataset(ids,'val');sample=json.loads((P/'classification-sample.json').read_text());src=[]
    for k in ids:
        for idx in sample['row_numbers_zero_based'][k]:src.append((k,idx,*cases[k][idx]))
    model=load_model();task='rhetorical role';tasks={task:{'labels':S,'multi_label':False}};order=sorted(range(len(src)),key=lambda i:len(src[i][2]));predict=np.full(len(src),-1)
    with torch.inference_mode():
        for j in range(0,len(order),4):
            ix=order[j:j+4];texts=[src[i][2] for i in ix];result=model.batch_classify_text(texts,tasks,batch_size=len(ix));assert len(result)==len(ix)
            for i,v in zip(ix,result):predict[i]=S.index(v[task])
    y=np.array([v[3] for v in src]); vals={'publisher_classify_text':predict};ref=json.loads((OUT/'four-predictions.json').read_text());vals.update({n:np.asarray([v[k][i] for k,i,_,_ in src]) for n,v in ref.items()});out={}
    for name,p in vals.items():
        cm=confusion(y,p);out[name]=dict(**scores_cm(cm),confusion=cm.tolist(),support=cm.sum(1).tolist(),predictions=p.tolist())
    file=dict(rows=len(src),judgments=ids,row_key=[[k,i] for k,i,_,_ in src],mode='Preselected 16 individual sentences per case. Publisher batch_classify_text isolated seven-way classification. Entity span head and incumbent use the same 64 row IDs with their original context. Exact spans cannot be computed on this discontinuous sample.',results=out);(OUT/'publisher-classify-64.json').write_text(json.dumps(file,indent=2));print('publisher64',{n:(round(z['macro_f1'],5),round(z['accuracy'],5)) for n,z in out.items()},flush=True)
if __name__=='__main__':run()
