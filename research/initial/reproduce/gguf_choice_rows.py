"""Read seven Q8_0 unembedding rows without expanding the vocabulary matrix."""
import mmap
import struct
import numpy as np

SIZES={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def read_rows(path,tokens):
    with path.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        pos=0
        def take(fmt):
            nonlocal pos
            value=struct.unpack_from(fmt,data,pos);pos+=struct.calcsize(fmt)
            return value[0] if len(value)==1 else value
        def string(read=True):
            nonlocal pos
            n=take("<Q");start=pos;pos+=n
            return data[start:pos].decode() if read else None
        def skip(kind):
            nonlocal pos
            if kind==8:string(False)
            elif kind==9:
                item,count=take("<IQ")
                if item in SIZES:pos+=SIZES[item]*count
                elif item==8:
                    for _ in range(count):string(False)
                else:raise ValueError("Unsupported nested metadata")
            else:pos+=SIZES[kind]
        if take("<4s")!=b"GGUF" or take("<I")!=3:raise ValueError("Expected GGUF v3")
        nt,nkv=take("<QQ");alignment=32
        for _ in range(nkv):
            field=string();kind=take("<I")
            if field=="general.alignment":
                if kind!=4:raise ValueError("Incorrect GGUF alignment type")
                alignment=take("<I")
            else:skip(kind)
        tensors={}
        for _ in range(nt):
            name=string();n=take("<I");dims=tuple(take("<Q") for _ in range(n))
            dtype,offset=take("<IQ")
            if name in ("output.weight","token_embd.weight"):tensors[name]=(dims,dtype,offset)
        name="output.weight" if "output.weight" in tensors else "token_embd.weight"
        (width,vocab),dtype,offset=tensors[name]
        if dtype!=8 or width%32:raise ValueError("Expected Q8_0 rows")
        beg=((pos+alignment-1)//alignment)*alignment+offset;rowbytes=width//32*34
        block=np.dtype([("scale","<f2"),("values","i1",32)])
        rows=[]
        for t in tokens:
            if not 0<=t<vocab:raise ValueError("Option token out of vocabulary")
            quant=np.frombuffer(data[beg+t*rowbytes:beg+(t+1)*rowbytes],dtype=block)
            rows.append((quant["scale"].astype(np.float32)[:,None]*quant["values"]).reshape(width))
        return np.array(rows,dtype=np.float32),name
