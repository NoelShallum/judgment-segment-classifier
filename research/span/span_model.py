"""A gradient-free sequence layer for seven rhetorical roles with Unicode offset output."""
import re
import numpy as np

# Titles, citations and initials encountered in English judgments can contain periods.
ABBREVIATIONS={"mr","mrs","ms","dr","hon","ltd","pvt","co","no","nos","vs","v","para","paras","sec","art","smt","shri","sr","jr","etc","viz","fig","rs","w","p","j","jj","e.g","i.e","ld","ors","anr"}
BOUNDARIES=re.compile(r"\n[ \t]*\n+|(?<=[.!?])(?:[”’\"')]*)(?P<gap>\s+)(?=[\"“‘(\[]?[A-Z0-9])")
def units(text):
    if not isinstance(text,str) or not text.strip():raise ValueError("Supply nonempty judgment text")
    starts=[0]
    for m in BOUNDARIES.finditer(text):
        if m.group("gap") is not None:
            prev=text[max(0,m.start()-48):m.start()]
            word=re.search(r"([\w.]+)\.$",prev)
            if word:
                token=word[1].casefold().rstrip(".")
                if token in ABBREVIATIONS or (len(token)==1 and token.isalpha()) or re.fullmatch(r"(?:[a-z]\.)+[a-z]",token):continue
        i=m.end()
        while i<len(text) and text[i].isspace():i+=1
        if i<len(text) and text[starts[-1]:i].strip():starts.append(i)
    return [(a,b) for a,b in zip(starts,starts[1:]+[len(text)])]

def authored_transitions(sticky,order):
    matrix=np.eye(7,dtype=np.float32)*sticky
    # Soft preferences allow any role at any position and in any order.
    for i,js in {0:(1,2),1:(2,3,4),2:(3,4,5),3:(4,5),4:(3,5),5:(6,),6:(0,)}.items():
        for j in js:matrix[i,j]+=order
    return matrix

def decode(logits,transition,kernel):
    x=np.asarray(logits,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=7 or not len(x):raise ValueError("Expected a nonempty sequence of seven-way logits")
    before=np.vstack((x[0],x[:-1]));after=np.vstack((x[1:],x[-1]))
    x=kernel[0]*before+kernel[1]*x+kernel[2]*after
    score=x[0].copy();parents=np.empty((len(x),7),dtype=np.int64)
    for i in range(1,len(x)):
        options=score[:,None]+transition
        parents[i]=np.argmax(options,axis=0)
        score=options[parents[i],np.arange(7)]+x[i]
    labels=np.empty(len(x),dtype=np.int64);labels[-1]=np.argmax(score)
    for i in reversed(range(1,len(x))):labels[i-1]=parents[i,labels[i]]
    return labels

def merge(offsets,labels,names):
    spans=[]
    for (a,b),y in zip(offsets,labels):
        role=names[int(y)]
        if spans and spans[-1]["label"]==role:spans[-1]["end"]=b
        else:spans.append({"start":a,"end":b,"label":role})
    return spans

def context_inputs(text,offsets,left):
    pieces=[text[a:b].strip() for a,b in offsets]
    return [" ".join(pieces[max(0,i-left):i+1]) for i in range(len(pieces))]
