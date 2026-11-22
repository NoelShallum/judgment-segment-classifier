"""Viterbi over optional within-unit character cuts and the seven LegalSeg roles."""
from bisect import bisect_right
from collections import defaultdict
import re

import numpy as np

PUNCTUATION = re.compile(r"""([.!?;:])(?:[”’"'\)\]]*)\s+(?=[A-Za-z0-9"“‘(\[])""")
NUMBERED = re.compile(r"(?<=\s)(?=(?:\([ivxlcdm0-9]{1,5}\)|[0-9]{1,3}[.)]|\([a-z]\))\s+)", re.I)
SIGNALS = re.compile(r"(?<=\s)(?=(?:learned counsel|counsel for|the question|in the result|we accordingly|"
                     r"we therefore|having heard|for the foregoing reasons|held that|order accordingly)\b)", re.I)
WORD = re.compile(r"[\w]+")
ROLE_COUNT=7


def proposals(text, units):
    """Return (original unit index, cut offset, lexical signatures)."""
    source=defaultdict(set)
    starts=[a for a,_ in units]
    for match in PUNCTUATION.finditer(text):
        source[match.end()].add(match[1])
    for rule,key in ((NUMBERED,"number"),(SIGNALS,"cue")):
        for m in rule.finditer(text):
            source[m.start()].add(key)
    for cut,kinds in sorted(source.items()):
        b=bisect_right(starts,cut)-1
        a,z=units[b]
        if cut<=a or cut>=z:
            continue
        left=text[max(a,cut-90):cut].rstrip()
        right=text[cut:min(z,cut+100)]
        before=WORD.findall(left)[-2:]
        after=WORD.findall(right)[:3]
        if not before or not after:
            continue
        kind="/".join(sorted(kinds))
        first=after[0].lower()
        last=before[-1].lower()
        next_word=after[1].lower() if len(after)>1 else "<end>"
        first=first if not first.isdecimal() else "#"
        last=last if not last.isdecimal() else "#"
        next_word=next_word if not next_word.isdecimal() else "#"
        shape="lower" if right[0].islower() else "upper" if right[0].isupper() else "other"
        sizes=np.searchsorted([20,45,100,220],[cut-a,z-cut]).tolist()
        yield b,cut,(f"type:{kind}",f"shape:{kind}/{shape}",f"left:{kind}/{last}",
                     f"right:{kind}/{first}",f"prefix:{kind}/{first} {next_word}",
                     f"length:{kind}/{sizes[0]}/{sizes[1]}")


def select_cuts(text, units, vocabulary, weights, keep=2):
    selected=defaultdict(list)
    for b,cut,keys in proposals(text,units):
        ids=[vocabulary.get(k,0) for k in keys]
        p=float(weights["cut.base"]+weights["cut.delta"][ids].sum())
        selected[b].append((cut,p))
    ranked=[(b,*value) for b,v in selected.items()
            for value in sorted(v,key=lambda d:(-d[1],d[0]))[:keep]]
    return sorted(ranked)


def decode_blocks(emission, between, transition, alternatives):
    """Each original unit either keeps one role or uses one cut and two distinct roles."""
    diagonal=np.arange(ROLE_COUNT)
    current=transition[1,3:]
    source=np.empty((len(emission),ROLE_COUNT),np.int8)
    entry=np.empty_like(source)
    option=np.full((len(emission),ROLE_COUNT,ROLE_COUNT),-1,np.int16)
    for b,unary in enumerate(emission):
        if b:
            from_previous=current[:,None]+between[b-1]
            source[b]=from_previous.argmax(0)
            current=from_previous[source[b],diagonal]
        block=np.full((ROLE_COUNT,ROLE_COUNT),-np.inf)
        block[diagonal,diagonal]=unary
        for i,pair in enumerate(alternatives[b]):
            improve=pair>block
            block[improve]=pair[improve]
            option[b][improve]=i
        out=current[:,None]+block
        entry[b]=out.argmax(0)
        current=out[entry[b],diagonal]
    result=[]
    role=int((current+transition[3:,2]).argmax())
    for b in range(len(emission)-1,-1,-1):
        first=int(entry[b,role])
        cut=int(option[b,first,role])
        result.append((first,role,cut))
        if b:
            role=int(source[b,first])
    return result[::-1]
