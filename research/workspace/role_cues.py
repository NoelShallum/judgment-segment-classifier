"""Authored evidence for seven rhetorical roles. Each feature has a separately written output row."""
import re
import numpy as np
# Features specify discourse forms and speaker attribution. They do not memorize case facts or quotations.
FEATURES=[
 ("formal_heading",0,r"^\s*(?:(?:reportable|non.reportable|reserved|coram|judg(?:e)?ment|order|appearances|contents)\b[\s:.,-]*){1,3}$"),
 ("navigation",0,r"^\s*(?:before proceeding further|at the outset|we (?:shall|now) (?:turn|proceed) to|the facts may be (?:briefly )?(?:stated|set out))\b"),
 ("heard_advocates",0,r"^\s*(?:we (?:have )?heard|heard)\s+(?:the\s+)?(?:learned\s+)?(?:senior\s+)?counsel\b"),
 ("background",1,r"\b(?:brief facts (?:of|are)|facts giving rise to|relevant facts (?:are|may be)|factual background|events leading (?:up )?to)\b"),
 ("filed_proceeding",1,r"\b(?:filed|instituted|preferred|lodged)\s+(?:an?\s+|the\s+)?(?:writ petition|appeal|suit|application|complaint|petition)\b"),
 ("earlier_order",1,r"\b(?:trial court|high court|tribunal|district court|lower appellate court)\s+(?:had\s+)?(?:dismissed|allowed|decreed|directed|remanded|set aside)\b"),
 ("procedural_event",1,r"\b(?:notice was (?:issued|served)|summons (?:were|was) issued|an? (?:agreement|sale deed) was executed|the matter was (?:listed|referred|remanded))\b"),
 ("question_for_court",2,r"\b(?:questions?|issues?|points?)\s+(?:that\s+|which\s+|now\s+|thus\s+|arising\s+){0,3}(?:arises?|falls?|before (?:us|this court)|for (?:our )?(?:determination|consideration|decision|adjudication))\b"),
 ("whether_opening",2,r"^\s*(?:\(?[a-z0-9ivx]+[.)]\s*)?whether\b"),
 ("determine_whether",2,r"\b(?:called (?:upon|on) to (?:consider|decide)|have to decide|must determine|requires determination)\s+(?:is\s+)?whether\b"),
 ("moving_speaker",3,r"\b(?:petitioner|appellant)s?(?:.s)?\s+(?:(?:has|have|further|also|respectfully|would)\s+){0,3}(?:contend|submit|urge|argue|assert|plead|maintain)\w*\b"),
 ("moving_counsel",3,r"\b(?:counsel|advocate)\b.{0,45}?\b(?:for|representing)\s+(?:the\s+)?(?:appellant|petitioner)s?\b.{0,55}\b(?:urge|argue|submit|contend|point|maintain|invite|press|emphas)\w*"),
 ("moving_attribution",3,r"\b(?:contend|submit|urge|argue)\w*\b.{0,30}?\bon behalf of (?:the )?(?:appellant|petitioner)s?\b"),
 ("opposing_speaker",4,r"\brespondents?(?:.s)?\s+(?:(?:has|have|further|also|respectfully|would)\s+){0,3}(?:contend|submit|urge|argue|assert|plead|maintain)\w*\b"),
 ("opposing_counsel",4,r"\b(?:counsel|advocate)\b.{0,45}?\b(?:for|representing)\s+(?:the\s+)?respondents?\b.{0,55}\b(?:urge|argue|submit|contend|point|maintain|invite|press|emphas)\w*"),
 ("opposing_attribution",4,r"\b(?:contend|submit|urge|argue)\w*\b.{0,30}?\bon behalf of (?:the )?respondents?\b"),
 ("judicial_evaluation",5,r"\b(?:in (?:our|my) (?:considered )?(?:view|opinion)|we (?:are (?:of the view|satisfied|unable to agree)|find (?:no|that)|consider that)|we see no reason)\b"),
 ("construction",5,r"\b(?:plain reading|harmonious construction|cannot be construed|it is (?:well )?settled (?:law|that)|the legislature (?:intended|could not have intended))\b"),
 ("weighing",5,r"\b(?:the evidence (?:shows|establishes|indicates)|this (?:interpretation|submission|contention) (?:cannot|does not)|the inference (?:is|follows))\b"),
 ("present_disposition",6,r"\b(?:we|i)\s+(?:(?:therefore|accordingly|hereby|consequently)\s+){0,2}(?:dismiss|allow|quash|remand|restore|set aside|direct)\b"),
 ("operative_order",6,r"\b(?:appeals?|petitions?|applications?|suits?)\s+(?:(?:is|are|stands?|shall stand|accordingly|hereby|therefore|consequently|partly|thus|herein)\s+){1,5}(?:dismissed|allowed|disposed of|remanded)\b"),
 ("order_costs",6,r"\b(?:no order as to costs|no order (?:on|for) costs|parties (?:shall|will) bear their own costs|pending applications? (?:shall )?stand)\b"),
 ("quash_order",6,r"\b(?:impugned (?:judgment|order|award)|conviction|direction)\s+(?:(?:is|are|stands?|hereby|accordingly|thus)\s+){1,4}(?:set aside|quashed|upheld)\b"),
 ("relief_sought",3,r"\b(?:petitioner|appellant)s?\s+(?:(?:also|further|therefore|have|has)\s+){0,3}(?:prays?|prayed|seeks?|sought)\b"),
 ("opposition_to_relief",4,r"\brespondents?\s+(?:(?:also|further|therefore|have|has)\s+){0,3}(?:oppos|resist)\w*\b"),
]
PATTERNS=[re.compile(pattern,re.I) for _,_,pattern in FEATURES]
def cue_features(texts):
    return np.asarray([[int(p.search(t) is not None) for p in PATTERNS] for t in texts],dtype=np.float32)
def programmed_rows():
    w=np.full((len(FEATURES),7),-1/7,dtype=np.float32)
    for i,(_,label,_) in enumerate(FEATURES):w[i,label]+=1
    return w
