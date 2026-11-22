"""Source-text screening for public seven-role excerpts."""
import re
ORDINAL=re.compile(r"^\s*(?:\(?\d{1,3}[.)]\s*|[ivxlcdm]+[.)]\s+)",re.I)
TITLE=re.compile(r"\b(?:Mr|Mrs|Ms|Messrs|Shri|Smt|Sir|Dr|Justice|village|address|telephone|email|named|resident|aged|diagnos\w*|abuse|sexual|medical|murder|rape|suicide|deceased|minor child|witness named|birth|hospital|criminal offence|blood|disease)\b",re.I)
CAPS=set("""A An The This That These Those It Its In On At As For From To Of By With If Where When Whether What Why How Which Who Whose While Although No Not Nor Before After Under Upon Since Further Besides Moreover Hence Thus Therefore Accordingly However Consequently Firstly Secondly Finally Mridha We Our There They He She His Her Their One Two Having Being Learned Counsel So Invoking Perusal Here Such Suchlike Admittedly Indeed Now Once Section Sections Article Articles Act Rules Rule Order Orders Civil Code Constitution Parliament Government Central State Union India Indian High Supreme Court Courts Bench Division Single Trial Tribunal Appellate Authority Assessing Officer Government Appeal Appellant Appellants Respondent Respondents Petitioner Petitioners Plaintiff Plaintiffs Defendant Defendants Application Petition Judgment Judgement Registrar Registry Advocate General Counsel Senior Commissioner Board Collector Revenue Arbitration Contract Evidence Procedure Limitation Rent Tax Sales Customs Income Service Industrial Labour Land Acquisition Motor Vehicles Sessions Sub Judge District Magistrate Bar Council First Second Issue Issues Decision Facts Ruling Discussion Analysis Arguments Attracting According Both Attention Thereafter Even Another Then All Only Next Apart Nevertheless Aforesaid Since Under Reading Submissions Solicitor Additional Sessions Re Special Leave Civil Appeal Appeals Petition Petitions Other Others Reserved Coram J JJ OR AND THE OF IN COURT SUPREME INDIA CIVIL APPELLATE JURISDICTION ORDER REPORTABLE NON ARBITRATION CONSTITUTION Question Proceedings Condonation Dismissed Allowed Against Consequential Considering Despite Irrespective Normally Ordinarily Strictly Regarding Sometimes Taking Turning Reverting Understood Nothing Suffice Undoubtedly Yet""".split())
CAPS.discard("Mridha")


def allow(text,paragraph=False):
    cleaned=ORDINAL.sub("",text.strip())
    cleaned=" ".join(cleaned.split())
    words=cleaned.split()
    if not (8 if not paragraph else 12)<=len(words)<=(62 if not paragraph else 210):
        return None
    if re.search(r"[0-9@/\\]|https?\b",cleaned,re.I) or TITLE.search(cleaned):
        return None
    capitals=re.findall(r"\b[A-Z][A-Za-z'-]*",cleaned)
    if any(token not in CAPS for token in capitals):
        return None
    return cleaned


SENSITIVE=re.compile(r"\b(?:wife|husband|marriage|married|divorce|children|child|father|mother|brother|sister|pregnan\w*|sex\w*|illness|injur\w*|assault\w*|deat\w*|kill\w*|stab\w*|weapon\w*|victim\w*|handicap\w*|patient\w*|caste\w*|religio\w*|body|bodies|blood|wound\w*|victim\w*|convict\w*|imprison\w*|arrest\w*|accused|prosecutrix|widow\w*|confess\w*|disab\w*|rap\w*|violence|narcotic\w*|bribe\w*|died|punish\w*|testament\w*|famili\w*|family|illtreatment|accident\w*|residence|apprehended|kidnap\w*|suffer\w*|kidney|dementia|pensioner|suicid\w*)\b",re.I)
