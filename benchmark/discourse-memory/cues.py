"""Named multi-sentence discourse features, with case-normalized activation."""

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTION = ROOT / "live-nov22/boundary/attribution"
sys.path.insert(0, str(ATTRIBUTION))
from signals import ACTORS, HEADINGS, QUOTED, scan

PHASE_CUTS = (.08, .31, .66, .91)
PHASES = len(PHASE_CUTS) + 1
BASE_NAMES = (
    "present-court", "present-disposition", "reports-an-argument", "direct-party-attribution",
    "named-counsel", "prior-authority", "opens-statute", "quote-lead", "citation",
    "explicit-issue", "heading", "quote-share", "opening-quotation", "closing-quotation",
    "takes-up-quotation", "colon", "starts-with-quotation", "petitioner-speaks", "respondent-speaks",
) + tuple("heading/" + t for t in HEADINGS)
META = {
    "present-inference": r"^\s*(?:\(?[\w]{1,6}[.)]\s*)?(?:therefore|thus|consequently|it (?:follows|seems|appears) (?:that|to us)|in (?:our|my) (?:view|opinion)|we (?:find|consider|are inclined|are satisfied|are unable|have no hesitation)|for (?:the|these) reasons)\b",
    "quoted-attribution": r"\b(?:observed|held|ruled|declared|noted|reproduced|recorded)\b[^.;!?]{0,45}\b(?:as follows|as under|in the following terms|thus)\b",
    "reported-inference": r"\b(?:the (?:lower|high|trial|appellate) court|the tribunal|learned judge|earlier bench|court below)\b[^;!?]{0,75}\b(?:held|found|concluded|observed|reasoned|stated)\b",
    "citation-element": r"(?:\(\d{4}\)\s*\d*\s*(?:scc|scr|sct|scc\s+online)|air\s+\d{4}|\bparagraph\s+\d+\s+of\s+(?:the|that)\s+(?:judgment|decision))",
    "takes-up-law": r"^\s*(?:\(?[\w]{1,6}[.)]\s*)?(?:\b(?:section|article|rule|clause|sub-section|proviso)\s+[\divxlcdm]+\b|\b(?:the|this) (?:section|article|sub-section|proviso|rule) (?:provides|reads|states|requires)\b)",
    "returns-to-application": r"\b(?:turning (?:back )?to (?:the present case|the facts)|in the facts (?:and circumstances )?of (?:this|the present) case|in the present case(?: before us)?|the principle (?:is|applies) to (?:the|this) case)\b",
    "new-side": r"^\s*(?:\(?[\w]{1,6}[.)]\s*)?(?:on the other hand|in response|in reply|in rejoinder|per contra|as against (?:that|this))\b",
    "return-to-court": r"^\s*(?:\(?[\w]{1,6}[.)]\s*)?(?:we have (?:heard|considered|examined|carefully considered)|having (?:heard|considered|examined)|the (?:question|issue) (?:that )?(?:arises|before us))\b",
}
PATTERNS = [re.compile(x, re.I) for x in META.values()]
ACTIVE = (0, 2, 5, 6, 7, 11, 12, 13, 17, 18)
HORIZONS = (3, 12)


@dataclass
class Features:
    local: np.ndarray
    context: np.ndarray
    edge: np.ndarray
    phase: np.ndarray


def feature_names():
    local = BASE_NAMES + tuple(META) + ("court-in-unquoted-text", "court-in-reported-text")
    structural = (
        tuple("previous-speaker/" + a for a in ACTORS) +
        tuple("following-speaker/" + a for a in ACTORS) +
        tuple("current-speaker/" + a for a in ACTORS) +
        tuple("active-quote-source/" + q for q in QUOTED) +
        ("petitioner-alias", "respondent-alias", "speaker-change", "quote-finished", "quote-begun",
         "source-inside-quote", "counsel-report-in-quote", "court-after-quote")
    )
    common = local + structural
    neighborhood = tuple(f"{tag}/{d}/{h}" for h in HORIZONS for d in ("left", "right") for tag in (BASE_NAMES[j] for j in ACTIVE))
    edges = (
        tuple("left-speaker/" + a for a in ACTORS) +
        tuple("right-speaker/" + a for a in ACTORS) +
        tuple("right-quote-source/" + q for q in QUOTED) +
        ("speaker-change", "opening-quotation", "closing-quotation", "inside-continuing-quote", "after-quote",
         "disposition-introduced", "earlier-court-reported", "court-resumes", "citation-introduced",
         "start-heading", "counsel-switch", "new-source") +
        tuple("heading/" + a for a in HEADINGS)
    )
    return {"local": local, "context": common + neighborhood, "edge": edges}


def averages(x, span):
    n = len(x)
    i = np.arange(n)
    a = np.maximum(0, i - span)
    b = np.minimum(n, i + span + 1)
    prefix = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)))
    return ((prefix[i] - prefix[a]) / np.maximum(1, i - a)[:, None],
            (prefix[b] - prefix[i + 1]) / np.maximum(1, b - i - 1)[:, None])


def actor_channels(events, n, reset):
    """Use the nearest preceding and following attributed speaker around each row."""
    left = np.zeros((n, len(ACTORS)), np.float32)
    right = left.copy()
    current = left.copy()
    if not events:
        return left, right, current
    loc = np.asarray([x.index for x in events])
    roles = np.asarray([x.value for x in events])
    current[loc, roles] = 1
    i = np.arange(n)
    earlier = np.searchsorted(loc, i, side="left") - 1
    sampled = earlier.clip(min=0)
    lag = i - loc[sampled]
    valid = (earlier >= 0) & (lag <= 24) & (loc[sampled] >= reset)
    left[i[valid], roles[sampled[valid]]] = np.exp(-lag[valid] / 7.5)
    later = np.searchsorted(loc, i, side="right")
    sampled = later.clip(max=len(loc) - 1)
    lead = loc[sampled] - i
    valid = (later < len(loc)) & (lead <= 6)
    right[i[valid], roles[sampled[valid]]] = np.exp(-lead[valid] / 2.5)
    return left, right, current


def shift(x):
    z = np.zeros_like(x)
    z[1:] = x[:-1]
    return z


def normalize_presence(x):
    # A repeated cue can give a long passage evidence without dominating the judgment.
    scale = 1. / np.sqrt(1. + 3. * x.mean(axis=0))
    return (x * scale).astype(np.float32)


def extract(texts):
    n = len(texts)
    if n == 0:
        raise ValueError("Discourse inputs must contain at least one sentence")
    raw, events, mentions, inferred, reset = scan(texts)
    base = raw[:, 3:]
    extra = np.asarray([[bool(regex.search(s)) for regex in PATTERNS] for s in texts], dtype=np.float32)
    quote = np.maximum(base[:, 11], .5 * base[:, 14])
    use_court = np.maximum(base[:, 0], extra[:, 0])
    local = np.column_stack((base, extra, use_court * (1 - quote), use_court * quote))
    l, r, c = actor_channels(events.get("speaker", ()), n, reset)
    aliases = np.zeros((n, 2), np.float32)
    for row, pairs in mentions.items():
        if pairs:
            aliases[row, pairs[0].value] = 1
    source = np.zeros((n, len(QUOTED)), np.float32)
    qevents = {x.index: x.value for x in events.get("quote", ())}
    for idx in np.flatnonzero(inferred >= 0):
        role = qevents[int(inferred[idx])]
        lag = idx - inferred[idx]
        source[idx, role] = max(float(quote[idx]), np.exp(-lag / 4))
    speaker_change = ((l.argmax(1) != c.argmax(1)) & (c.max(1) > 0) & (l.max(1) > .15)).astype(np.float32)
    prev_quote = shift(quote)
    closes = np.maximum(base[:, 13], ((prev_quote > .35) & (quote < .15)).astype(np.float32))
    opens = np.maximum(base[:, 12], ((prev_quote < .15) & (quote > .35)).astype(np.float32))
    structural = np.column_stack((l, r, c, source, aliases, speaker_change, closes, opens,
                                  source.max(1) * quote, base[:, 2] * quote, use_court * prev_quote * (1 - quote)))
    context = np.column_stack((local, structural, *(v for h in HORIZONS for v in averages(base[:, ACTIVE], h))))
    inherited = np.maximum(c, l)
    edge = np.column_stack((shift(inherited), inherited, source, speaker_change, opens, closes,
                            np.minimum(quote, prev_quote), closes * use_court, base[:, 1],
                            extra[:, 2], np.maximum(extra[:, 7], extra[:, 5]), base[:, 8],
                            base[:, 10], extra[:, 6], source.max(1) * opens, base[:, 19:]))
    edge[0] = 0
    assert local.shape[1] == len(feature_names()["local"])
    assert context.shape[1] == len(feature_names()["context"])
    assert edge.shape[1] == len(feature_names()["edge"])
    phase = np.searchsorted(PHASE_CUTS, (np.arange(n) + .5) / n)
    return Features(normalize_presence(local), normalize_presence(context), normalize_presence(edge), phase)
