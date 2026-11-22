"""Frozen premise construction and class definitions for the NLI check."""

from hashlib import file_digest

ROLES = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
GROUPS = ("fit", "select", "confirm", "fresh")
PAIRS = ((0, 5), (1, 2), (3, 4))
HYPOTHESES = (
    "The target sentence is a procedural or connective detail outside the judgment's substantive roles.",
    "The target sentence describes what happened in the dispute or gives the factual background.",
    "The target sentence states a legal question that this court must decide in the present case.",
    "The target sentence reports a submission or argument from the party asking the court for relief.",
    "The target sentence reports a submission or argument from the party opposing the requested relief.",
    "The target sentence gives this court's analysis or justification for its decision in the present case.",
    "The target sentence states the operative decision, disposition, or order in the present case.",
)
MAX_TOKENS = 160


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


class PairEncoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.labels = self.plain(HYPOTHESES)
        self.prefix = self.plain(("Previous:", " Target sentence:", " Next:", " ... "))
        self.padding = tokenizer.pad_token_id
        self.cls = tokenizer.cls_token_id
        self.sep = tokenizer.sep_token_id
        a, b = self.plain(("context", "hypothesis"))
        pair = tokenizer("context", "hypothesis").input_ids
        if pair != [self.cls, *a, self.sep, *b, self.sep]:
            raise ValueError(f"Unexpected tokenizer special-token layout: {pair}")
        if self.padding is None:
            raise ValueError("No tokenizer padding token")
        self.target_limit = min(86, MAX_TOKENS - 3 - max(map(len, self.labels)) -
                                sum(map(len, self.prefix[:3])) - 18 - 16)
        assert self.target_limit >= 70

    def plain(self, texts):
        return self.tokenizer(list(texts), add_special_tokens=False, verbose=False).input_ids

    def encode(self, previous, focus, following):
        before, target, after = self.plain((" " + previous, " " + focus, " " + following))
        truncated = len(target) > self.target_limit
        if truncated:
            remaining = self.target_limit - len(self.prefix[3])
            head = (remaining + 1) // 2
            target = target[:head] + self.prefix[3] + target[-(remaining - head):]
        first = self.prefix[0] + before[-18:] if previous else []
        last = self.prefix[2] + after[:16] if following else []
        premise = [self.cls, *first, *self.prefix[1], *target, *last, self.sep]
        pairs = [[*premise, *terms, self.sep] for terms in self.labels]
        assert max(map(len, pairs)) <= MAX_TOKENS
        return pairs, truncated
