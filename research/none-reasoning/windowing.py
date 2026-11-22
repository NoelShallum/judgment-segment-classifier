"""Cover a supplied excerpt completely and assign each content token to one window."""


def content_windows(length, capacity, overlap=128):
    """Return (start, stop, voting weight) tuples. Weights sum to content length."""
    if length < 0 or capacity < 2 or overlap < 0:
        raise ValueError("Invalid excerpt window dimensions")
    if length <= capacity:
        return [(0, length, max(1, length))]
    overlap = min(overlap, capacity // 2)
    step = capacity - overlap
    starts = list(range(0, length - capacity + 1, step))
    last = length - capacity
    if starts[-1] != last:
        starts.append(last)
    boundaries = [0]
    for previous, current in zip(starts, starts[1:]):
        boundaries.append((previous + capacity + current) // 2)
    boundaries.append(length)
    return [(start, start + capacity, high - low)
            for start, low, high in zip(starts, boundaries, boundaries[1:])]


def token_windows(tokenizer, content_ids, model_limit):
    capacity = model_limit - tokenizer.num_special_tokens_to_add(pair=False)
    return [(tokenizer.build_inputs_with_special_tokens(content_ids[left:right]), weight)
            for left, right, weight in content_windows(len(content_ids), capacity)]
