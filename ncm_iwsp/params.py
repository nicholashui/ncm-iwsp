"""Parameter counting, matching HazyResearch/ncm ``utils/misc.py:count_params``."""


def count_params(model, exclude_params=()):
    """Return ``(total, non_embedding)`` parameter counts.

    A parameter is excluded from the second number when its name contains any
    of ``exclude_params`` as a substring. Buffers are not counted. Parameters
    with ``requires_grad=False`` (the residual codebooks) are counted.
    """
    total = sum(p.numel() for p in model.parameters())
    non_emb = total
    if exclude_params:
        for name, param in model.named_parameters():
            if any(piece in name for piece in exclude_params):
                non_emb -= param.numel()
    return int(total), int(non_emb)
