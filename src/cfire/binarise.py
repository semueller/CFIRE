

import numpy as np


from copy import deepcopy








def __preprocess_explanations(explanations, filtering):
    # normalize magnitude
    e = deepcopy(explanations)
    _max = np.expand_dims(np.max(np.abs(e), -1), -1)
    _max[np.argwhere(_max == 0)[:, 0], 0] = 1  # to avoid divide by zero
    e /= _max

    if type(filtering) == str:  # top_k, k either being an integer or expressed via fraction
        _k = eval(filtering.split('topkabs')[1] if 'abs' in filtering
                       else filtering.split('topk')[1])
        if _k < 1:
            _k = int(np.ceil(_k * e.shape[-1]))
        if 'abs' in filtering:
            _arg_sorted = np.argsort(np.abs(e), axis=-1)[:, :-_k]
        else:
            _arg_sorted = np.argsort(e, axis=-1)[:, :-_k]

        for i, _args in enumerate(_arg_sorted):
            e[i, _args] = 0.

        e[e < 0.] = 0.
    else:
        # e[np.abs(e) < significance_threshold] = 0. USING NEGATIVE VALUES IN EXPLS AS A PATTERN TO COMPUTE A DNF MAKES NO SENSE
        e[e < filtering] = 0.
    return e


def __preprocess_explanations_ext(
    explanations: np.ndarray,
    *,
    top_k: int | None = None,
    fraction: float | None = None,
    threshold: float | None = None,
    use_abs: bool = False
):
    """
    Scale each row to [-1, 1] then zero-out less relevant features using a filter, choose ONE:
    - `top_k`: keep k most relevant features per row
    - `fraction`: keep fraction of most relevant values per row
    - `threshold`: keep features with values above threshold

    Optional Flag:
    - `use_abs`: Consider absolute values, instead of priotizing positive ones
    """

    # make sure only one filter is applied
    if sum(arg is not None for arg in (top_k, fraction, threshold)) != 1:
        raise ValueError("Specify exactly one of top_k, fraction, or threshold")

    # normalize magnitude
    e = deepcopy(explanations)
    _max = np.expand_dims(np.max(np.abs(e), -1), -1)
    _max[np.argwhere(_max == 0)[:, 0], 0] = 1  # to avoid divide by zero
    e /= _max

    # TODO: we keep negatives for the normalization, good idea?!

    # apply value threshold, early return
    if threshold is not None:
        e[e < threshold] = 0. # TODO: should we also clamp here?
        return e

    # determine k value (via top_k or fraction)
    k = int(np.ceil(fraction * e.shape[-1]) if fraction is not None else top_k)

    # zero out all values expect top k
    order = np.argsort(np.abs(e) if use_abs else e, axis=-1)[:, :-k]
    for i, _args in enumerate(order):
        e[i, _args] = 0.

    # clamp negatives TODO: can this actually happen?
    e[e < 0.] = 0.

    return e


def binarise_threshold(explanations, threshold: float = 0.01):
    """Binary relevance matrix from attribution scores (default binariser of CFIRE).

    Each row is divided by its largest absolute value; features with a normalised score below `threshold`
    (including all negative scores) are not relevant. Returns a boolean array of the same shape.
    """
    return __preprocess_explanations(explanations, threshold) > 0
