





from itertools import chain










from .rules import DNFClassifier


from .mining import ItemsetNode


def greedy_cover(universe_to_cover: set[int], candidate_rules: list[tuple[set[int], ItemsetNode]]):
    remaining_universe = universe_to_cover.copy()
    selected_cover_set = []

    while len(remaining_universe) > 0:
        # pick next rule by most samples of remaining covered
        _best_rule = sorted(candidate_rules, key=lambda rule: - len(remaining_universe.intersection(rule[0])))[0]
        candidate_rules.remove(_best_rule)
        remaining_universe = remaining_universe - _best_rule[0]
        selected_cover_set.append(_best_rule[1])
    return selected_cover_set


def _comp_greedy_cover(_supp, _node_set):
    # create universe set of items to be covered
    __items_covered = set(chain.from_iterable(_supp))

    nodes = greedy_cover(__items_covered, list(zip(_supp, _node_set)))
    dnf = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes]))],
                        tie_break="accuracy")
    return dnf


