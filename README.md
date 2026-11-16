# CFIRE

CFIRE turns local explanations of a black-box classifier into a global rule model. For each class it marks the
features each explanation deems relevant, mines closed frequent sets of these features, turns them into interval rules
with shallow decision trees, and selects a compact set of rules by greedy set cover. The result is a rule model in
disjunctive normal form that describes the black box's predictions, and that abstains where no rule applies.

The local explainer is a free choice: CFIRE takes any attribution matrix (SHAP, LIME, Integrated Gradients, ...).
`CFIRESelect` fits one rule model per explainer and selects one per black box.

Work in progress: the package covers the CFIRE algorithm of the paper versions below; more features will follow.

## Install

```bash
pip install git+https://github.com/semueller/CFIRE
```

Requires Python >= 3.10, numpy and scikit-learn.

## Usage

```python
from cfire import CFIRE, CFIRESelect

# X_fit: (n, d) samples, Y_fit: the black box's predicted classes for them,
# E: (n, d) local attributions of the black box for the predicted classes (e.g. from shap, lime or captum)
cf = CFIRE(localexplainer_fn=None, explanations=E, inference_fn=None, random_state=0)
cf.fit(X_fit, Y_fit)

cf.dnf.predict(X_new)    # predicted class per sample, -1 where no rule applies
cf.dnf.to_dict()         # per class: rules as lists of [feature, lower, upper] (None = unbounded)

# several explainers for the same black box: one rule model each, selected by precision on the fit data
sel = CFIRESelect({"shap": E_shap, "lime": E_lime, "ig": E_ig}, random_state=0).fit(X_fit, Y_fit)
sel.selected_, sel.scores_
sel.dnf.predict(X_new)
```

Settings (see the docstring of `CFIRE`): relevance threshold of the binarisation (default 0.01 of the largest
attribution per sample; top-k variants in `cfire.binarise`), frequency threshold of the itemsets (default 0.01),
depth of the rule trees (adaptive by default), `random_state` for the trees.

## Paper versions

The code used for the papers is available at tags:
- CFIRE (World Conference on Explainable AI 2025): tag `wcxai2025`
- Pruning of CFIRE rule models (arXiv:2601.03776): tag `pruning-paper` (branch `pruning`)

## Contributors

Tobias Schneider and Ruben Kemna developed the pruning extension and fixed many parts of the code base on a fork of
this repository, https://github.com/iMilchshake/CFIRE, where the history of their work is kept. The package in
`src/cfire/` started from their version.

## Licence

MIT, see [LICENSE](LICENSE).
