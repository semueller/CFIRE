import os
from copy import deepcopy
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score

from cfire.cfire_module import CFIRE

from threshold_pruning.pruning import prune_rules, decide_by_wins
from threshold_pruning.pruning_metrics import compute_rule_metrics
from eval_esann import comp_accuracy_metrics, comp_confusion_stats
from lxg.models import DNFClassifier

from tqdm import tqdm

tasks = [
'abalone',      'breastcancer', 'diggle', 'spambase',     'wine',
'autouniv',     'ionosphere',  'breastw', 'spf',
'beans',        'btsc',         'iris', 'vehicle', 'heloc'
]

data_dir = './pruning_data/'
pfx_exp = './results'

def _load_model_outputs():
    results = []
    pth = os.path.join(data_dir, 'black_box_outputs')
    for task in tasks:
        fname = os.path.join(pth, f'{task}_model_outputs.pkl')
        results.append(pkl.load(open(fname, 'rb')))
    results_pd = pd.concat(results)
    return results_pd

def _load_data():
    results = []
    pth = os.path.join(data_dir, 'datasets')
    for task in tasks:
        fname = os.path.join(pth, f'{task}_data.pkl')
        results.append(pkl.load(open(fname, 'rb')))
        results[-1]['task'] = task
    results_pd = pd.DataFrame(results)
    return results_pd

def load_dnfs():
    results = []
    pth = os.path.join(data_dir, 'cfire_models')
    for task in tasks:
        fname = os.path.join(pth, task, f'{task}_dnfrules.pkl')
        results.append(pkl.load(open(fname, 'rb')))
        results[-1]['task'] = task
    results_pd = pd.concat(results)
    cols_keep = ['task','expl_method','model_seed','dnf','time']
    results_pd = results_pd[cols_keep]
    return results_pd

def perform_pruning():
    model_outputs = _load_model_outputs()
    datasets = _load_data()
    dnfs = load_dnfs()
    print("loaded evertyhing")

    all_the_results = []
    for _, dnf_row in tqdm(dnfs.iterrows(), total=len(dnfs)):
        task, model_seed, expl_method, = dnf_row['task'], dnf_row['model_seed'], dnf_row['expl_method']
        x_fit = datasets[datasets['task'] == task]['X_orig'].iloc[0]
        x_test = datasets[datasets['task'] == task]['X_val'].iloc[0]
        y_fit = model_outputs[(model_outputs['task'] == task) & (model_outputs['model_seed'] == model_seed)]['y_orig'].iloc[0]
        y_test = model_outputs[(model_outputs['task'] == task) & (model_outputs['model_seed'] == model_seed)]['y_val'].iloc[0]
        dnf = dnf_row['dnf'] # already contains rule_performances

        cdnf = CFIRE(localexplainer_fn=None, explanations=None, inference_fn=None)
        cdnf.dnf = deepcopy(dnf)
        og_metrics = compute_rule_metrics(cdnf, x_fit)
        og_accuracy = accuracy_score(dnf(x_fit), y_fit)

        decision = decide_by_wins(og_metrics, win_threshold=0)
        safe_rules = prune_rules(dnf, decision.to_remove)

        safe_dnf = DNFClassifier(safe_rules, "accuracy")
        safe_dnf.compute_rule_performance(x_fit, y_fit)
        safe_acc = accuracy_score(safe_dnf(x_fit), y_fit)

        assert safe_acc == og_accuracy
        safe_cdnf = CFIRE(localexplainer_fn=None, explanations=None, inference_fn=None)

        safe_cdnf.dnf = deepcopy(safe_dnf)
        rule_metrics_safe = compute_rule_metrics(safe_cdnf, x_fit)
        best_pruned_dnf = deepcopy(safe_dnf)
        best_prune_threshold = 0
        for win_threshold in range(0, 100):
            decision = decide_by_wins(og_metrics, win_threshold=win_threshold)
            pruned_candidate_rules = prune_rules(dnf.rules, decision.to_remove)
            pruned_candidate_dnf = DNFClassifier(pruned_candidate_rules, "accuracy")
            pruned_candidate_dnf.compute_rule_performance(x_fit, y_fit)
            pruned_acc = accuracy_score(pruned_candidate_dnf(x_fit), y_fit)
            if 0 < og_accuracy and pruned_acc < 0.95*og_accuracy: # 5% allowance on original performance
                break
            else:
                best_pruned_dnf = deepcopy(pruned_candidate_dnf)
                best_acc = pruned_acc
                best_prune_threshold = win_threshold
        best_cdnf = CFIRE(localexplainer_fn=None, explanations=None, inference_fn=None)
        best_cdnf.dnf = deepcopy(best_pruned_dnf)
        rule_metrics_best = compute_rule_metrics(best_cdnf, x_fit)

        eval = []
        for _dnf, str_dnf in zip([dnf, safe_dnf, best_pruned_dnf],['original', 'safe', 'best']):
            confusion_fit = comp_confusion_stats(_dnf, _, _, x_fit, y_fit)
            confusion_fit = {'val_'+k: v for k,v in confusion_fit.items()}
            confusion_test = comp_confusion_stats(_dnf, _, _, x_test, y_test)
            confusion_test = {'test_' + k: v for k, v in confusion_test.items()}
            y_fit_pred = _dnf(x_fit)
            performance_fit = comp_accuracy_metrics(y_fit, y_fit_pred)
            performance_fit = {'val_'+k: v for k,v in performance_fit.items()}
            y_test_pred = _dnf(x_test)
            performance_test = comp_accuracy_metrics(y_test, y_test_pred)
            performance_test = {'test_'+k: v for k,v in performance_test.items()}
            r = dict(pruning=str_dnf,
                     dnf=_dnf.rules,
                     n_rules=_dnf.n_rules,
                     prune_threshold = best_prune_threshold if str_dnf == 'best' else 0,
                     task=task,
                     model_seed=model_seed,
                     expl_method=expl_method,
                     )
            r.update(confusion_fit)
            r.update(confusion_test)
            r.update(performance_fit)
            r.update(performance_test)

            eval.append(r)
        eval_pd = pd.DataFrame(eval)
        all_the_results.append(eval_pd)
    results_pd = pd.concat(all_the_results)
    fname = pfx_exp+'/all_tasks.csv'
    results_pd.to_csv(fname, index=False)
    return

if __name__ == '__main__':

    # model_outputs = _load_model_outputs()
    # datasets = _load_data()
    # dnfs = load_dnfs()
    perform_pruning()