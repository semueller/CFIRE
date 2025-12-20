import os
import pickle as pkl
import pandas as pd
import numpy as np
from pyparsing import results
from setuptools.discovery import remove_stubs

import matplotlib.pyplot as plt
import seaborn as sns

from lxg.models import DNFClassifier


# ============================================================================
tasks = [
'abalone',      'breastcancer', 'diggle',       'spambase',     'wine',
'autouniv',     'breastw',      'ionosphere',   'spf',
'beans',        'btsc',         'iris',         'vehicle',
]

pfx_exp = './results'
pfx_plts = './plots'
pfx_tbls = './tables'


# ============================================================================


def comp_accuracy_metrics(Y_true, Y_pred):
    from sklearn.metrics import precision_score, recall_score

    accuracy_include_rejected = np.mean(Y_true == Y_pred)

    rejection = np.mean(Y_pred == -1)
    coverage = 1. - rejection

    recall = recall_score(Y_true, Y_pred, average='micro', zero_division=0)

    _predicted_idxs = np.argwhere(Y_pred != -1).squeeze(-1)

    _Yt = Y_true[_predicted_idxs]
    _Yp = Y_pred[_predicted_idxs]

    if len(_Yp) == 0:
        accuracy_exclude_rejected = 0
        precision = 0
        f1 = 0
    else:
        accuracy_exclude_rejected = np.mean(_Yt == _Yp)
        precision = precision_score(_Yt, _Yp, average='micro', zero_division=0)

        f1 = 0
        if precision > 0 or recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'accuracy': accuracy_include_rejected,
        'accuracy_no_rejected': accuracy_exclude_rejected,
        'coverage': coverage,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def comp_confusion_stats(dnfc, X_fit, Y_fit, X_test: np.ndarray, Y_test):
    '''
    compute how often
    1) single rule returned, no conflict
    2) multiple rules, but all same class, intra-conflict
    3) multiple rules, but also multiple classes, inter-conflict
    '''
    if type(dnfc) != DNFClassifier:
        dnfc = DNFClassifier(rules=dnfc, tie_break='accuracy')
        dnfc.compute_rule_performance(X_fit, Y_fit)

    explanations = dnfc(X_test, explain=True)
    y_pred = np.array([e[0] for e in explanations])
    n_samples = len(X_test)
    n_no_conflict = 0
    n_intra = 0
    n_inter = 0
    n_missing = 0
    for _class, rules in explanations:
        # rules are list[tuple[tuple[class, rule_idx], dnf]]
        if _class == -1 or rules is None:
            n_missing += 1
        elif len(rules) == 1:
            n_no_conflict += 1
        else:
            classes = [t[0][0] for t in rules]
            if len(np.unique(classes)) == 1:
                n_intra += 1
            else:
                n_inter += 1
    d = comp_accuracy_metrics(Y_test, y_pred)
    _results = dict(
        n_samples=n_samples,
        n_no_rule=n_missing,
        n_no_conflict=n_no_conflict,
        n_intra_conflict=n_intra,
        n_inter_conflict=n_inter,
    )
    _results.update(d)
    return _results

# ============================================================================

def fmt(mean, std, digits_mean=2, digits_std=2):
    return (
        f"{mean:.{digits_mean}f}"
        f"{{\\scriptsize$\pm${std:.{digits_std}f}}}"
    )

def print_latex_overview_table(new_results=True, pruning='original'):

    # read csv
    df = pd.read_csv(pfx_exp+'/all_tasks.csv')
    df = df[df['pruning'] == pruning]
    method_map = {
        "ks": "CFIRE-KS",
        "li": "CFIRE-LI",
        "ig": "CFIRE-IG",
    }
    df["method"] = df["expl_method"].map(method_map)

    # %-ambiguous per run
    df["ambiguous_pct"] = 100 * df["val_n_inter_conflict"] / df["val_n_samples"]

    # aggregate over 50 model_idx per (task, method)
    grouped = df.groupby(["task", "method"]).agg(
        F1_mean=("test_f1", "mean"),
        F1_std=("test_f1", "std"),
        Size_mean=("n_rules", "mean"),
        Size_std=("n_rules", "std"),
        Amb_mean=("ambiguous_pct", "mean"),
        Amb_std=("ambiguous_pct", "std"),
    )
    rows = []
    methods_order = ["CFIRE-KS", "CFIRE-LI", "CFIRE-IG"]

    for task, sub in df.groupby("task"):
        # sample size: take test_n_samples for that dataset (they should all be equal)
        sample_size = int(sub["test_n_samples"].iloc[0])

        # three metrics in the order you want
        metric_specs = [
            ("F1", "F1_mean", "F1_std", "F1", 2, 2),
            ("Size", "Size_mean", "Size_std", "Size", 1, 1),
            ("Amb", "Amb_mean", "Amb_std", r"\%-$\lightning$", 1, 1),
        ]

        for i, (metric_key, mean_col, std_col, metric_label, d_mean, d_std) in enumerate(metric_specs):
            if i == 0:
                dataset_cell = task
            elif i == 1:
                dataset_cell = f"$n={sample_size}$"
            else:
                dataset_cell = "acc=XX"

            row = {
                "Dataset": dataset_cell,
                "Metric": metric_label,
            }

            for m in methods_order:
                if (task, m) in grouped.index:
                    g = grouped.loc[(task, m)]
                    row[m] = fmt(g[mean_col], g[std_col], d_mean, d_std)
                else:
                    row[m] = "--"

            rows.append(row)

    table_df = pd.DataFrame(rows, columns=["Dataset", "Metric"] + methods_order)
    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="llccc"
    )

    lines = latex.splitlines()

    # find the header/body \midrule
    mid_idx = next(i for i, line in enumerate(lines) if r"\midrule" in line)

    header = lines[:mid_idx + 1]  # includes the first \midrule
    body = lines[mid_idx + 1:-2]  # table rows
    footer = lines[-2:]  # \bottomrule and \end{tabular}

    new_body = []
    for i, line in enumerate(body):
        new_body.append(line)
        # after every 3 data rows, insert a midrule except after the last block
        if (i % 3 == 2) and (i != len(body) - 1):
            new_body.append(r"\midrule")

    new_latex = "\n".join(header + new_body + footer)
    print()
    print(f"Table PRUNING = {pruning}")
    print(new_latex)
    print(f"Table PRUNING = {pruning}")
    print()
    _f = pfx_tbls+f'full_overview_{pruning}.tex'
    with open(_f, 'w') as f:
        f.write(new_latex)


def plot_stuff(new_results=True):
    # './experiments/5_final_with_artifacts/results/cfire_orig.csv'
    if new_results:
        df = pd.read_csv(pfx_exp+'/all_tasks.csv')
        orig = df[df['pruning'] == 'original'].copy()
        best = df[df['pruning'] == 'best'].copy()
        safe = df[df['pruning'] == 'safe'].copy()
        method_title = {
            "ks": "CFIRE-KS",
            "li": "CFIRE-LI",
            "ig": "CFIRE-IG",
        }
    else:
        orig = pd.read_csv(os.path.join(pfx_exp, 'full_metrics_origs.csv'))
        best = pd.read_csv(os.path.join(pfx_exp, 'full_metrics_bests.csv'))
        safe = pd.read_csv(os.path.join(pfx_exp, 'full_metrics_safes.csv'))
        method_title = {
            "kernelshap": "KS",
            "lime": "LI",
            "IG": "IG",
        }

    orig["pruning"] = "none"
    safe["pruning"] = "safe"
    best["pruning"] = "best"
    df = pd.concat([orig, safe, best], ignore_index=True)

    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{stmaryrd}",
        "font.size": 9,  # base font size
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,  # default legend font
        "figure.titlesize": 9,
    })
    # ---------- aggregation ----------
    df["ambiguous_pct"] = 100 * df["test_n_inter_conflict"] / df["test_n_samples"]

    agg = (
        df.groupby(["task", "expl_method", "pruning"])
          .agg(
              f1=("test_f1", "mean"),
              size=("n_rules", "mean"),
              amb=("ambiguous_pct", "mean"),
          )
          .reset_index()
    )

    baseline = (
        agg[agg["pruning"] == "none"]
        .set_index(["task", "expl_method"])
        .rename(columns={"size": "size_base", "amb": "amb_base"})
    )

    agg = agg.set_index(["task", "expl_method"]).join(
        baseline[["size_base", "amb_base"]]
    ).reset_index()

    agg["size_rel"] = 100 * agg["size"] / agg["size_base"]
    agg["amb_rel"] = np.where(
        agg["amb_base"] > 0,
        100 * agg["amb"] / agg["amb_base"],
        np.nan,
    )

    # ---------- common plotting setup ----------
    sns.set(style="whitegrid")
    tasks = sorted(agg["task"].unique())
    palette = sns.color_palette("tab20", n_colors=len(tasks))
    color_map = dict(zip(tasks, palette))

    marker_map = {"none": "o", "safe": "s", "best": "^"}
    style_order = ["none", "safe", "best"]

    def add_lines(ax, data, y_col):
        for (task, _), sub in data.groupby(["task", "expl_method"]):
            sub = sub.set_index("pruning").loc[style_order].reset_index()
            ax.plot(
                sub["f1"],
                sub[y_col],
                linewidth=1,
                color=color_map[task],
                alpha=0.8,
            )

    from matplotlib.lines import Line2D

    # ============================================================
    # ONE FIGURE: 2 rows (size_rel, amb_rel) × 3 methods (IG, KS, LI)
    # ============================================================
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6),
        sharex=True,  # shared F1 axis
        sharey=True,
    )

    if new_results:
        methods = ['ks', 'li', 'ig']
    else:
        methods = ["kernelshap", "lime", "IG"]

    for col, method in enumerate(methods):
        title = method_title[method]
        data = agg[agg["expl_method"] == method]

        # ---------- top row: relative rule size ----------
        ax_top = axes[0, col]
        sns.scatterplot(
            data=data,
            x="f1",
            y="size_rel",
            hue="task",
            style="pruning",
            hue_order=tasks,
            style_order=style_order,
            palette=color_map,
            markers=marker_map,
            s=80,
            ax=ax_top,
            legend=False,
        )
        add_lines(ax_top, data, y_col="size_rel")

        ax_top.set_title(title, fontsize=17)
        ax_top.set_xlim(0.3, 1.0)
        ax_top.set_ylim(-5, 105)
        ax_top.set_xlabel("")            # x-label only on bottom row
        if col == 0:
            ax_top.set_ylabel(r"Size ($\Delta$\%)", fontsize=14)
        else:
            ax_top.set_ylabel("")

        # ---------- bottom row: relative %-ambiguous ----------
        ax_bot = axes[1, col]
        sns.scatterplot(
            data=data,
            x="f1",
            y="amb_rel",
            hue="task",
            style="pruning",
            hue_order=tasks,
            style_order=style_order,
            palette=color_map,
            markers=marker_map,
            s=80,
            ax=ax_bot,
            legend=False,
        )
        add_lines(ax_bot, data, y_col="amb_rel")

        ax_bot.set_xlim(0.3, 1.0)
        ax_bot.set_ylim(-5, 105)
        ax_bot.set_xlabel("F1")

        if col == 0:
            # ax_bot.set_ylabel(r"$\Delta\lightning$ (%)")
            ax_bot.set_ylabel(r"Amb ($\Delta$\%)", fontsize=14)
        else:
            ax_bot.set_ylabel("")

    # ---------- dataset legend to the right ----------
    dataset_handles = [
        Line2D([], [], marker="o", linestyle="", color=color_map[t], label=t)
        for t in tasks
    ]

    # fig.legend(
    #     handles=dataset_handles,
    #     title="Dataset",
    #     loc="center left",
    #     bbox_to_anchor=(0.85, 0.5),  # <- moved left (from 1.02)
    # )

    fig.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="center left",
        bbox_to_anchor=(0.83, 0.62),
        frameon=False,  # remove the box
        fontsize=15,  # legend text
        title_fontsize=16,  # legend title
        markerscale=1.2,  # slightly larger markers, optional
    )
    shape_handles = [
        Line2D([], [], marker="o", linestyle="", color="black", label="none"),
        Line2D([], [], marker="s", linestyle="", color="black", label=r"$\theta=0$ (safe)"),
        Line2D([], [], marker="^", linestyle="", color="black", label=r"$\theta=0.05$"),
    ]

    fig.legend(
        handles=shape_handles,
        title="Pruning",
        loc="center left",
        bbox_to_anchor=(0.83, 0.13),  # below the dataset legend
        fontsize=15,
        title_fontsize=16,
        frameon=False,  # remove legend box
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])  # <- was 0.8
    # fig.savefig(pfx_plts+"/rel_changes_plot.eps", format="eps", dpi=600)
    fig.savefig(pfx_plts+"/rel_changes_plot.pdf", format="pdf", dpi=600)
    fig.suptitle(
        "Pruning effects on CFIRE rule models",
        y=1.,
    )
    plt.show()


def print_latex_pruning_table():
    # load extended metrics

    df = pd.read_csv(pfx_exp+'/all_tasks.csv')
    orig = df[df['pruning'] == 'original']
    best = df[df['pruning'] == 'best']
    safe = df[df['pruning'] == 'safe']

    orig['pruning'] = 'original'
    best['pruning'] = 'best'
    safe['pruning'] = 'safe'

    df = pd.concat([orig, safe, best], ignore_index=True)

    key_cols = ['expl_method', 'task', 'model_seed']

    # baseline (original) per run
    base = (
        df[df['pruning'] == 'original']
        [key_cols + [
            'val_f1', 'test_f1', 'n_rules',
            'val_n_inter_conflict', 'val_n_samples',
            'test_n_inter_conflict', 'test_n_samples',
        ]]
        .rename(columns=lambda c: c + '_orig' if c not in key_cols else c)
    )

    # pruned runs (safe + best), with baseline joined
    pruned = (
        df[df['pruning'].isin(['safe', 'best'])]
        .merge(base, on=key_cols, how='inner')
    )

    # %-ambiguous (val, test) for pruned and orig
    pruned['amb_val'] = 100 * pruned['val_n_inter_conflict'] / pruned['val_n_samples']
    pruned['amb_val_orig'] = 100 * pruned['val_n_inter_conflict_orig'] / pruned['val_n_samples_orig']

    pruned['amb_test'] = 100 * pruned['test_n_inter_conflict'] / pruned['test_n_samples']
    pruned['amb_test_orig'] = 100 * pruned['test_n_inter_conflict_orig'] / pruned['test_n_samples_orig']

    # relative changes in percent (Δ%)
    pruned['d_f1_val'] = 100 * (pruned['val_f1'] - pruned['val_f1_orig']) / pruned['val_f1_orig']
    pruned['d_f1_test'] = 100 * (pruned['test_f1'] - pruned['test_f1_orig']) / pruned['test_f1_orig']
    pruned['d_size'] = 100 * (pruned['n_rules'] - pruned['n_rules_orig']) / pruned['n_rules_orig']

    pruned['d_amb_val'] = np.where(
        pruned['amb_val_orig'] > 0,
        100 * (pruned['amb_val'] - pruned['amb_val_orig']) / pruned['amb_val_orig'],
        np.nan,
    )
    pruned['d_amb_test'] = np.where(
        pruned['amb_test_orig'] > 0,
        100 * (pruned['amb_test'] - pruned['amb_test_orig']) / pruned['amb_test_orig'],
        np.nan,
    )

    # aggregate mean/std over all datasets/models per (pruning, expl_method)
    summary = (
        pruned.groupby(['pruning', 'expl_method'])
        .agg(
            d_f1_val_mean=('d_f1_val', 'mean'),
            d_f1_val_std=('d_f1_val', 'std'),
            d_f1_test_mean=('d_f1_test', 'mean'),
            d_f1_test_std=('d_f1_test', 'std'),
            d_size_mean=('d_size', 'mean'),
            d_size_std=('d_size', 'std'),
            d_amb_val_mean=('d_amb_val', 'mean'),
            d_amb_val_std=('d_amb_val', 'std'),
            d_amb_test_mean=('d_amb_test', 'mean'),
            d_amb_test_std=('d_amb_test', 'std'),
        )
        .reset_index()
    )

    method_map = {
        "ks": "CFIRE-KS",
        "li": "CFIRE-LI",
        "ig": "CFIRE-IG",
    }
    methods_order = ['ks', 'li', 'ig']
    pruning_order = ['safe', 'best']
    pruning_label = {'safe': 'Safe', 'best': 'Best'}

    # metric specification: (key, mean_col, std_col, LaTeX label)
    metric_specs = [
        ('d_f1_val',  'd_f1_val_mean',  'd_f1_val_std',  r'$F1_{\mathrm{val}}$'),
        ('d_f1_test', 'd_f1_test_mean', 'd_f1_test_std', r'$F1_{\mathrm{test}}$'),
        ('d_size',    'd_size_mean',    'd_size_std',    r'$\mathrm{Size}$'),
        ('d_amb_val', 'd_amb_val_mean', 'd_amb_val_std', r'$\lightning_{\text{val}}$'),
        ('d_amb_test','d_amb_test_mean','d_amb_test_std',r'$\lightning_{\text{test}}$'),
    ]

    rows = []

    for pr in pruning_order:
        sub = summary[summary['pruning'] == pr].set_index('expl_method')

        for i, (_, mean_col, std_col, metric_label) in enumerate(metric_specs):
            row = {
                'Strat': pruning_label[pr] if i == 0 else '',
                'Metric ($\\Delta\\%$)': metric_label,
            }
            for m in methods_order:
                if m in sub.index:
                    g = sub.loc[m]
                    row[method_map[m]] = fmt(g[mean_col], g[std_col])
                else:
                    row[method_map[m]] = '--'
            rows.append(row)

    table_df = pd.DataFrame(
        rows,
        columns=['Strat', 'Metric ($\\Delta\\%$)', 'CFIRE-KS', 'CFIRE-LI', 'CFIRE-IG']
    )

    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format='llccc'
    )
    _f = pfx_tbls+'/pruned_results.tex'
    with open(_f, 'w') as f:
        f.write(latex)
    print(latex)

if __name__=='__main__':
    print_latex_overview_table(pruning='original')
    print_latex_overview_table(pruning='safe')
    print_latex_overview_table(pruning='best')
    print_latex_pruning_table()
    plot_stuff()