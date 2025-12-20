import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from lxg.models import make_ff, simple_training
from lxg.attribution import integrated_gradients
from lxg.datasets import dataset_callables, NumpyRandomSeed, TorchRandomSeed
from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations
from threshold_pruning.pruning import threshold_pruning, compute_rule_metrics
from eval_esann import comp_accuracy_metrics, comp_confusion_stats

import matplotlib.pyplot as plt

def train_model(model, tr_data: tuple[np.ndarray, np.ndarray], te_data: tuple[np.ndarray, np.ndarray], n_epochs=15):
    train_loader = DataLoader(TensorDataset(torch.from_numpy(tr_data[0]).float(),
                                            torch.from_numpy(tr_data[1]).long()),
                              shuffle=True, batch_size=32, generator=None)
    test_loader = (torch.from_numpy(te_data[0]).float(), torch.from_numpy(te_data[1]).long())
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    acc, best_model = simple_training(model, optim, loss_fn, train_loader, test_loader, device='cpu',
                                      n_batches=len(train_loader)*n_epochs, return_best_model=True)
    return acc, best_model


def comp_rule_metrics(cfire, X_fit, Y_fit, X_test, Y_test):
    stats = comp_confusion_stats(cfire.dnf, X_fit, Y_fit, X_test, Y_test)
    # compute accuracy based measures on X_fit as well:
    y_fit_pred = cfire(X_fit)
    acc_fit = comp_accuracy_metrics(Y_fit, y_fit_pred)
    acc_fit = {'fit_'+k: v for k, v in acc_fit.items()}
    stats.update(acc_fit)
    return stats

def plot_accuracy_v_ambig_v_size(pruned_rules, title):
    fig, ax1 = plt.subplots()

    _x_ticks = [p['theta'] for p in pruned_rules]
    _y_ticks_acc = [p['accuracy'] for p in pruned_rules]
    _y_ticks_size = [p['cfire'].dnf.n_rules for p in pruned_rules]
    _y_ticks_ambig = [p['n_inter_conflict'] / p['n_samples'] for p in pruned_rules]

    # left y-axis: accuracy + ambiguity
    ax1.plot(_x_ticks, _y_ticks_acc,
             label='Accuracy', color='tab:blue')
    ax1.plot(_x_ticks, _y_ticks_ambig,
             label='Ambiguity', color='tab:green', linestyle='--')

    ax1.set_ylabel('Accuracy/Ambiguity rate')
    ax1.set_xlabel('Theta')
    ax1.set_ylim(0, 1)
    ax1.set_xticks(_x_ticks)

    # right y-axis: model size
    ax2 = ax1.twinx()
    ax2.plot(_x_ticks, _y_ticks_size,
             label='Number of Rules', color='tab:orange')
    ax2.set_ylabel('Number of Rules')

    # combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)
    plt.title(title)
    plt.show()

with NumpyRandomSeed(0):
    with TorchRandomSeed(0):
        # choose a task
        task = 'abalone'
        # task = 'breastw'
        # task = 'vehicle'
        # task = 'spf'
        # task = 'vehicle'

        # train/test NN on _1/_2, train/test rules on _2/_3
        print("load data")
        (X_1, Y_1), (X_2, Y_2), (X_3, _), n_dim, n_classes = dataset_callables[task](random_state=42, as_torch=False)

        # create and train a small neural network
        print("train model")
        hidden_layers_shapes = 4 * [32]
        model = make_ff([n_dim]+ hidden_layers_shapes +[n_classes])
        acc, model = train_model(model, tr_data=(X_1, Y_1), te_data=(X_2, Y_2), n_epochs=5)
        print(f"NN accuracy: {acc}")
        # helper functions
        Y_2_model = model.predict_batch(torch.from_numpy(X_2).float()).detach().numpy()

        # choose an attribution method
        localexplainer_fn = lambda inference_fn, x, y: integrated_gradients(model=model,
                                                              inference_fn=inference_fn,
                                                              data=torch.from_numpy(x).float(),
                                                              targets=torch.from_numpy(y),
                                                              n_samples=200, return_convergence_delta=True)[0].detach().numpy()


        # compute a cfire model
        _model_inference_cfire = model.predict_batch_softmax
        expl_binarization_fn = lambda x: __preprocess_explanations(x, filtering=0.01) > 0
        cfire = CFIRE(localexplainer_fn=localexplainer_fn,
                      inference_fn=_model_inference_cfire,
                      expl_binarization_fn=expl_binarization_fn
                      )
        cfire.fit(X_2, Y_2_model)

        Y_3_model = model.predict_batch(torch.from_numpy(X_3).float()).detach().numpy()

        # collect baseline performance statistics, such as size and accuracy
        og_statistics = comp_rule_metrics(cfire, X_2, Y_2_model, X_3, Y_3_model)
        og_statistics.update(dict(theta=-.05, cfire=cfire, absolute_pruning_threshold=-1))
        # perform pruning at various thresholds
        pruning_thresholds = np.arange(0., 0.51, 0.05 )
        pruned_rules = [og_statistics]
        for i, theta in enumerate(pruning_thresholds):
            print(f"{i+1} / {len(pruning_thresholds)} pruning at theta = {theta:.2f}")
            (pruned_cfire,
             pruned_accuracy,
             absolute_pruning_threshold # number of wins each rule had to clear to remain in pruned model
             ) \
                = threshold_pruning(cfire, X_2, Y_2_model, theta)

            pruned_stats = comp_rule_metrics(pruned_cfire, X_2, Y_2_model, X_3, Y_3_model)

            r = dict(theta=theta, cfire=pruned_cfire,
                    absolute_pruning_threshold=absolute_pruning_threshold)
            r.update(pruned_stats)
            pruned_rules.append(r)

        # report performance statistics for pruned instances
        plot_accuracy_v_ambig_v_size(pruned_rules, title='beans')
        # done