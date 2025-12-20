import logging
from itertools import combinations

import numpy as np
from numpy.linalg import norm

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from captum.attr import DeepLiftShap, DeepLift, KernelShap, Lime

from cmaes import CMA

from .util import TorchRandomSeed, _get_outputs, _get_targets

def decision_surface_to_attribution(ds, x, tol=1e-5, as_torch=True):
    # ds is a point on the decision surface, x is the sample the point serves as reference for
    ds = np.array(ds).reshape(1, -1)
    x = np.array(x).reshape(1, -1)
    matrix_decision_surface_to_attribution(ds, x, tol, as_torch)

def attribution_from_surface(ds, x, tol=5e-2):
    diffs = np.array(np.abs(ds - x))
    diffs[diffs < tol] = 0
    min_diff = np.min(diffs[diffs>0])
    attrs = min_diff/diffs # set n/0 = 0
    return attrs
    attributions = attrs/norm(attrs, axis=1).reshape(-1, 1)

def matrix_decision_surface_to_attribution(Ds, X, tol=5e-2, as_torch=True, mass=1.):
    diffs = np.array(np.abs(Ds - X))
    diffs = np.where(diffs < tol, 0., diffs)
    min_diff = np.min(np.where(diffs > 0, diffs, np.inf), axis=1).reshape(-1, 1) # minimum of each row
    # _scaled = min_diff / diffs
    attrs = np.divide(min_diff, diffs, out=np.zeros_like(diffs, dtype=float), where=diffs>0)
    # attrs = np.where(_scaled == np.inf, 0., _scaled)
    attrs = attrs/norm(attrs, axis=1).reshape(-1, 1)

    if mass < 1.:
        # Adjust each row to keep only the largest values summing to mass * sum(original_row)
        target_sums = mass * np.sum(attrs, axis=1)  # target sum for each row
        filtered_diffs = np.zeros_like(attrs)
        for i, row in enumerate(attrs):
            # Sort values in descending order and calculate cumulative sum
            sorted_indices = np.argsort(-row)  # indices for descending order
            sorted_row = row[sorted_indices]
            cumulative_sum = np.cumsum(sorted_row)

            # Find the cutoff point where the cumulative sum first exceeds the target sum
            cutoff_idx = np.searchsorted(cumulative_sum, target_sums[i])

            # Set values in filtered_diffs to keep only those up to the cutoff point
            filtered_diffs[i, sorted_indices[:cutoff_idx + 1]] = sorted_row[:cutoff_idx + 1]

        attrs = filtered_diffs

    if as_torch:
        return torch.from_numpy(attrs)
    return attrs

def nearest_uniform_prediction_baseline(inference_fn, samples, label, X, Y,
                                        n_classes, distfun=None):
    # return _distance_uniform_prediction_baseline(inference_fn, samples, label, X, Y, n_classes,
    #                                              distfun=distfun, mode='nearest')
    _uniform_prediction_bls = []

    for b in samples:
        cmaes_solution, _ = cmaes_baseline(inference_fn, initial_baseline=b,
                                           n_dims=samples.shape[1], n_classes=n_classes)
        torch_cmaes_baseline = torch.tensor(cmaes_solution, dtype=torch.float)
        _bl = torch_cmaes_baseline.unsqueeze(0)
        _uniform_prediction_bls.append(_bl)
    _uniform_prediction_bls = torch.cat(_uniform_prediction_bls, dim=0)
    return _uniform_prediction_bls

def nup_ds_attribution(model, inference_fn, data, targets=None, device=None, n_classes=None,
                                        tol=5e-2, mass=1., as_torch=True):
    if device is not None:
        model = model.to(device)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model

    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)

    if n_classes is None:
        n_classes = inference_fn(data[0].unsqueeze(0)).shape[1]

    baselines = nearest_uniform_prediction_baseline(inference_fn, samples=data, label=targets, X=None,
                                             Y=None, n_classes=n_classes).detach().cpu().numpy()

    attrs = matrix_decision_surface_to_attribution(baselines, np.array(data), tol=tol, mass=mass, as_torch=as_torch)

    return attrs




def _cma_cost_function_uniform_output(inference_fn, x, target):
    y = inference_fn(x).detach().numpy()
    loss = np.sum(np.abs(y - target), axis=1)
    return loss

def is_numeric(value):
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)

def cmaes_baseline(inference_fn, n_dims, n_classes, initial_baseline=0., std=0.5, n_generations=500,
                   criterion=_cma_cost_function_uniform_output, threshold=None):
    if threshold is None:  # magic numbers
        if n_classes <= 3:
            threshold = 1e-2
        else:
            threshold = 2e-2*n_classes

    # given a starting vector, find a vector that
    # confuses model as best as possible.
    if is_numeric(initial_baseline):
        initial_baseline = np.array([initial_baseline]*n_dims)
    b = np.array(initial_baseline)
    population_size = int(4 * (4 + np.floor(3 * np.log(n_dims))))  #4 + math.floor(3 * math.log(n_dim)) # default population size
    optimizer = CMA(mean=b, sigma=std, population_size=population_size)

    _criterion = lambda x: criterion(inference_fn, x, 1 / n_classes)
    best_score, solution, update_generation = np.inf, None, -1
    for generation in range(n_generations):
        batch = []
        for _ in range(optimizer.population_size):
            c = optimizer.ask()
            batch.append(c)
        batch = np.vstack(batch)
        torch_batch = torch.tensor(np.vstack(batch), dtype=torch.float)
        fitness = _criterion(torch_batch)

        tell_data = list(zip(batch, fitness.astype(np.float64)))
        optimizer.tell(tell_data)

        if best_score > np.min(fitness):
            update_generation = generation
            best_score = np.min(fitness)
            solution = batch[np.argmin(fitness)]

        if (generation - update_generation) > 0.1*n_generations:
            # haven't updated for more than 10% of generations
            break

        if np.min(fitness) <= threshold:
            # print(fitness)
            # print(np.min(fitness))
            # print(f"Converged at generation {generation} (threshold={threshold})")
            # solution = batch[np.argmin(fitness)]
            break

        if optimizer.should_stop():
            # print(f"CMA early stop after {generation} generations")
            # solution = optimizer.mean
            break

    # if solution is None:
    #     solution = batch[np.argmin(fitness)]

    prediction = inference_fn(torch.tensor(solution, dtype=torch.float).unsqueeze(0)).detach().numpy()
    return solution, prediction


def _calc_gradients(model, data, inference_fn=None, device='cpu', return_outputs=False):
    """
    calculates gradients of data wrt. labels for inference_fn on device
    :param inference_fn: callable that returns full model output for a datum
    :param data: iterable containing the data
    :param device: device to where data is loaded
    :return:
    """
    if inference_fn is None:
        inference_fn = model
    model.zero_grad()
    gradients = []
    if return_outputs:
        _outputs = []
    try:
        with torch.set_grad_enabled(True):
            for i, (x, y) in enumerate(data):
                x = x.to(device).requires_grad_()
                y = y.to(device)
                outputs = inference_fn(x)
                # _selected_outputs = outputs[:, y]  # since y is a tensor and not a scalar this doesn't work
                # see: e=torch.eye(10); y=torch.arange(10); torch.gather(e, 1, y.unsqueeze(1)).squeeze() return vector of just 1.'s
                _selected_outputs = torch.gather(outputs, 1, y.unsqueeze(1)).squeeze()
                grads = torch.autograd.grad(torch.unbind(_selected_outputs), x)
                gradients.append(grads[0].to('cpu').detach())
                if return_outputs:
                    _outputs.append(outputs.to('cpu').detach())

        if return_outputs:
            return torch.vstack(gradients), torch.vstack(_outputs)
        else:
            return torch.vstack(gradients)

    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            logging.warning(f"attribution._calc_gradients: CUDA out of memory. Retrying on CPU.")
            model.to('cpu')  # this is why we need the model
            # value of return_outputs doesn't matter for the return syntax
            return_value = _calc_gradients(model, data, inference_fn=inference_fn,
                                           device='cpu', return_outputs=return_outputs)
            logging.warning(f"attribution._calc_gradients: Loading model back to GPU.")
            model.to('cuda')  # putting model back
            return return_value
        else:
            raise re


def vanilla_grad(model, data, targets=None, inference_fn=None, device=None, simplified=False, pre_process_fn=None):
    return smooth_grad(inference_fn=inference_fn, model=model, data=data, targets=targets,
                       n_samples=1, std=0., random_state=0, device=device, simplified=simplified,
                       pre_process_fn=pre_process_fn)


def smooth_grad(model, data, targets=None, inference_fn=None, std=1., n_samples=50,
                random_state=42, device=None, simplified=False, noise_level=0.1, pre_process_fn=None, force_random=True):
    # expecting targets.shape=(n,)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model


    if pre_process_fn is not None:
        model = model.to('cpu')
        data = data.to('cpu')
        data = pre_process_fn(data)
        model = model.to(device)

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)

    noise = noise_level * std
    if force_random:
        perturbation = torch.normal(mean=0., std=noise, size=(n_samples, *data.shape[1:]))
    else:
        with TorchRandomSeed(random_state):
            perturbation = torch.normal(mean=0., std=noise, size=(n_samples, *data.shape[1:]))

    if n_samples <= 1:
        perturbed_data = data
        targets_expanded = targets
        batch_size = data.shape[0]
    else:
        # each element in dim 0 is repeated n_samples times, contiguously
        perturbed_data = torch.repeat_interleave(data, repeats=n_samples, dim=0)
        targets_expanded = targets.unsqueeze(0).expand(n_samples, targets.shape[0]).reshape(-1)
        perturbation = perturbation.unsqueeze(0).expand(data.shape[0], *perturbation.shape).reshape(-1, *perturbation.shape[1:])
        batch_size = n_samples

    perturbed_data = perturbed_data + perturbation
    _data = DataLoader(TensorDataset(perturbed_data, targets_expanded),
                       shuffle=False, batch_size=batch_size)

    _prev_train_state = model.training
    model.train(True)
    _grads = _calc_gradients(model=model, data=_data, inference_fn=inference_fn, device=device)
    model.train(_prev_train_state)

    # sum everything up and out they go!
    idxs = torch.arange(0, perturbed_data.shape[0]+n_samples, n_samples)
    gradients = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        gradients.append(
            torch.mean(_grads[a:b], dim=0)
        )

    gradients = torch.stack(gradients)

    if simplified:
        gradients = torch.sum(gradients, dim=-1)

    if targets_was_none:
        return (gradients, targets)

    return gradients


def integrated_gradients(model, data, targets=None, inference_fn=None, baselines=None, n_samples=100, simplified=False,
                         calc_paths=None, calc_baselines=None, fit_baseline_data=False, device=None, outputmode=None,
                         pre_process_fn=None, return_convergence_delta=False, _batch_size=256,
                         subtract_baseline=False):
    """
    :param model: Classifier model
    :param data: input data, batch first
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: if none given, zero baseline is used,
        "integral approximated with left riemannian integration plus inclusion of final point",
        ie we interpolate linearly , x_0 + t(x - x_0) t = [0., ..., 1.], and include start and end points
    :param n_samples: number of steps between baseline and input -> actual steps = steps+2; only used when calc_paths is not a callable
    :param simplified: return attribution of shape (1, seqlen), accumulate attribution for each vector
            if false, returns all collected gradients of size [n_samples, n_steps, seqlen, embedding_size]
    :param calc_paths: needs to return paths between all samples as well as distance between consecutive samples
    :param calc_baselines: must return one baseline per sample
    :return:
    """

    if device is None:
        device = next(model.parameters()).device  #  this looks ugly but is apparently the way to go in vanilla pytorch

    if inference_fn is None:
        inference_fn = model

    _use_tuple = isinstance(data, tuple)

    # data = data[:2]
    # targets = targets[:2]

    if pre_process_fn is not None:
        model = model.to('cpu')
        data = data.to('cpu')
        data = pre_process_fn(data)
        model = model.to(device)

    ## put everything on cpu for now
    if _use_tuple:
        data = data[0].to('cpu')
    else:
        data = data.to('cpu')

    input = data

    # support multiple baselines per sample?
    if callable(calc_baselines):
        baselines = calc_baselines(data)  # (samples, model)?
    if baselines is None:
        baselines = torch.zeros_like(input)
    elif fit_baseline_data:
        if len(baselines.shape) == 1:
            baselines = np.expand_dims(baselines, axis=0)
        baselines = np.repeat(baselines, len(input), 0)


    # paths.size = [n_samples, n_steps, seqlen, embedding]
    paths, scaling = None, None
    if callable(calc_paths):
        paths, scaling = calc_paths(baselines, input, n_samples)
    else:
        # fallback: straight line between baseline and input,
        p = np.linspace(baselines, input, num=n_samples, dtype=np.float32)#, retstep=True)
        paths = torch.tensor(p).swapaxes(0, 1).requires_grad_()
        scaling = 1. / n_samples
        # paths = torch.tensor(np.linspace(baselines, samples, num=steps)).swapaxes(0,1).requires_grad_()
        # paths = paths[:, :-1, :]

    assert scaling is not None and paths is not None

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)


    output_shape = inference_fn(paths[0].to(device)).shape
    # places gradients in attribution
    _prev_train_state = model.training
    model.train(True)
    attribution = torch.zeros_like(paths)
    outputs = torch.zeros(*paths.shape[:2], output_shape[1])
    for i in range(paths.shape[0]):
        _path_one_sample = paths[i]
        _target_output = targets[i].expand(paths.shape[1])
        _dataset = DataLoader(TensorDataset(_path_one_sample, _target_output), shuffle=False, batch_size=_batch_size)
        attribution[i], outputs[i] = _calc_gradients(model=model, data=_dataset,
                                      inference_fn=inference_fn, device=device, return_outputs=True)
    model.train(_prev_train_state)


    # from captum.attr import IntegratedGradients
    # explainer = IntegratedGradients(inference_fn,  multiply_by_inputs=False)
    # _attr2, delta = explainer.attribute(input, target=targets,
    #                                     n_steps=steps, return_convergence_delta=True)
    #
    # _attr_eq = torch.mean(attribution, dim=1)

    # scale gradients according to distance between point and previous point
    if hasattr(scaling, 'shape') and scaling.shape == paths.shape[:-1]:
        for _ in range(paths.shape[0]):
            attribution[0] = attribution[0] * scaling[0].unsqueeze(-1)
    else:
        if np.isscalar(scaling):
            scaling = torch.full((n_samples,), scaling)
        if scaling.shape[:2] != (input.shape[0], n_samples):
            for _ in range(len(paths.shape[2:])):
                scaling = scaling.unsqueeze(-1)
            for i in range(attribution.shape[0]):
                attribution[i] *= scaling

    # sum up over path
    attribution = torch.sum(attribution, dim=1)

    if simplified:
        attribution = torch.sum(attribution, dim=-1)

    if _use_tuple:
        attribution = (attribution,)  # OMG DUH GRHNG

    _numerical_delta = torch.nan
    if outputmode == 'full' or return_convergence_delta:

        at_baseline = torch.gather(outputs[:, 0], -1, targets.unsqueeze(1)).squeeze()
        at_sample = torch.gather(outputs[:, -1], -1, targets.unsqueeze(1)).squeeze()
        diff = at_sample - at_baseline
        # for tabular data -> obtain attribution mass of each sample
        _global_attrs = attribution * (data - baselines)
        _sums = torch.sum(_global_attrs, -1)
        # for non-tabular data we need to keep summing
        while len(_sums.shape) > 1:
            _sums = torch.sum(_sums, -1)
        _numerical_delta = torch.abs(diff - _sums)
        # perc = diff / _sums
        # _delta_percent = (1.0 - perc.abs()).abs()
        # print(torch.sum(torch.le(_delta_percent, 0.05)))
        # assert torch.all(torch.le( _delta_percent, 0.05))  # check if percentage of difference is less than 5% for all samples

    if subtract_baseline:
        attribution = attribution * (data - baselines)

    if outputmode == 'full':
        if targets_was_none:
            return attribution, paths, scaling, _numerical_delta, targets
        return attribution, paths, scaling, _numerical_delta


    if return_convergence_delta:
        if targets_was_none:
            return attribution, _numerical_delta, targets
        return attribution, _numerical_delta

    if targets_was_none:
        return attribution, targets

    return attribution



def left_integrated_gradients(model, data, targets=None, inference_fn=None, baselines=None, steps=150, simplified=False,
                         calc_paths=None, calc_baselines=None, fit_baseline_data=False, device=None, outputmode=None,
                              alpha=.9, needs_softmax=True):
    """
    From
    **Miglani, V., Kokhlikyan, N., Alsallakh, B., Martin, M., & Reblitz-Richardson, O. (2020).
    Investigating Saturation Effects in Integrated Gradients. Whi. http://arxiv.org/abs/2010.12697**

    :param model:
    :param data:
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: if none given, zero baseline is used,
        "integral approximated with left riemannian integration plus inclusion of final point",
        ie we interpolate linearly , x0 + t(x - x0) t = [0., ..., 1.], and include start and end points
    :param steps: number of steps between baseline and input -> actual steps = steps+2; only used when calc_paths is not a callable
    :param simplified: return attribution of shape (1, seqlen), accumulate attribution for each vector
            if false, returns all collected gradients of size [n_samples, n_steps, seqlen, embedding_size]
    :param calc_paths: needs to return paths between all samples as well as distance between consecutive samples
    :param calc_baselines: must return one baseline per sample
    :return:
    """


    if device is None:
        device = next(model.parameters()).device  #  this looks ugly but is apparently the way to go in vanilla pytorch

    if inference_fn is None:
        inference_fn = model

    _use_tuple = isinstance(data, tuple)
    ## put everything on cpu for now
    if _use_tuple:
        data = data[0].to('cpu')
    else:
        data = data.to('cpu')

    input = data

    # support multiple baselines per sample?
    if callable(calc_baselines):
        baselines = calc_baselines(data) # (samples, model)?
    if baselines is None:
        baselines = torch.zeros_like(input)
    elif fit_baseline_data:
        baselines = baselines.repeat(input.shape[:-1] + (1,))


    # paths.size = [n_samples, n_steps, seqlen, embedding]
    paths = None
    if callable(calc_paths):
        paths = calc_paths(baselines, input, steps)
    else:
        # fallback: straight line between baseline and input,
        p = np.linspace(baselines, input, num=steps)#, retstep=True)
        paths = torch.tensor(p).swapaxes(0, 1).requires_grad_()

    assert paths is not None

    if targets is None:
        predictions = _get_targets(inference_fn, data, model, device)
    else:
        predictions = targets

    # places gradients in attribution
    attribution = torch.zeros_like(paths)
    outputs = []
    for i in range(paths.shape[0]):
        path = paths[i]
        prediction = predictions[i].expand(paths.shape[1])
        _dataset = DataLoader(TensorDataset(path, prediction), shuffle=False, batch_size=paths.shape[0])
        attribution[i], _outputs = _calc_gradients(model=model, data=_dataset,
                                      inference_fn=inference_fn, device=device, return_outputs=True)
        outputs.append(_outputs)

# --- CALCULATE WHAT IS "LEFT"
    # outputs.shape = (N, steps, n_classes)
    outputs = torch.stack(outputs)
    if needs_softmax:
        outputs = torch.softmax(outputs, dim=-1)
    # throw all outputs except for the target class away
    outputs = torch.stack([outputs[i, :, t] for i, t in enumerate(targets)]) # shape (samples, steps, 1)
    # get confidence at sample itself
    conf = outputs[:, -1] # shape (samples, 1)
    threshold = alpha * conf
    threshold = torch.reshape(torch.repeat_interleave(threshold, steps), shape=outputs.shape) # shape (samples, steps, 1)
    mask = outputs <= threshold # shape (samples, steps)
    '''
    _each_ step that is below threshold is further considered, regardless of whether some prior steps might
    have been below threshold already;
    also: during testing I observed cases where outputs[i, -1] was not only not the maxmium, it acutally was 
    the _minimum_ -> meaning the mask zero'd out the whole attribution sequence
    '''
    # shape (samples, steps, *data.shape)
    mask = torch.reshape(mask, shape=(*mask.shape, *(1,)*len(attribution.shape[2:])))
    # all attributions at steps where outputs were below threshold will be kept, all other set to 0
    attribution = mask * attribution

    attribution = torch.sum(attribution, dim=1)

    if simplified:
        attribution = torch.sum(attribution, dim=-1)

    if _use_tuple:
        attribution = (attribution,)

    if outputmode == 'full':
        return attribution, paths, threshold, mask

    return attribution



def compute_output_curves(model, sample, target, baselines, inference_fn=None, steps=150,
                  saturation_threshold=.9, device=None, return_grads=False,
                  ):
    ''' computes outputs on paths from **INPUT TO BASELINE** '''

    if device is None:
        device = next(model.parameters()).device  #  this looks ugly but is apparently the way to go in vanilla pytorch

    if inference_fn is None:
        inference_fn = model

    p = np.linspace(sample, baselines, num=steps)#, retstep=True)
    paths = torch.tensor(p).swapaxes(0, 1).requires_grad_()

    outputs = []
    if return_grads:
        gradients = []
        targets = target.expand(paths.shape[1])

    for i, path in enumerate(paths):
        if return_grads:
            _data = DataLoader(TensorDataset(path, targets), shuffle=False, batch_size=paths.shape[1])
            grads, outs = _calc_gradients(model, _data, inference_fn, device, return_outputs=True)
            gradients.append(grads)
            outputs.append(outs[:, target])
        else:
            outs = _get_outputs(inference_fn, path, model, device)[:, target]
            outputs.append(outs)


    outputs = torch.stack(outputs)
    if return_grads:
        gradients = torch.stack(gradients)
        return outputs, gradients

    return outputs


class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

def _check_baselines(baselines):
    # make it iterable if it is a scalar

    if not hasattr(baselines, '__iter__'):
        return [baselines]
    return baselines

def deeplift(model, data, targets=None, inference_fn=None, baselines=None, device=None,
             multiply_by_inputs=False, pre_process_fn=None):
    """
    **tested for tabular data only so far**
    :param model: model
    :param data: input data, expects first dim to be batchsize
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: if non given, 0 is used. If a scalar or a single vector is given DeepLift is used, if multiple vectors are given DeepLiftShap is used
    :param samples: number of samples used to approximate shapley values
    :param multiply_by_inputs: default False, Multiply attribution scores by (input - baselines) if True;
     Captum default is True.
    :return: Attribution scores computed by DeepLift or DeepLiftShap, default DeepLift

    """

    # TODO: Is this call seeded?

    if device is None:
        device = next(model.parameters()).device

    if pre_process_fn is not None:
        model = model.to('cpu')
        data = data.to('cpu')
        data = pre_process_fn(data)
        model = model.to(device)

    if inference_fn is None:
        inference_fn = model


    # captum registers hooks via functions built into nn.Module; hence we wrap any callable this way to make it compatible
    inference_fn_wrapped = WrapperModel(inference_fn)
    inference_fn_wrapped = inference_fn_wrapped.to(device)
    # if baselines is a single example or scalar, use DeepLift, else use DeepLiftShap
    # scalar                                tensor                      more than one example
    if baselines is not None and type(baselines) is torch.Tensor \
            and len(baselines.shape) > 1 and baselines.shape[0] > 1:
        explainer = DeepLiftShap(inference_fn_wrapped, multiply_by_inputs=multiply_by_inputs)
    else:  # we have a scalar or a single baseline sample
        explainer = DeepLift(inference_fn_wrapped, multiply_by_inputs=multiply_by_inputs)

    data = data.to(device)  # why is this not handled by captum
    if targets is not None:
        targets = targets.to(device)
    _prev_train_state = model.training
    model.train(True)
    attributions, delta = explainer.attribute(data, baselines, targets, return_convergence_delta=True)
    model.train(_prev_train_state)

    return attributions

def kernelshap(model, data, targets=None, inference_fn=None, baselines=None, device=None,
             n_samples=None, masks=None, baseline=None):
    """
    **tested for tabular data only so far**
    uses lime with appropriate loss function (MSE), weighting kernel (Shapley Kernel) and regularization (ie none)
    :param model: model
    :param data: DataLoader
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: defaultis None; scalar, single baseline or one baseline per input
    :param n_samples: default 25; number of samples to train regression on
    :return: Attribution scores computed by KernelShap

    """

    if device is not None:
        model = model.to(device)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model

    if n_samples is None:
        n_samples = 2 * data[0].shape[0]

    # attributions = []
    # baselines = _check_baselines(baselines)
    # if masks is None:
    #     masks = [None]
    # for x, b, y, m in zip(data, cycle(baselines), targets, cycle(masks)):
    explainer = KernelShap(inference_fn)
    data = data.to(device)

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)

    if targets is not None:
        targets = targets.to(device)
    if masks is not None:
        masks = masks.to(device)
    _prev_train_state = model.training
    model.train(True)
    attributions = explainer.attribute(inputs=data, baselines=baselines, target=targets, n_samples=n_samples,
                                       feature_mask=masks, perturbations_per_eval=2048)
    model.train(_prev_train_state)

    if targets_was_none:
        return attributions, targets
    return attributions

def lime(model, data, targets=None, inference_fn=None, baselines=None, device=None,
             n_samples=25, distance_mode="cosine"):
    """
    **tested for tabular data only so far**

    :param model: model
    :param data: input data, expects first dim to be batchsize
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: defaultis None; scalar, single baseline or one baseline per input
    :param n_samples: default 25; number of samples to train regression on
    :param distance_mode: default "cosine"; distance mode to be used in loss for surrogate model, alternatively "euclidean"
    :return: Attribution scores computed by Lime
    """

    if device is not None:
        model = model.to(device)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)


    from captum.attr._core.lime import get_exp_kernel_similarity_function
    similarity_fn = get_exp_kernel_similarity_function(distance_mode=distance_mode)

    explainer = Lime(inference_fn, similarity_func=similarity_fn)

    data = data.to(device)
    if targets is not None:
        targets = targets.to(device)
    _prev_train_state = model.training
    model.train(True)
    attributions = explainer.attribute(inputs=data, baselines=baselines, target=targets, n_samples=n_samples,
                                       perturbations_per_eval=2048)  # n_samples=25 provided the same ranking for top 10 features as n=250
    model.train(_prev_train_state)

    if targets_was_none:
        return attributions, targets
    return attributions


def _compare_ig_captum():
    n = 100
    n_steps_l = [15, 25, 50, 75, 100, 250, 500, 750, 1000]

    _, iono_test, n_dim, n_classes = get_ionosphere(random_state=42, batch_sizes=(4, 100))
    (x, y) = iter(iono_test).next()

    from util import _sample_new_model
    model, inference_fn, _ = _sample_new_model('ionosphere', [8, 8], seed=1234)

    for n_steps in n_steps_l:

        from captum.attr import IntegratedGradients
        IG = IntegratedGradients(model)

        attrIG = IG.attribute()

# def binary_cmaes(population):
#     pass
#
# def evolutionary_attribution(sample, fitness_fn, k_min=2, k_max=None,
#                              population_init_size=None, iter_budget=100,
#                              splits=(0.2, 0.5)):
#
#     n_dims = sample.shape[1]
#
#     if k_max is None:
#         k_max = int(n_dims*0.5)
#
#     if population_init_size is None or population_init_size <= 0:
#         population_init_size = 2*n_dims
#
#
#     # init population
#     _dims = np.arange(n_dims)
#     np.random.shuffle(_dims)
#     population = []
#     for i in range(population_init_size):
#
#         if len(_dims) <= 1:
#             _dims = np.arange(n_dim)
#         _idxs = _dims[:k_min]
#         _dims = _dims[k_min:]
#         p = np.zeros(n_dims)
#         p[_idxs] = 1
#         population.append(p)
#     population = np.vstack(population)
#
#
#     best_fitness = -np.inf
#     best_individual = population[0]
#
#     for n_iter in range(iter_budget):
#         fitness = np.zeros(len(population))
#         for _pi, p in enumerate(population):
#             fitness[_pi] = fitness_fn(p)
#
#         split_top = int(len(population)*splits[0])
#         split_mid = int(len(population)*splits[1]) + split_top
#         _ranking = np.argsort(-fitness)
#         _best = _ranking[:split_top]
#         _mid = _ranking[split_top:split_mid]
#         _mid = np.random.choice(_mid, size=int(len(_mid)*0.1), replace=False)
#
#         best_fitness = fitness[_ranking[0]]
#         best_individual = None
#
#         survivors = population[np.union1d(_best, _mid)]
#         popluatino = binary_cmaes(survivors)#next_generation(survivors)

def greedy_pgi_attribution(model, data, targets=None, inference_fn=None, device=None, pgi_fn=None):
    if device is None:
        device = next(model.parameters()).device  #  this looks ugly but is apparently the way to go in vanilla pytorch

    if inference_fn is None:
        inference_fn = model

    if pgi_fn is None:
        pgi_fn = lambda x, m, t, p: _default_pgi_fn(
            model=model, sample=x, target=t, init_prediction=p, binary_mask=m, inference_fn=inference_fn
        )

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)

    init_probs = torch.gather(inference_fn(data), 1, targets.unsqueeze(1)).squeeze().detach().numpy()

    attributions = []
    for x, y, ip in zip(data, targets, init_probs):
        attributions.append(
            __greedy_pgi_attribution(
                sample=x, init_prob=ip, target=y, pgi_fn=pgi_fn, min_k=1, max_k=None
            )
        )
    attributions = torch.Tensor(attributions)
    if targets_was_none:
        return attributions, targets
    return attributions


def __greedy_pgi_attribution(sample, target, init_prob, pgi_fn, min_k=1, max_k=None):
    n_dims = len(sample)

    if max_k == None:
        max_k = int(n_dims * 0.33)

    _dims = set(np.arange(n_dims))


    _init_combinations = np.array([c for c in combinations(list(_dims), min_k)])
    a = np.zeros((len(_init_combinations), n_dims))

    for i, co in enumerate(_init_combinations):
        for c in co:
            a[i, c] = 1

    gaps = [pgi_fn(sample, _a, target, init_prob) for _a in a]
    best_score = np.max(gaps)
    top_a = a[np.argmax(gaps)]
    if min_k == 1:
        top_k = [np.argwhere(top_a == 1).squeeze().item()]
    else:
        top_k = list(np.argwhere(top_a == 1).squeeze())

    for _ in range(min_k+1, max_k):
        _dims = _dims - set(top_k)
        a = np.zeros((len(_dims), n_dims))
        a[:, top_k] = 1
        for i, d in enumerate(_dims):
            a[i, d] = 1

        gaps = [pgi_fn(sample, _a, target, init_prob) for _a in a]
        _top_score = np.max(gaps)
        if best_score < _top_score:
            best_score = _top_score
            top_a = a[np.argmax(gaps)]
            _top_k = np.argwhere(top_a == 1).squeeze()
            for _k in _top_k:
                if _k not in top_k:
                    top_k.append(_k)
        else:
            break

    # improvise a weighting
    a = np.zeros(n_dims)
    for i, k in enumerate(top_k):
        if i < min_k:
            a[k] = len(top_k)
        else:
            a[k] = len(top_k) - i
    a = a/len(top_k)

    return a

def _default_pgi_fn(model, sample, target, init_prediction, binary_mask, masking_value=0.,
                   inference_fn=None):

    # binary_mask: 1 == cover feature

    idxs = np.where(binary_mask == 1)
    sample = sample.detach().clone()
    sample[idxs] = masking_value

    if inference_fn is None:
        inference_fn = model

    _prev_model_state = model.training

    model.eval()
    if len(sample.shape) < 2:
        sample = torch.unsqueeze(sample, 0)
    p = inference_fn(sample)[0, target].detach().numpy().item()
    gap = init_prediction - p
    model.train(_prev_model_state)

    return gap

if __name__ == '__main__':

    _compare_ig_captum()

    from models import make_ff
    from lxg.datasets import *

    def test_attr(model, data):
        x, y = next(iter(data))
        print(f"testing batch of size {x.shape} with labels of shape {y.shape}")

        print("deeplift and kernel shap")
        print("baselines is None")
        _l = lime(model, x, y)
        _dl = deeplift(model, data=x, targets=y)
        _ks = kernelshap(model, x, y)
        print("baseline is scalar")
        _dl = deeplift(model, data=x, targets=y, baselines=0.)
        _ks = kernelshap(model, x, y, baselines=0.)
        print("baseline is zero vector with batch dimension = 1")
        bl = torch.zeros_like(x[0]).unsqueeze(0)
        _dl = deeplift(model, data=x, targets=y, baselines=bl)
        _ks = kernelshap(model, x, y, baselines=bl)
        print("baselines are vectors with batch dimension > 1")
        idxs = torch.randint(x.shape[0], (10,))  # sample ints from range with replacement
        bl = x[idxs]
        _dl = deeplift(model, data=x, targets=y, baselines=bl)
        _ks = kernelshap(model, x, y, baselines=x-1.)

        print("start left-ig")
        _lig = left_integrated_gradients(model, x, y)
        print("start sg")
        _sg = smooth_grad(model, x, y, std=0.1, random_state=42)
        print(_sg[0])
        print(f"sg: {_sg.shape}")

        print("start vg")
        _vg = vanilla_grad(model, x, y)
        print(_vg[0])
        print(f"vg: {_vg.shape}")

        print("start ig")
        _ig = integrated_gradients(model=model, data=x, targets=None, simplified=False)
        print(_ig[0])
        print(f"ig: {_ig.shape}")


    print("==================================================")
    print("=== RUNNING IG, SG, VG ON DIFFERENT DATA TYPES ===")
    print("==================================================")

    # print("=== NLP ===")
    # _, nlp_test, size_vocab, n_classes = get_agnews(random_state=42, batch_sizes=(64, 100))
    # nlp_model = BiLSTMClassif(nr_classes=14, embed_dim=128, hid_size=256, vocab_size=size_vocab)
    # # nlp_model = torch.nn.DataParallel(nlp_model)  # DataParallel does not interface with cusstom functions, eg BiLSTMClassif.embed_sequence
    # nlp_model.to('cuda')
    #
    # start = time()
    # x, y = next(iter(nlp_test))
    # x = x.to('cuda')
    # y = y.to('cuda')
    # _x_emb = nlp_model.embed_sequences(x)
    # print(f"data shape: {_x_emb.shape}")
    # print(f"labels: {len(y)}")
    # print(torch.unique(y, return_counts=True))
    # print(f"testing batch of size {x.shape} with labels of shape {y.shape}")
    # print(f"labels: {torch.unique(y)}")
    #
    # print("start ig")
    # del x
    # x = _x_emb.to('cuda')
    # nlp_model.to('cuda')
    # _ig = integrated_gradients(inference_fn=nlp_model._forward_embedded, data=x, targets=None, model=nlp_model,
    #                            simplified=True)
    # print(f"ig: {_ig.shape}")
    # print(_ig[0])
    #
    # print("start sg")
    # x = _x_emb.to('cpu')
    # nlp_model.to('cuda')
    # _sg = smooth_grad(inference_fn=nlp_model._forward_embedded, model=nlp_model, data=x, targets=y,
    #                   std=0.1, random_state=42, simplified=True)
    # print(f"sg: {_sg.shape}")
    # print(_sg[0])
    #
    # print("start vg")
    # del x
    # x = _x_emb.to('cpu')
    # nlp_model.to('cuda')
    # _vg = vanilla_grad(inference_fn=nlp_model._forward_embedded, model=nlp_model, data=x, targets=y,
    #                    simplified=True)
    # print(f"vg: {_vg.shape}")
    # print(_vg[0])
    #
    # end = time()
    # print(f"took {end - start} seconds for {x.shape[0]} samples")
    #
    # exit()
    #
    # print("==================================================")
    # print("=== IMAGES ===")
    # _, xmnist_test, _, n_clases = get_fmnist(42, batch_sizes=(64, 20))
    # xmnist_cnn = make_fmnist_small(n_clases)
    # xmnist_cnn.to('cuda')
    # start1 = time()
    # test_attr(xmnist_cnn, xmnist_test)
    # end1 = time()
    # xmnist_cnn = torch.nn.DataParallel(xmnist_cnn)
    # xmnist_cnn.to('cuda')
    # start2 = time()
    # test_attr(xmnist_cnn, xmnist_test)
    # end2 = time()
    # print(f"single gpu: {end1-start1}")
    # print(f"dataparallel: {end2-start2}")
    # exit()

    print("==================================================")
    print("=== TABULAR ===")
    _, iono_test, n_dim, n_classes = get_ionosphere(random_state=42, batch_sizes=(4, 24))
    ff_cov = make_ff([n_dim, 8, n_classes])
    test_attr(ff_cov, iono_test)
