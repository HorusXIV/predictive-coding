"""Train several identically shaped PCNs in lockstep on one GPU.

The grid search scores 5 cross-validation folds that share an architecture and
differ only in their weights and their data. Run one after another they spend
most of their time launching kernels rather than computing: one
:meth:`PredictiveCodingNetwork.predictive_step` issues around 30 CUDA kernels,
each doing too little arithmetic to cover its own launch. Stacking the folds
turns each of those launches into one ``bmm`` covering all of them.

Every fold keeps its own 2D weight parameters and its own optimizer.
``torch.optim.Muon`` rejects a 3D parameter outright, and per-fold optimizer
state has to stay separate in any case. The stack is rebuilt once per batch,
which is safe because the relaxation never writes to the weights, and the
batched gradients are sliced back into each fold's ``.grad`` before that fold's
own optimizer steps.

The arithmetic mirrors ``model.py`` exactly, so results match training the
folds one at a time up to ``bmm`` and ``mm`` accumulating floats in a
different order.
"""

import torch
import torch.nn.functional as F


def _predict(a, index, slope):
    """Apply layer ``index``'s activation, mirroring ``model.py``.

    Layer 0 predicts the real-valued visible input, so it stays linear where
    every layer above it is a LeakyReLU. That is ``visible_activation`` in
    :class:`PredictiveCodingNetwork`, and missing it silently corrupts both the
    latent init and the error signal.
    """
    return a if index == 0 else torch.where(a > 0, a, a * slope)


def _gain(a, index, slope):
    """Derivative of layer ``index``'s activation at the pre-activation."""
    return torch.ones_like(a) if index == 0 else torch.where(a > 0, 1.0, slope)


def _stack_weights(models):
    """Stack each layer's weight across folds into ``(folds, out, in)``."""
    layers = [
        torch.stack([model.layers[i].W for model in models])
        for i in range(len(models[0].layers))
    ]
    return layers, torch.stack([model.readout.weight for model in models])


def _init_latents(x, weights, slope):
    latents, h = [x], x
    for index, W in enumerate(weights):
        h = _predict(h @ W, index, slope)
        latents.append(h)
    return latents


def _errors(latents, weights, slope):
    errors, gains = [], []
    for i, W in enumerate(weights):
        a = latents[i + 1] @ W.transpose(1, 2)
        error = latents[i] - _predict(a, i, slope)
        errors.append(error)
        gains.append(error * _gain(a, i, slope))
    return errors, gains


def _relax(latents, weights, readout, target, eta_infer, T_infer, slope):
    for _ in range(T_infer):
        errors, gains = _errors(latents, weights, slope)
        errors.append((latents[-1] @ readout.transpose(1, 2) - target) @ readout)
        for i in range(1, len(weights) + 1):
            latents[i] -= eta_infer * (errors[i] - gains[i - 1] @ weights[i - 1])


def _scatter_grads(models, latents, weights, readout, target, slope):
    batch_size = latents[0].shape[1]
    _, gains = _errors(latents, weights, slope)

    for i in range(len(weights)):
        grad = -(gains[i].transpose(1, 2) @ latents[i + 1]) / batch_size
        for fold, model in enumerate(models):
            model.layers[i].W.grad = grad[fold]

    output_error = latents[-1] @ readout.transpose(1, 2) - target
    grad = (output_error.transpose(1, 2) @ latents[-1]) / batch_size
    for fold, model in enumerate(models):
        model.readout.weight.grad = grad[fold]


def fit_epoch(models, optimizers, loaders, eta_infer, T_infer, slope, device):
    """Train every model for one epoch, one stacked batch at a time.

    ``models``, ``optimizers`` and ``loaders`` are parallel lists describing the
    folds still training. The loaders must yield the same number of batches,
    which 5-fold cross-validation over MNIST does exactly.
    """
    batch_counts = {len(loader) for loader in loaders}
    if len(batch_counts) != 1:
        raise ValueError(f"folds must have equal batch counts, got {batch_counts}")

    for model in models:
        model.train()
    n_classes = models[0].readout.out_features

    for batches in zip(*loaders):
        x = torch.stack([images.reshape(images.shape[0], -1) for images, _ in batches]).to(device)
        target = torch.stack(
            [F.one_hot(labels.long(), n_classes).float() for _, labels in batches]
        ).to(device)

        with torch.no_grad():
            weights, readout = _stack_weights(models)
            latents = _init_latents(x, weights, slope)
            _relax(latents, weights, readout, target, eta_infer, T_infer, slope)
            _scatter_grads(models, latents, weights, readout, target, slope)

        for optimizer in optimizers:
            optimizer.step()
