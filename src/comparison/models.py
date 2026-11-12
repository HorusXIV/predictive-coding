"""Model and optimizer construction for both arms of the comparison."""

import torch
from torch import nn

from pcn.model import NEGATIVE_SLOPE, PredictiveCodingNetwork


class MLP(nn.Module):
    """Plain feedforward classifier, built the way MNIST MLP baselines are.

    ``dims`` is read the same way as in ``PredictiveCodingNetwork``: the input
    width first, then each hidden width, with ``output_dim`` appended as the
    final layer. ReLU between hidden layers, dropout after each of them, bias
    terms throughout.

    Those last three are where this parts company with the PCN, which uses a
    leaky slope its relaxation depends on and carries no biases. The two models
    no longer match layer for layer; they are each set up to do their best, and
    ``n_params`` is logged per trial so the size difference stays visible in
    the results (see :mod:`comparison.config`).
    """

    def __init__(self, dims, output_dim, dropout=0.0):
        super().__init__()
        widths = list(dims) + [output_dim]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.input_dim = dims[0]

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], self.input_dim))


def build_pcn(dims, output_dim, negative_slope=NEGATIVE_SLOPE):
    # activation_deriv must match activation, so both close over the same slope.
    return PredictiveCodingNetwork(
        dims=list(dims),
        output_dim=output_dim,
        activation=nn.LeakyReLU(negative_slope),
        activation_deriv=lambda a: torch.where(a > 0, 1.0, negative_slope),
    )


def build_mlp(dims, output_dim, dropout=0.0):
    return MLP(dims, output_dim, dropout)


# SGD is deliberately plain, without momentum: this is the SGD the grid search
# scores, and a config's optimizer must mean the same thing in the search and
# in the comparison that inherits its winner.
_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "muon": torch.optim.Muon,
}


def build_optimizer(name, parameters, lr, weight_decay):
    """Look up an optimizer by the name a config carries.

    Unknown names raise rather than falling back to a default, so a typo in a
    config surfaces as an error instead of a run that silently trained with
    something else.

    Muon takes 2D parameters only. That fits the PCN, whose layers and readout
    are all bias-free matrices, but it rejects the MLP's bias vectors, so an
    MLP config asking for Muon fails here rather than halfway through a run.
    """
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name!r}; expected one of {sorted(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](parameters, lr=lr, weight_decay=weight_decay)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
