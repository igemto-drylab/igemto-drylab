from typing import Optional

import numpy as np
import torch
from torch.utils import data

from src.models.cbas_models import Generator, Oracle


def cbas_maximize(oracle: Oracle, prior: Generator, gen: Generator,
                  q: float, m: int, iters: int,
                  train_data: data.Dataset,
                  epochs: int, batch_size: int, lr: float,
                  cutoff: float = 1e-6,
                  valid_data: Optional[data.Dataset] = None,
                  verbose: bool = False) -> None:
    """Runs the conditioning by adaptive sampling algorithm, in the context
    of a maximization design problem.
    """

    # starting with t = 0
    # train the prior
    weights = [1.] * len(train_data)
    prior.train_self(train_data, weights, epochs, batch_size, lr,
                     valid_data, verbose)

    # load the generative model with the prior's parameters
    gen.load_state_dict(prior.state_dict(), strict=True)

    gamma = float('-inf')

    for t in range(1, iters + 1):

        # take samples and calculate scores.
        Z_t, X_t = prior.sample(m)
        scores = oracle.get_scores(X_t)

        # calculate q-th percentile of scores
        gamma_t = np.percentile(scores.numpy(), q * 100.).item()
        gamma = max(gamma, gamma_t)

        # calculate weights
        weights = prior.gen_prob(X_t, Z_t).div(gen.gen_prob(X_t, Z_t))
        weights = oracle.prob_desideratum(X_t, gamma) * weights

        # As in the original CbAS code, we cut off elements of X_t, that have
        # corresponding weights < cutoff. I convert to numpy since I don't see
        # how to fully write this in PyTorch.
        cutoff_idx = torch.where(weights < cutoff)[0].numpy()

        X_t_numpy = np.delete(X_t.numpy(), cutoff_idx, axis=0)
        weights_numpy = np.delete(weights.numpy(), cutoff_idx, axis=0)

        X_t = torch.from_numpy(X_t_numpy)  # convert back to PyTorch
        weights = torch.from_numpy(weights_numpy)

        # train generative model
        gen.train_self(data.TensorDataset(X_t), weights.tolist(),
                       epochs, batch_size, lr,
                       valid_data, verbose)
