"""The abstract generative and oracle classes that should be inherited in
order to use the CbAS algorithm.
"""

from typing import Optional, Tuple

from torch import Tensor, nn
from torch.utils import data


class Generator(nn.Module):
    """An abstract generative model used in the CbAS.
    """

    def sample(self, n: int) -> Tuple[Optional[Tensor], Tensor]:
        """Samples <n> times from this model, returning a tuple of:
            1. the sampled latent variables, or None if this model is not
               a latent variable model.
            2. the generated samples.
        Recall that the first dimension of the returned Tensors is <n>.
        """
        raise NotImplementedError

    def train_self(self, train_data: data.Dataset, weights: Tensor,
                   epochs: int, batch_size: int, lr: float,
                   valid_data: Optional[data.Dataset] = None,
                   verbose: bool = True) -> None:
        """Trains this model on the weighted data set, provided by
        <train_data> and <weights>.
        """
        raise NotImplementedError

    def gen_prob(self, x_set: Tensor, z_set: Optional[Tensor]) -> Tensor:
        """Returns a tensor, where each element is p(x, z). Actually,
        p(x|z) can also be returned, as the p(z) cancels during CbAS. If this
        model is not a latent variable model, then p(x) is returned.
        """
        raise NotImplementedError


class Oracle:
    """A abstract oracle(s) used in CbAS.
    """

    def get_scores(self, x: Tensor) -> Tensor:
        """Returns a tensor of predicted scores for the input <x>.
        """
        raise NotImplementedError

    def prob_desideratum(self, x: Tensor, gamma: float) -> float:
        """Returns the probability the desired desideratum is satisfied, i.e.,
        P(S|x). This is given by 1 - CDF(x, gamma) in maximization problems.
        """
        raise NotImplementedError
