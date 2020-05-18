from typing import List, Optional, Tuple

from torch import Tensor, nn
from torch.utils import data


class Generator(nn.Module):

    def sample(self, n: int) -> Tuple[Optional[Tensor], Tensor]:
        raise NotImplementedError

    def train_self(self, train_data: data.Dataset, weights: List[float],
                   epochs: int, batch_size: int, lr: float,
                   valid_data: Optional[data.Dataset] = None,
                   verbose: bool = True) -> None:
        raise NotImplementedError

    def gen_prob(self, x_set: Tensor, z_set: Optional[Tensor]) -> Tensor:
        raise NotImplementedError


class Oracle:

    def get_scores(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def cdf_gamma(self, x: Tensor, gamma: float) -> Tensor:
        raise NotImplementedError
