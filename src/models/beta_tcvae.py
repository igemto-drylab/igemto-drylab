from typing import List, Tuple

import torch
from torch import Tensor, nn


class BetaTCVAE(nn.Module):
    """A VAE using a feed-forward network for its encoder, and an RNN for its
    decoder. Depending on the mode, this VAE is a beta-VAE or a beta-TCVAE.
    """

    def __init__(self,
                 seq_len: int,
                 encoder_dims: List[int],
                 latent_dim: int,
                 gru_hidden_size: int,
                 gru_layer_size: int,
                 beta: int = 1,
                 beta_tcvae: bool = False) -> None:
        """
        Initializes this VAE with the specified parameters.

        :param beta_tcvae: if true, then this VAE is a beta-TCVAE with
                           parameters alpha = 1, beta = <beta>, and gamma = 1.
        """
        super(BetaTCVAE, self).__init__()

        # Attributes
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_layer_size = gru_layer_size
        self.beta = beta
        self.beta_tcvae = beta_tcvae

        # Create the feed-forward encoder
        encoder_dims = [21 * seq_len] + encoder_dims
        encoder_layers = []
        for prev, curr in zip(encoder_dims, encoder_dims[1:]):
            encoder_layers.extend([
                nn.Linear(prev, curr),
                nn.ReLU()
            ])

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(encoder_dims[-1], latent_dim)

        # Create the RNN decoder
        self.decoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_layer_size,
            batch_first=True
        )
        self.fc_recons = nn.Sequential(
            nn.Linear(gru_hidden_size, 21),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes <x> through the encoder, and returns a tuple of the mean
        and log variance in this order. <x> is expected to be a Tensor with
        size (batch, seq_len, 21).
        """
        x = torch.flatten(x, start_dim=1)  # reshape

        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Samples from a normal distribution with mean <mu> and log variance
        <log_var> using the reparameterization trick, i.e. z = μ + σε where
        ε ~ N(0, I).
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.rand_like(std)
        return mu + epsilon * std

    def decode(self, z: Tensor) -> Tensor:
        """Decodes latent variables <z> by decoding from the RNN decoder
        <seq_len> number of times, and returns the reconstruction.
        """
        batch_size = z.size(0)

        z_seq = torch.stack([z] * self.seq_len, dim=1)
        h_0 = torch.zeros(self.gru_layer_size, batch_size, self.gru_hidden_size)
        output, _ = self.decoder(z_seq, h_0)

        return self.fc_recons(output)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Feeds inputs <x> through the VAE, and returns a tuple containing
        the reconstruction, mean, and log variance in this order. <x> is
        expected to be a Tensor with size (batch, seq_len, 21) and the
        returned reconstruction will be of the same dimensions.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, n: int = 1) -> Tensor:
        """Samples <n> times from this VAE, and returns the newly generated
        proteins, as a tensor of size (n, seq_len, 21)
        """
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)

    def compute_loss(self, x: Tensor, recon: Tensor, mu: Tensor,
                     log_var: Tensor) -> Tensor:
        """Returns the calculated loss between inputs <x> and reconstructions
        <recon>, given means <mu> and log variances <log_var>.
        """

        if not self.beta_tcvae:  # standard beta-VAE loss
            x = x.view(-1, len(self.alphabet))
            target = torch.argmax(x, 1)

            recon = recon.view(-1, len(self.alphabet))

            recon_loss = nn.CrossEntropyLoss()
            kld = 0.5 * torch.mean(log_var.exp() + mu.pow(2) - 1. - log_var)

            return recon_loss(recon, target) + self.beta * kld

        # TODO: otherwise, we calculate the beta-TCVAE loss
        raise NotImplementedError
