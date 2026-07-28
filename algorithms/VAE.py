import torch
from torch import nn


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for learning embedded features from each view

    Parameters:
    input_dim: Input dimension size
    hidden_dims: List of hidden layer dimensions
    latent_dim: Dimension of the latent space
    """

    def __init__(self, input_dim=20, hidden_dims=[128, 64], latent_dim=32):
        super(VariationalAutoencoder, self).__init__()
        self.vae = []  # VAE
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for i in range(len(hidden_dims) - 1):
            hidden_dim = hidden_dims[i]
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # last layer
        encoder_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Mean and log variance layers for latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims[::-1]:  # Reverse order
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var)

        Args:
            mu: Mean from the encoder's latent space
            logvar: Log variance from the encoder's latent space

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through VAE

        Args:
            x: Input tensor

        Returns:
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent vector
        """
        # Encode
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar, z
