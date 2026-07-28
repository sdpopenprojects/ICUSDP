import numpy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class Autoencoder(nn.Module):
    """
    Autoencoder model for learning embedded features from each view

    Parameters:
    input_dim: Input dimension size
    hidden_dims: List of hidden layer dimensions
    """

    def __init__(self, input_dim, hidden_dims):
        super(Autoencoder, self).__init__()

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for i in range(len(hidden_dims) - 1):
            hidden_dim = hidden_dims[i]
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=0.2))
            prev_dim = hidden_dim

        # last layer
        encoder_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (symmetric structure)
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = hidden_dims_rev[0]
        for i in range(1, len(hidden_dims_rev)):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dims_rev[i]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(p=0.2))
            prev_dim = hidden_dims_rev[i]

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass through autoencoder

        Args:
            x: Input tensor

        Returns:
            z: Embedded features
            x_recon: Reconstructed input
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for learning embedded features from each view

    Parameters:
    input_dim: Input dimension size
    hidden_dims: List of hidden layer dimensions
    latent_dim: Dimension of the latent space
    """

    def __init__(self, input_dim, hidden_dims=[128, 64], latent_dim=32, epochs=200, lr=1e-3, device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.vae = []  # VAE

        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for i in range(len(hidden_dims) - 1):
            hidden_dim = hidden_dims[i]
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            # encoder_layers.append(nn.Dropout(p=0.2))
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
            # decoder_layers.append(nn.Dropout(p=0.2))
            prev_dim = hidden_dim

        # output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # decoder_layers.append(nn.Sigmoid())

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

    def vae_loss(self, x_recon, x, mu, logvar):
        """
        Calculate VAE loss (reconstruction + KL divergence)

        Args:
            x_recon: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            total_loss: Combined VAE loss
            recon_loss: Reconstruction loss
            kld_loss: KL divergence loss
        """
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + 0.01 * kld_loss

        return total_loss, recon_loss, kld_loss

    def train_vae(self, X):
        X = torch.FloatTensor(X).to(self.device)
        # X = X.float().to(self.device)

        # Create VAE for current data
        input_dim = X.shape[1]
        hidden_dims = self.hidden_dims
        latent_dim = self.latent_dim

        vae = VariationalAutoencoder(input_dim, hidden_dims, latent_dim).to(self.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=self.lr)

        # Training loop
        vae.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass through VAE
            x_recon, mu, logvar, z = vae(X)

            # Calculate VAE loss
            total_loss, recon_loss, kld_loss = self.vae_loss(x_recon, X, mu, logvar)

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Print progress
            # if epoch % 50 == 0:
            #     print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
            #           f"Recon: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f}")

        self.vae = vae

    def extract_embedded_features(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)

        vae = self.vae
        vae.eval()
        # Extract features without gradient computation
        with torch.no_grad():
            x_recon, _, _, z = vae(X_tensor)

            X_recon = x_recon.numpy()
            Z = z.numpy()

        return X_recon, Z


class LR(nn.Module):
    def __init__(self, input_dim=20, num_classes=2, epochs=200, lr=1e-3, device='cpu'):
        super(LR, self).__init__()
        self.model = []

        self.epochs = epochs
        self.lr = lr
        self.device = device

        self.linear = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out).squeeze(1)
        # out = self.relu(out).squeeze(1)
        return out

    def fit_model(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        input_dim = X.shape[1]
        num_classes = len(numpy.unique(y))
        model = LR(input_dim, num_classes).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for epoch in range(self.epochs):
            y_hat = model(X_tensor)
            loss = F.cross_entropy(y_hat, y_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress
            # if (epoch+1) % 100 == 0:
            #     print(f"Epoch {epoch+1}, Total Loss: {loss.item():.4f}")

        self.model = model

        # for epoch in range(self.epochs):
        #     allLoss = 0.0
        #     model.train()
        #     for X, y in train_loader:
        #         X = X.float().to(self.device)
        #         y = y.long().to(self.device)
        #
        #         optimizer.zero_grad()
        #         y_hat = model(X)
        #         # loss = F.cross_entropy(y_hat, y)
        #         loss = criterion(y_hat, y)
        #         loss.backward()
        #         optimizer.step()
        #
        #         with torch.no_grad():
        #             allLoss += float(loss.sum())

        # if (epoch+1) % 10 == 0:
        #     print("Epoch {epoch + 1}/{self.epochs}: train loss {:.5f}".format(epoch, allLoss))

    def predict_model(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)

        model = self.model
        model.eval()
        with torch.no_grad():
            y_hat = model(X_tensor)
            _, y_pred = torch.max(y_hat.data, 1)
            # y_pred = numpy.argmax(y_hat.numpy(), axis=1)

            y_pred = y_pred.numpy()

            return y_pred

        # with torch.no_grad():
        #     preds = []
        #     trues = []
        #     for X, y in test_loader:
        #         X = X.float().to(self.device)
        #         y = y.long().to(self.device)
        #         y_hat = model(X)
        #         # y_pred = np.argmax(y_hat.cpu().numpy(), axis=1)
        #         _, y_pred = torch.max(y_hat.data, 1)
        #         preds.extend(y_pred.cpu().numpy())
        #         trues.extend(y.cpu().numpy())
        #
        #     return preds, trues

    def get_interpretability_report(self, feature_names):
        """
          Generate interpretability analysis report
          Returns:
             report: Dictionary containing interpretability metrics
        """

        if self.model is None:
            raise ValueError("Please call fit_model() first to train the model")

        coef_dict = extract_coefficients(self.model, feature_names)[0]
        coef_dict.pop('intercept')
        feature_importances_ = pandas.DataFrame([coef_dict]).T
        sorted_feature_importances = feature_importances_.sort_values(ascending=False, by=0)

        report = {
            'model': self.model,
            'feature_importances': sorted_feature_importances,
        }

        return report


def extract_coefficients(model, feature_names=None):
    """
    提取逻辑回归模型的系数
    """
    # 获取权重和偏置
    weights = model.linear.weight.detach().numpy().flatten()
    bias = model.linear.bias.detach().numpy().flatten()[0]

    # 创建系数字典
    coef_dict = {}

    if feature_names is not None:
        for i, name in enumerate(feature_names):
            coef_dict[name] = weights[i]
    else:
        for i in range(len(weights)):
            coef_dict[f'feature_{i}'] = weights[i]

    coef_dict['intercept'] = bias

    return coef_dict, weights, bias


def plot_feature_importance(coef_dict, top_k=None):
    """
    可视化特征重要性
    """
    # 排除截距项
    feature_coefs = {k: v for k, v in coef_dict.items() if k != 'intercept'}

    # 按绝对值排序
    sorted_features = sorted(feature_coefs.items(),
                             key=lambda x: abs(x[1]),
                             reverse=True)

    if top_k:
        sorted_features = sorted_features[:top_k]

    features, importances = zip(*sorted_features)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance (Logistic Regression Coefficients)')

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width / 2, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='center', va='center')

    plt.tight_layout()
    plt.show()


def predict_proba_with_interpretation(model, X, feature_names):
    """
    预测概率并提供可解释性分析
    """
    model.eval()
    with torch.no_grad():
        # 获取线性部分的输出（logits）
        logits = model.linear(X)
        probabilities = model(X)

    # 提取系数
    coef_dict, weights, bias = extract_coefficients(model, feature_names)

    # 计算每个样本的特征贡献
    feature_contributions = X * weights

    return {
        'probabilities': probabilities.numpy(),
        'logits': logits.numpy(),
        'feature_contributions': feature_contributions.numpy(),
        'coefficients': coef_dict
    }

