import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.tree import export_text

from algorithms.SC import SC
from algorithms.Classifiers2 import CLF
from algorithms.labelingCluster import labelCluster


# ==================== 统一的神经网络表征学习器组件 ====================
class UniversalAutoencoder(nn.Module):
    """
    集成了 VAE, AE, DAE 功能的通用自编码器
    无论哪种模式，统一通过 forward 返回 4 个参数 (x_recon, mu, logvar, z)
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, mode='VAE'):
        super(UniversalAutoencoder, self).__init__()
        self.mode = mode.upper()

        # 1. 构建 Encoder 架构
        encoder_layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(curr_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            curr_dim = h_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)

        # 2. 潜在特征映射层
        if self.mode == 'VAE':
            self.fc_mu = nn.Linear(curr_dim, latent_dim)
            self.fc_logvar = nn.Linear(curr_dim, latent_dim)
        else:
            # AE 和 DAE 只需要一个确定的特征隐层
            self.fc_z = nn.Linear(curr_dim, latent_dim)

        # 3. 构建 Decoder 架构
        decoder_layers = []
        curr_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(curr_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            curr_dim = h_dim
        decoder_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 如果是 DAE 模式，前向传播时隐式添加高斯度量噪声 (均值0, 标准差0.1)
        if self.mode == 'DAE' and self.training:
            noise = torch.randn_like(x) * 0.1
            x_input = x + noise
        else:
            x_input = x

        # 编码阶段
        features = self.encoder_backbone(x_input)

        if self.mode == 'VAE':
            mu = self.fc_mu(features)
            logvar = self.fc_logvar(features)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar, z
        else:
            # AE 和 DAE 模式：产生确定性表征
            z = self.fc_z(features)
            # 产生和 z 形状一致的虚拟零矩阵做占位符，保持 4 值无痛解包，避免任何 Unpack 错误
            dummy_mu = torch.zeros_like(z)
            dummy_logvar = torch.zeros_like(z)
            return self.decoder(z), dummy_mu, dummy_logvar, z


# ==================== 主实验集成框架 ====================
class InterpretableClustering:
    """
    模块化无监督缺陷预测闭环框架
    - 支持特征学习器 (feature_learner) 开关: 'VAE', 'AE', 'DAE' 自由切换
    - 支持聚类器 (cluster_type) 和解释器 (clf) 保持不变
    """

    def __init__(self, n_clusters=2, hidden_dims=[128, 64], latent_dim=32, clf='DT', cluster_type='sc',
                 feature_learner='VAE', e1_epochs=200, e2_epochs=200, lambda_ce=0.1, lambda_kld=0.01, random_state=42):

        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.clf = clf
        self.cluster_type = cluster_type.lower()
        self.feature_learner = feature_learner.upper()  # 统一大写 ['VAE', 'AE', 'DAE']

        self.e1_epochs = e1_epochs
        self.e2_epochs = e2_epochs
        self.lambda_ce = lambda_ce
        self.lambda_kld = lambda_kld
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.vae = None  # 兼容旧名称，承载通用自编码器
        self.tree = None
        self.cluster_centers = None
        self.pseudo_labels = None
        self.feature_importances_ = None

    def _get_cluster_labels(self, z_np):
        if self.cluster_type == 'sc':
            return SC(z_np)
        elif self.cluster_type == 'gmm':
            model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
            return model.fit_predict(z_np)
        elif self.cluster_type == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=self.n_clusters)
            return model.fit_predict(z_np)
        elif self.cluster_type == 'kmedoids':
            model = KMedoids(n_clusters=self.n_clusters, random_state=self.random_state)
            return model.fit_predict(z_np)
        elif self.cluster_type == 'kmeans':
            model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            return model.fit_predict(z_np)
        else:
            raise ValueError(f"不支持的聚类方法: {self.cluster_type}")

    def vae_loss(self, x_recon, x, mu, logvar):
        # 1. 基础重构误差 (AE, DAE, VAE 共有)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # 2. 条件自适应计算概率约束 (只有 VAE 计算 KLD 惩罚)
        if self.feature_learner == 'VAE':
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + self.lambda_kld * kld_loss
        else:
            kld_loss = torch.tensor(0.0)
            total_loss = recon_loss

        return total_loss, recon_loss, kld_loss

    def _pretrain_vae(self, X):
        X_tensor = torch.FloatTensor(X)
        input_dim = X.shape[1]

        # 根据配置参数动态配置神经网络内核
        self.vae = UniversalAutoencoder(input_dim, self.hidden_dims, self.latent_dim, mode=self.feature_learner)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)

        self.vae.train()
        for epoch in range(self.e1_epochs):
            optimizer.zero_grad()
            # 完美的 4 值解包，绝对不会报 ValueError
            x_recon, mu, logvar, z = self.vae(X_tensor)
            total_loss, _, _ = self.vae_loss(x_recon, X_tensor, mu, logvar)
            total_loss.backward()
            optimizer.step()

    def _extract_embedded_features(self, X):
        X_tensor = torch.FloatTensor(X)
        self.vae.eval()
        with torch.no_grad():
            _, _, _, z = self.vae(X_tensor)
            Z = z.numpy()
        return Z

    def _construct_initial_tree(self, X_original, pseudo_labels):
        current_clf = CLF(classifier=self.clf)
        self.tree = current_clf.getCLF()
        self.tree.fit(X_original, pseudo_labels)
        self.feature_importances_ = self.tree.feature_importances_

    def _optimize_feature_representation(self, X, tree_labels):
        X_tensor = torch.FloatTensor(X)
        tree_labels_onehot = np.eye(self.n_clusters)[tree_labels]
        labels_tensor = torch.FloatTensor(tree_labels_onehot)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        self.vae.train()
        for epoch in range(self.e2_epochs):
            optimizer.zero_grad()
            x_recon, mu, logvar, z = self.vae(X_tensor)
            z_np = z.detach().numpy()

            vae_loss, _, _ = self.vae_loss(x_recon, X_tensor, mu, logvar)
            current_labels = self._get_cluster_labels(z_np)

            soft_assignment = np.eye(self.n_clusters)[current_labels]
            soft_assignment_tensor = torch.FloatTensor(soft_assignment)
            ce_loss = F.cross_entropy(soft_assignment_tensor, labels_tensor)

            total_loss = vae_loss + self.lambda_ce * ce_loss
            total_loss.backward()
            optimizer.step()

    def _optimize_tree(self, X_original, pseudo_labels):
        current_clf = CLF(classifier=self.clf)
        self.tree = current_clf.getCLF()
        self.tree.fit(X_original, pseudo_labels)
        self.feature_importances_ = self.tree.feature_importances_

    def fit_predict(self, X, max_iters=10):
        X_normalized = X
        self._pretrain_vae(X_normalized)

        Z = self._extract_embedded_features(X_normalized)
        initial_pseudo_labels = self._get_cluster_labels(Z)
        initial_pseudo_labels = labelCluster(X_normalized, initial_pseudo_labels)
        self.pseudo_labels = initial_pseudo_labels

        self._construct_initial_tree(X_normalized, initial_pseudo_labels)

        try:
            safe_iters = int(max_iters.item()) if hasattr(max_iters, 'item') else int(max_iters)
        except:
            safe_iters = 10

        for iteration in range(safe_iters):
            if self.lambda_ce > 0:
                tree_labels = self.tree.predict(X_normalized)
                self._optimize_feature_representation(X_normalized, tree_labels)

            Z = self._extract_embedded_features(X_normalized)
            new_pseudo_labels = self._get_cluster_labels(Z)
            new_pseudo_labels = labelCluster(X_normalized, new_pseudo_labels)

            self._optimize_tree(X_normalized, new_pseudo_labels)

        final_labels = self.tree.predict(X_normalized)
        self.pseudo_labels = final_labels
        return final_labels

    def get_interpretability_report(self, feature_names):
        if self.tree is None:
            raise ValueError("Please call fit_predict() first to train the model")

        feature_importances_ = self.tree.feature_importances_
        feature_importances_ = pandas.DataFrame(feature_importances_)
        feature_importances_.index = feature_names
        sorted_feature_importances = feature_importances_.sort_values(ascending=False, by=0)

        if self.clf == 'DT':
            n_nodes = self.tree.tree_.node_count
            max_depth = self.tree.get_depth()

            if hasattr(feature_names, 'tolist'):
                feature_names = feature_names.tolist()

            tree_rules = export_text(self.tree, feature_names=feature_names)

            report = {
                'tree': self.tree,
                'feature_importances': sorted_feature_importances,
                'pseudo_labels': self.pseudo_labels,
                'latent_dim': self.latent_dim,
                'tree_max_depth': max_depth,
                'tree_n_nodes': n_nodes,
                'tree_rules': tree_rules
            }
        else:
            report = {
                'tree': self.tree,
                'feature_importances': sorted_feature_importances,
                'pseudo_labels': self.pseudo_labels,
                'latent_dim': self.latent_dim
            }
        return report

    def visualize_decision_rules(self, feature_names=None):
        if self.tree is None:
            raise ValueError("Please call fit_predict() first to train the model")

        if feature_names is None:
            n_features = len(self.feature_importances_)
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        tree_rules = export_text(self.tree, feature_names=feature_names)
        return tree_rules

    def generate_latent_samples(self, X, n_samples=10):
        if not self.vae:
            raise ValueError("Model not trained. Please call fit_predict() first.")
        X_sample = X[:n_samples]
        X_tensor = torch.FloatTensor(X_sample)
        self.vae.eval()
        with torch.no_grad():
            _, _, _, z = self.vae(X_tensor)
            latent_samples = [z.numpy()]
        return latent_samples