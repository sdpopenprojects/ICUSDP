import numpy as np
import pandas
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier

# 导入 sklearn-extra 中的 KMedoids，并做异常处理预防未安装环境
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

from algorithms.SC import SC
from algorithms.Classifiers2 import CLF
from algorithms.labelingCluster import labelCluster
# 彻底移除 DataLoader 和 TensorDataset 导入
from sklearn.tree import DecisionTreeClassifier


class ContrastiveAutoencoder(nn.Module):
    """
    集成了Encoder（全局拓扑对比学习）与Decoder（全局重构原始JIRA特征流形）
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(ContrastiveAutoencoder, self).__init__()

        # 编码器部分 (Encoder Backbone)
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = h_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        self.projection_head = nn.Linear(last_dim, latent_dim)

        # 解码器部分 (Decoder Backbone)
        decoder_layers = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            last_dim = h_dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        feat = self.encoder_backbone(x)
        z = self.projection_head(feat)
        return z, self.decoder(z)


class InterpretableClustering:
    def __init__(self, n_clusters=2, hidden_dims=[128, 64], latent_dim=32, clf='DT',
                 epochs=200, lr=1e-3, device='cpu', lambda_recon=0.7, margin=1.0, random_state=42):
        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.clf_type = clf
        self.clf = clf
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.lambda_recon = lambda_recon
        self.margin = margin
        self.random_state = random_state

        self.encoder = None
        self.tree = None
        self.pseudo_labels = None
        self.feature_importances_ = None

    def _contrastive_loss(self, z1, z2, y_pair):
        """基于全局局部密度和欧氏距离的 Contrastive Loss"""
        dist = torch.norm(z1 - z2, dim=1)
        pos_loss = y_pair * (dist ** 2)
        neg_loss = (1 - y_pair) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return torch.mean(pos_loss + neg_loss)

    def _build_global_density_pairs(self, X_tensor):
        """
        纯 Full-batch 全局自监督对构造：
        在全量 JIRA 数据流形上为每个样本寻找全局随机对，并基于全局流形密度定义正负样本
        """
        n_samples = X_tensor.size(0)

        # 1. 全局随机配对
        rand_idx = torch.randperm(n_samples).to(self.device)
        xi = X_tensor
        xj = X_tensor[rand_idx]

        # 2. 全局距离矩阵 (n_samples, n_samples)用欧氏距离计算，N个样本可得到N×N矩阵
        dist_matrix = torch.cdist(X_tensor, X_tensor, p=2.0)

        # 3. 计算每个样本在整个项目中的全局距离中位数（50%分位数）作为密度边界
        # 它对距离矩阵的每一行（即每个样本）计算其与全量其他样本距离的 50% 分位数（中位数）。
        # 物理意义：每个样本都以自己为中心，将整个项目中最靠近自己的前 50% 的样本圈定为“高密度近邻流形”。
        thresholds = torch.quantile(dist_matrix, 0.5, dim=1)

        # 4. 计算全局随机对的实际距离并赋予标签
        pair_dists = torch.norm(xi - xj, dim=1)    #pair_dists 计算的是当前样本 x_i与其全局随机配对样本x_j之间的实际距离。
        y_pair = (pair_dists < thresholds).float()

        return xi, xj, y_pair

    def _train_encoder(self, X_tensor):
        input_dim = X_tensor.shape[1]

        # 搬运数据到指定设备
        X_tensor = X_tensor.to(self.device)

        self.encoder = ContrastiveAutoencoder(input_dim, self.hidden_dims, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        criterion_recon = nn.MSELoss()

        # 彻底移除 DataLoader 循环，直接执行全局 Epoch 迭代
        for epoch in range(self.epochs):
            self.encoder.train()

            # --- 每一轮都基于最新的全局流形动态构造对 ---
            xi, xj, y_pair = self._build_global_density_pairs(X_tensor)

            # 纯 Full-batch 前向传播（整个矩阵一次性输入）
            zi, xi_recon = self.encoder(xi)
            zj, _ = self.encoder(xj)

            # 计算联合损失
            loss_contrastive = self._contrastive_loss(zi, zj, y_pair)
            loss_recon = criterion_recon(xi_recon, xi)
            loss = loss_recon + self.lambda_recon * loss_contrastive

            # 纯 Full-batch 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fit_predict(self, X, max_iters=10, cluster_method='kmeans'):
        if isinstance(X, pandas.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)

        # 1. 训练全局自编码器
        self._train_encoder(X_tensor)

        # 2. 提取隐空间特征
        self.encoder.eval()
        with torch.no_grad():
            latent_features, _ = self.encoder(X_tensor.to(self.device))
            latent_features = latent_features.cpu().numpy()

        # # 3. 聚类
        # if cluster_method.lower() == 'sc':
        #     self.pseudo_labels = SC(latent_features)
        # else:
        #     kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        #     self.pseudo_labels = kmeans.fit_predict(latent_features)

        # 3. 聚类路由器扩展：完美路由 5 种算法
        method = cluster_method.lower()
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.pseudo_labels = kmeans.fit_predict(latent_features)

        elif method == 'sc':
            # 优先调用你本地导入的专用算法 algorithms.SC
            self.pseudo_labels = SC(latent_features)

        elif method == 'gmm':
            gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
            self.pseudo_labels = gmm.fit_predict(latent_features)

        elif method == 'agglomerative':
            agg = AgglomerativeClustering(n_clusters=self.n_clusters)
            self.pseudo_labels = agg.fit_predict(latent_features)

        elif method == 'kmedoids':
            if KMedoids is None:
                raise ImportError("未检测到 scikit-learn-extra 库，请先运行 'pip install scikit-learn-extra'。")
            kmed = KMedoids(n_clusters=self.n_clusters, random_state=42)
            self.pseudo_labels = kmed.fit_predict(latent_features)

        else:
            raise ValueError(
                f"不支持的聚类方法: {cluster_method}，请确保在 ['kmeans', 'sc', 'gmm', 'agglomerative', 'kmedoids'] 内。")


        # 4. 训练代理树模型
        self.tree = DecisionTreeClassifier(random_state=42)
        self.tree.fit(latent_features, self.pseudo_labels)
        self.feature_importances_ = self.tree.feature_importances_

        return self.pseudo_labels

    def predict(self, X):
        if isinstance(X, pandas.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)

        self.encoder.eval()
        with torch.no_grad():
            latent_features, _ = self.encoder(X_tensor.to(self.device))
            latent_features = latent_features.cpu().numpy()

        return self.tree.predict(latent_features)

    def get_report(self, feature_names):
        if self.tree is None:
            raise ValueError("请先调用 fit_predict()")

        latent_feature_names = [f"z_{i}" for i in range(self.latent_dim)]
        fi = pandas.DataFrame(self.tree.feature_importances_, index=latent_feature_names, columns=[0])
        sorted_fi = fi.sort_values(ascending=False, by=0)

        report = {
            'feature_importances': sorted_fi,
            'pseudo_labels': self.pseudo_labels,
            'latent_dim': self.latent_dim
        }
        if self.clf_type == 'DT':
            report['tree_rules'] = export_text(self.tree, feature_names=latent_feature_names)
            report['tree_depth'] = self.tree.get_depth()
            report['tree_n_nodes'] = self.tree.tree_.node_count
        return report