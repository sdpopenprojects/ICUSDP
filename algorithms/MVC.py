import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

from algorithms.AutoEncoder import Autoencoder
from algorithms.labelingCluster import labelCluster_v2
from algorithms.Classifiers import OptimizeTree


class InterpretableMVC:
    def __init__(self, n_views, n_clusters, input_dims, latent_dims, lambda_=0.1,
                 max_depth=10, min_samples_split=10):
        super().__init__()

        self.labels_ = None
        self.n_views = n_views
        self.n_clusters = n_clusters
        self.latent_dims = latent_dims
        self.lambda_ = lambda_

        if np.ndim(latent_dims) == 1:
            self.autoencoders = [Autoencoder(input_dims[v], latent_dims) for v in range(n_views)]
        else:
            self.autoencoders = [Autoencoder(input_dims[v], latent_dims[v]) for v in range(n_views)]

        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        # self.tree = DecisionTreeClassifier()

        # self.centers = [torch.randn(n_clusters, latent_dim[:,-1]) for _ in range(n_views)]

    def pretrain_autoencoders(self, X_views, epochs=200, batch_size=256, lr=0.001):
        for v in range(self.n_views):
            optimizer = optim.Adam(self.autoencoders[v].parameters(), lr=lr)
            dataset = TensorDataset(torch.FloatTensor(X_views[v]))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                total_loss = 0
                for batch in loader:
                    optimizer.zero_grad()
                    _, recon = self.autoencoders[v](batch[0])
                    loss = nn.MSELoss()(recon, batch[0])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # print(f'View {v}, Epoch {epoch}, Loss: {total_loss / len(loader)}')

        # Get latent features
        Z = []
        with torch.no_grad():
            for v in range(self.n_views):
                z_v, _ = self.autoencoders[v](torch.FloatTensor(X_views[v]))
                Z.append(z_v.numpy())

        # Concatenate features and get pseudo-labels
        Z_concat = np.concatenate(Z, axis=1)
        kmeans = KMeans(n_clusters=self.n_clusters)
        pseudo_labels = kmeans.fit_predict(Z_concat)

        pseudo_labels = labelCluster_v2(pseudo_labels)

        return pseudo_labels

    def initialize_tree(self, X_views, pseudo_labels):
        # Build decision tree
        X_concat = np.concatenate(X_views, axis=1)

        # tree = OptimizeTree(X_concat, pseudo_labels)
        # self.tree = tree.optimizer()

        self.tree.fit(X_concat, pseudo_labels)
        Y_ = self.tree.predict(X_concat)

        return Y_

    def optimize_features(self, X_views, pseudo_labels, epochs=200, lr=0.001):
        for v in range(self.n_views):
            optimizer = optim.Adam(self.autoencoders[v].parameters(), lr=lr)
            dataset = TensorDataset(torch.FloatTensor(X_views[v]), torch.LongTensor(pseudo_labels))
            loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

            # 计算初始聚类中心
            with torch.no_grad():
                for batch in loader:
                    z_v, _ = self.autoencoders[v](batch[0])
                    kmeans = KMeans(n_clusters=self.n_clusters).fit(z_v.numpy())
                    centers = torch.FloatTensor(kmeans.cluster_centers_)

                    # Y = np.eye(self.n_clusters)[batch[1]]
                    # Y_tensor = torch.FloatTensor(Y)

            for epoch in range(epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    z_v, recon = self.autoencoders[v](batch[0])

                    # Reconstruction loss
                    recon_loss = nn.MSELoss()(recon, batch[0])

                    # Cluster assignment loss
                    dist = torch.cdist(z_v, centers) # self.centers[v]
                    q = 1.0 / (1.0 + dist ** 2)
                    q = q / torch.sum(q, dim=1, keepdim=True)
                    # q = q / q.sum(1, keepdim=True)

                    # ce_loss = -torch.mean(torch.sum(Y_tensor * torch.log(q + 1e-10), dim=1))
                    # Cross entropy loss
                    ce_loss = nn.CrossEntropyLoss()(q, batch[1])

                    # Total loss
                    loss = recon_loss + self.lambda_ * ce_loss

                    loss.backward()
                    optimizer.step()

                    if epoch % 20 == 0:
                        with torch.no_grad():
                            z_v, _ = self.autoencoders[v](batch[0])
                            kmeans = KMeans(n_clusters=self.n_clusters).fit(z_v.numpy())
                            centers = torch.FloatTensor(kmeans.cluster_centers_)

                # print(f'Optimizing View {v}, Epoch {epoch}, Loss: {loss.item()}')

        # Get updated latent features
        Z = []
        with torch.no_grad():
            for v in range(self.n_views):
                z_v, _ = self.autoencoders[v](torch.FloatTensor(X_views[v]))
                Z.append(z_v.numpy())

        # Get new pseudo-labels
        Z_concat = np.concatenate(Z, axis=1)
        kmeans = KMeans(n_clusters=self.n_clusters)
        optimize_pseudo_labels = kmeans.fit_predict(Z_concat)

        optimize_pseudo_labels = labelCluster_v2(optimize_pseudo_labels)

        return optimize_pseudo_labels

    def optimize_tree(self, X_views, pseudo_labels):
        # # Get updated latent features
        # Z = []
        # with torch.no_grad():
        #     for v in range(self.n_views):
        #         z_v, _ = self.autoencoders[v](torch.FloatTensor(X_views[v]))
        #         Z.append(z_v.numpy())
        #
        # # Get new pseudo-labels
        # Z_concat = np.concatenate(Z, axis=1)
        # kmeans = KMeans(n_clusters=self.n_clusters)
        # new_labels = kmeans.fit_predict(Z_concat)

        # Update decision tree
        X_concat = np.concatenate(X_views, axis=1)

        tree = OptimizeTree(X_concat, pseudo_labels)
        self.tree = tree.optimizer()

        self.tree.fit(X_concat, pseudo_labels)
        optimize_Y_ = self.tree.predict(X_concat)

        return optimize_Y_

    def fit(self, X_views, max_iters=10):
        # Step 1: Pretrain autoencoders
        pseudo_labels = self.pretrain_autoencoders(X_views)

        # Step 2: Initialize decision tree
        Y_ = self.initialize_tree(X_views, pseudo_labels)

        # Step 3: Joint optimization
        for _ in range(max_iters):
            # Optimize feature representations
            optimize_pseudo_labels = self.optimize_features(X_views, Y_)

            # Optimize decision tree
            optimize_Y_ = self.optimize_tree(X_views, optimize_pseudo_labels)

            Y_ = optimize_Y_

        self.labels_ = optimize_Y_

        return self

    def predict(self, X_views):
        X_concat = np.concatenate(X_views, axis=1)
        return self.tree.predict(X_concat)
