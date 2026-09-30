import numpy as np
import pandas
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import export_text

from algorithms.VAE import VariationalAutoencoder
# from algorithms.Classifiers import CLF, OptimizingCLF
from algorithms.Classifiers2 import CLF
from algorithms.labelingCluster import labelCluster


class InterpretableClustering:
    """
    Interpretable Clustering Algorithm with Variational Autoencoders

    Key features:
    - Learns embedded features using VAEs
    - Uses decision trees for interpretable clustering rules
    - Alternating optimization between feature learning and tree refinement
    - Provides feature importance analysis
    """
    def __init__(self, n_clusters=2, hidden_dims=[128, 64], latent_dim=32, clf='DT',
                 e1_epochs=200, e2_epochs=200, lambda_ce=0.1, lambda_kld=0.01, random_state=42):
        """
        Initialize the interpretable clustering model with VAEs

        Args:
            n_clusters: Number of clusters
            hidden_dims: List of hidden layer dimensions for VAEs
            latent_dim: Dimension of the latent space in VAEs
            e1_epochs: Number of epochs for pre-training phase
            e2_epochs: Number of epochs for optimization phase
            lambda_ce: Weight for cross-entropy loss in total loss function
            lambda_kld: Weight for KL divergence loss in VAE training
            random_state: Random seed for reproducibility
        """

        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.clf = clf

        self.e1_epochs = e1_epochs
        self.e2_epochs = e2_epochs
        self.lambda_ce = lambda_ce
        self.lambda_kld = lambda_kld
        self.random_state = random_state

        # Set random seeds for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        # Model components
        self.vae = []  # VAE
        self.tree = None  # Interpretable decision tree
        self.cluster_centers = None  # Cluster centers for each view
        self.pseudo_labels = None  # Pseudo-labels for clustering
        self.feature_importances_ = None  # Feature importance scores

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
        total_loss = recon_loss + self.lambda_kld * kld_loss

        return total_loss, recon_loss, kld_loss

    def _pretrain_vae(self, X):
        """
        Pre-train VAE

        Args:
            X: data arrays
        """

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X)

        # Create VAE for current data
        input_dim = X.shape[1]
        hidden_dims = self.hidden_dims
        latent_dim = self.latent_dim

        vae = VariationalAutoencoder(input_dim, hidden_dims, latent_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

        # Training loop
        vae.train()
        for epoch in range(self.e1_epochs):
            optimizer.zero_grad()

            # Forward pass through VAE
            x_recon, mu, logvar, z = vae(X_tensor)

            # Calculate VAE loss
            total_loss, recon_loss, kld_loss = self.vae_loss(x_recon, X_tensor, mu, logvar)

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Print progress
            # if epoch % 50 == 0:
            #     print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
            #           f"Recon: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f}")

        self.vae = vae

    def _extract_embedded_features(self, X):
        """
        Extract embedded features from pre-trained VAEs

        Args:
            X: data arrays

        Returns:
            Z: embedded features for current data
        """

        X_tensor = torch.FloatTensor(X)

        vae = self.vae
        vae.eval()
        # Extract features without gradient computation
        with torch.no_grad():
            _, _, _, z = vae(X_tensor)
            Z = z.numpy()

        return Z

    def _construct_initial_tree(self, X_original, pseudo_labels):
        """
        Construct initial decision tree using pseudo-labels

        Args:
            X_original: Original feature matrix
            pseudo_labels: Initial cluster labels from k-means
        """

        # self.tree = DecisionTreeClassifier(random_state=self.random_state)
        # current_clf = CLF(classifier=self.clf, randseed=self.random_state)
        current_clf = CLF(classifier=self.clf)
        self.tree = current_clf.getCLF()

        # # Train decision tree on original features with pseudo-labels
        self.tree.fit(X_original, pseudo_labels)

        # Extract feature importance scores
        self.feature_importances_ = self.tree.feature_importances_

    def _optimize_feature_representation(self, X, tree_labels):
        """
        Optimize feature representations while keeping decision tree fixed

        Args:
            X: data arrays
            tree_labels: Cluster labels from current decision tree
        """

        X_tensor = torch.FloatTensor(X)

        # Convert tree labels to one-hot encoding
        tree_labels_onehot = np.eye(self.n_clusters)[tree_labels]
        labels_tensor = torch.FloatTensor(tree_labels_onehot)

        vae = self.vae
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

        vae.train()
        for epoch in range(self.e2_epochs):
            optimizer.zero_grad()

            # Forward pass through VAE
            x_recon, mu, logvar, z = vae(X_tensor)
            z_np = z.detach().numpy()

            # VAE loss (reconstruction + KL divergence)
            vae_loss, recon_loss, kld_loss = self.vae_loss(x_recon, X_tensor, mu, logvar)

            # Clustering loss (cross-entropy with tree labels)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            current_labels = kmeans.fit_predict(z_np)
            centers = kmeans.cluster_centers_

            # Compute soft assignments for current features
            distances = np.zeros((z_np.shape[0], len(centers)))
            for i, center in enumerate(centers):
                distances[:, i] = np.linalg.norm(z_np - center, axis=1)

            numerator = (1 + distances**2 )**(-1)
            denominator = np.sum(numerator, axis=1, keepdims=True)
            soft_assignment = numerator / denominator

            # Using Spectrum clustering
            # current_labels = SC(z_np)
            # soft_assignment = np.eye(self.n_clusters)[current_labels]

            # Using Agglomerative Clustering
            # clustering = AgglomerativeClustering().fix(z_np)
            # current_labels = clustering.labels_
            # soft_assignment = np.eye(self.n_clusters)[current_labels]

            soft_assignment_tensor = torch.FloatTensor(soft_assignment)
            ce_loss = F.cross_entropy(soft_assignment_tensor, labels_tensor)

            # Total loss (VAE + clustering)
            total_loss = vae_loss + self.lambda_ce * ce_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Print training progress
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
            #           f"VAE: {vae_loss.item():.4f}, CE: {ce_loss.item():.4f}")

    def _optimize_tree(self, X_original, pseudo_labels):
        """
        Optimize decision tree while keeping feature representations fixed

        Args:
            X_original: Original feature matrix
            pseudo_labels: Updated pseudo-labels from k-means
        """

        # Rebuild decision tree with updated pseudo-labels
        # self.decision_tree = DecisionTreeClassifier(random_state=self.random_state)
        # current_clf = CLF(classifier=self.clf, randseed=self.random_state)
        current_clf = CLF(classifier=self.clf)
        self.tree = current_clf.getCLF()

        # Optimizing Classifiers using Optuna
        # current_clf = OptimizingCLF(X_original, pseudo_labels, classifier='DT', randseed=self.random_state)
        # self.tree = current_clf.getOptCLF()

        self.tree.fit(X_original, pseudo_labels)
        self.feature_importances_ = self.tree.feature_importances_

    def fit_predict(self, X, max_iters=10):
        """
        Train the model and return cluster assignments

        Args:
            X: data arrays
            max_iters: Maximum number of alternating optimization iterations

        Returns:
            final_labels: Final cluster assignments
        """

        # n_samples = X.shape[0]
        # print("=" * 60)
        # print("Interpretable Clustering with VAEs")
        # print(f"Samples: {n_samples}, Clusters: {self.n_clusters}")
        # print("=" * 60)

        # Step 1: Data preprocessing
        # print("Step 1: Data preprocessing...")
        # scaler = StandardScaler()
        # X_normalized = scaler.fit_transform(X)
        X_normalized = X

        # Step 2: Pre-train VAEs
        # print("\nStep 2: Pre-training VAEs...")
        self._pretrain_vae(X_normalized)

        # Step 3: Extract features and generate initial pseudo-labels
        # print("\nStep 3: Feature extraction and initial clustering...")
        Z = self._extract_embedded_features(X_normalized)

        # Initial k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        initial_pseudo_labels = kmeans.fit_predict(Z)
        initial_pseudo_labels = labelCluster(X_normalized, initial_pseudo_labels)

        # Initial Spectrum clustering
        # initial_pseudo_labels = SC(Z)

        # Initial Agglomerative Clustering
        # clustering = AgglomerativeClustering().fix(Z)
        # initial_pseudo_labels = clustering.labels_
        # initial_pseudo_labels = labelCluster(X_normalized, initial_pseudo_labels)

        self.pseudo_labels = initial_pseudo_labels

        # Step 4: Build initial decision tree
        # print("\nStep 4: Building initial decision tree...")
        self._construct_initial_tree(X_normalized, initial_pseudo_labels)

        # Step 5: Alternating optimization
        # print("\nStep 5: Alternating optimization...")
        for iteration in range(max_iters):
            # print(f"\n--- Iteration {iteration + 1} ---")

            # Phase 1: Optimize feature representations (fix tree)
            # print("a) Optimizing feature representations...")
            tree_labels = self.tree.predict(X_normalized)
            # tree_labels = self.randomforest.predict(X_normalized)

            self._optimize_feature_representation(X_normalized, tree_labels)

            # Update features and pseudo-labels
            Z = self._extract_embedded_features(X_normalized)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            new_pseudo_labels = kmeans.fit_predict(Z)
            new_pseudo_labels = labelCluster(X_normalized, new_pseudo_labels)

            # new_pseudo_labels = SC(Z)

            # Agglomerative Clustering
            # clustering = AgglomerativeClustering().fix(Z)
            # new_pseudo_labels = clustering.labels_
            # new_pseudo_labels = labelCluster(X_normalized, new_pseudo_labels)

            # Phase 2: Optimize decision tree (fix features)
            # print("b) Optimizing decision tree...")
            self._optimize_tree(X_normalized, new_pseudo_labels)

        # Final cluster assignments
        final_labels = self.tree.predict(X_normalized)
        # final_labels = self.randomforest.predict(X_normalized)
        self.pseudo_labels = final_labels

        # print("\n" + "=" * 60)
        # print("Clustering completed successfully!")
        # print("=" * 60)

        return final_labels

    def get_interpretability_report(self, feature_names):
        """
        Generate interpretability analysis report

        Returns:
            report: Dictionary containing interpretability metrics
        """
        if self.tree is None:
            raise ValueError("Please call fit_predict() first to train the model")

        # sort feature importance
        feature_importances_ = self.tree.feature_importances_
        feature_importances_ = pandas.DataFrame(feature_importances_)
        feature_importances_.index = feature_names
        sorted_feature_importances = feature_importances_.sort_values(ascending=False,by=0)

        if self.clf == 'DT':
            # Tree complexity metrics
            n_nodes = self.tree.tree_.node_count
            max_depth = self.tree.get_depth()

            # select top-k features for output tree rules
            # top_importances = sorted_feature_importances[sorted_feature_importances[0] >= threshold]
            # top_features = top_importances.index
            #
            # mask = [fea in top_features for fea in feature_names]
            # custom_names = [feature_names[i] if m else "ignore" for i, m in enumerate(mask)]
            # tree_rules = export_text(self.tree, feature_names=custom_names)

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
        """
        Visualize decision tree rules for interpretability

        Args:
            feature_names: List of feature names for better readability

        Returns:
            tree_rules: Text representation of decision tree
        """
        if self.tree is None:
            raise ValueError("Please call fit_predict() first to train the model")

        if feature_names is None:
            n_features = len(self.feature_importances_)
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        tree_rules = export_text(self.tree, feature_names=feature_names)
        # print("Decision Tree Rules:")
        # print(tree_rules)

        return tree_rules

    def generate_latent_samples(self, X, n_samples=10):
        """
        Generate samples from the latent space of VAEs

        Args:
            X: data arrays
            n_samples: Number of samples to generate

        Returns:
            latent_samples: Generated latent samples
        """
        if not self.vae:
            raise ValueError("VAEs not trained. Please call fit_predict() first.")

        # Sampling
        X_sample = X[:n_samples]
        X_tensor = torch.FloatTensor(X_sample)

        latent_samples = []
        vae = self.vae
        vae.eval()
        with torch.no_grad():
            _, mu, logvar, z = vae(X_tensor)
            latent_samples.append(z.numpy())

        return latent_samples
