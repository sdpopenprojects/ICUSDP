import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from algorithms.AutoEncoder import Autoencoder
from algorithms.labelingCluster import labelCluster_v2
# from algorithms.Classifiers import OptimizeTree


class InterpretableMVC:
    """
    Interpretable Multi-View Clustering Algorithm

    Based on the paper: "Interpretable multi-view clustering" by Jiang et al.
    This algorithm combines deep autoencoders with decision trees for interpretable clustering.

    Key features:
    - Learns embedded features for each view using autoencoders
    - Uses decision trees for interpretable clustering rules
    - Alternating optimization between feature learning and tree refinement
    - Provides feature importance and view importance analysis
    """

    def __init__(self, n_clusters, hidden_dims,
                 e1_epochs=200, e2_epochs=200, max_depth=10,
                 min_samples_split=10, lambda_ce=0.1, random_state=42):
        """
        Initialize the interpretable multi-view clustering model

        Args:
            n_clusters: Number of clusters
            hidden_dims: List of hidden layer dimensions for autoencoders
            e1_epochs: Number of epochs for pre-training phase
            e2_epochs: Number of epochs for optimization phase
            max_depth: Maximum depth of decision tree
            min_samples_split: Minimum samples required to split a node
            lambda_ce: Weight for cross-entropy loss in total loss function
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.e1_epochs = e1_epochs
        self.e2_epochs = e2_epochs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ce = lambda_ce
        self.random_state = random_state

        # Set random seeds for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        # Model components
        self.autoencoders = []  # List of autoencoders for each view
        self.decision_tree = None  # Interpretable decision tree
        self.cluster_centers = None  # Cluster centers for each view
        self.pseudo_labels = None  # Pseudo-labels for clustering
        self.feature_importances_ = None  # Feature importance scores

    def _pretrain_autoencoders(self, X_list):
        """
        Pre-train autoencoders for each view independently

        Args:
            X_list: List of multi-view data arrays
        """
        self.autoencoders = []

        for v, X in enumerate(X_list):
            # print(f"Pre-training autoencoder for view { v +1}...")

            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X)

            # Create autoencoder for current view
            input_dim = X.shape[1]

            if np.ndim(self.hidden_dims) == 1:
                hidden_dims = self.hidden_dims
            else:
                hidden_dims = self.hidden_dims[v]

            autoencoder = Autoencoder(input_dim, hidden_dims)
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

            # Training loop
            autoencoder.train()
            for epoch in range(self.e1_epochs):
                optimizer.zero_grad()

                # Forward pass
                z, x_recon = autoencoder(X_tensor)

                # Compute reconstruction loss
                recon_loss = F.mse_loss(x_recon, X_tensor)

                # Backward pass and optimization
                recon_loss.backward()
                optimizer.step()

                # Print progress
                # if epoch % 50 == 0:
                #     print(f"View { v +1}, Epoch {epoch}, Recon Loss: {recon_loss.item():.4f}")

            self.autoencoders.append(autoencoder)

    def _extract_embedded_features(self, X_list):
        """
        Extract embedded features from pre-trained autoencoders

        Args:
            X_list: List of multi-view data arrays

        Returns:
            Z_list: List of embedded features for each view
            Z_concat: Concatenated embedded features from all views
        """
        Z_list = []

        for v, (X, autoencoder) in enumerate(zip(X_list, self.autoencoders)):
            X_tensor = torch.FloatTensor(X)
            autoencoder.eval()

            # Extract features without gradient computation
            with torch.no_grad():
                z, _ = autoencoder(X_tensor)
                Z_list.append(z.numpy())

        # Concatenate features from all views
        Z_concat = np.concatenate(Z_list, axis=1)
        return Z_list, Z_concat

    def _construct_initial_decision_tree(self, X_original, pseudo_labels):
        """
        Construct initial decision tree using pseudo-labels

        Args:
            X_original: Original feature matrix (concatenated views)
            pseudo_labels: Initial cluster labels from k-means
        """
        self.decision_tree = DecisionTreeClassifier(
            # max_depth=self.max_depth,
            # min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )

        # Train decision tree on original features with pseudo-labels
        self.decision_tree.fit(X_original, pseudo_labels)

        # Extract feature importance scores
        self.feature_importances_ = self.decision_tree.feature_importances_

    def _compute_soft_assignments(self, Z_list, cluster_centers_list):
        """
        Compute soft cluster assignments using Student's t-distribution

        Args:
            Z_list: List of embedded features for each view
            cluster_centers_list: List of cluster centers for each view

        Returns:
            soft_assignments: List of soft assignment matrices
        """
        soft_assignments = []

        for Z, centers in zip(Z_list, cluster_centers_list):
            # Calculate distances to cluster centers
            distances = np.zeros((Z.shape[0], len(centers)))
            for i, center in enumerate(centers):
                distances[:, i] = np.linalg.norm(Z - center, axis=1)

            # Compute soft assignments using t-distribution
            numerator = (1 + distances**2 )**(-1)
            denominator = np.sum(numerator, axis=1, keepdims=True)
            soft_assignment = numerator / denominator

            soft_assignments.append(soft_assignment)

        return soft_assignments

    def _optimize_feature_representation(self, X_list, tree_labels):
        """
        Optimize feature representations while keeping decision tree fixed

        Args:
            X_list: List of multi-view data arrays
            tree_labels: Cluster labels from current decision tree
        """
        # Convert tree labels to one-hot encoding
        tree_labels_onehot = np.eye(self.n_clusters)[tree_labels]

        for v, (X, autoencoder) in enumerate(zip(X_list, self.autoencoders)):
            # print(f"Optimizing feature representation for view { v +1}...")

            X_tensor = torch.FloatTensor(X)
            labels_tensor = torch.FloatTensor(tree_labels_onehot)

            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

            autoencoder.train()
            for epoch in range(self.e2_epochs):
                optimizer.zero_grad()

                # Forward pass
                z, x_recon = autoencoder(X_tensor)

                # Reconstruction loss
                recon_loss = F.mse_loss(x_recon, X_tensor)

                # Clustering loss (cross-entropy with tree labels)
                z_np = z.detach().numpy()
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

                soft_assignment_tensor = torch.FloatTensor(soft_assignment)
                ce_loss = F.cross_entropy(soft_assignment_tensor, labels_tensor)

                # Total loss (reconstruction + clustering)
                total_loss = recon_loss + self.lambda_ce * ce_loss

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()

                # Print training progress
                # if epoch % 100 == 0:
                #     print(f"View { v +1}, Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
                #           f"Recon: {recon_loss.item():.4f}, CE: {ce_loss.item():.4f}")

    def _optimize_decision_tree(self, X_original, pseudo_labels):
        """
        Optimize decision tree while keeping feature representations fixed

        Args:
            X_original: Original feature matrix
            pseudo_labels: Updated pseudo-labels from k-means
        """
        # Rebuild decision tree with updated pseudo-labels

        self.decision_tree = DecisionTreeClassifier(
            # max_depth=self.max_depth,
            # min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )

        # tree = OptimizeTree(X_original, pseudo_labels)
        # self.decision_tree = tree.optimizer()

        self.decision_tree.fit(X_original, pseudo_labels)
        self.feature_importances_ = self.decision_tree.feature_importances_

    def fit_predict(self, X_list, max_iters=10):
        """
        Train the model and return cluster assignments

        Args:
            X_list: List of multi-view data arrays
            max_iters: Maximum number of alternating optimization iterations

        Returns:
            final_labels: Final cluster assignments
        """
        # n_samples = X_list[0].shape[0]
        # n_views = len(X_list)

        # print("=" * 60)
        # print("Interpretable Multi-View Clustering")
        # print(f"Samples: {n_samples}, Views: {n_views}, Clusters: {self.n_clusters}")
        # print("=" * 60)

        # Step 1: Data preprocessing
        # print("Step 1: Data preprocessing...")
        # X_normalized_list = []
        # scalers = []
        #
        # for X in X_list:
        #     scaler = StandardScaler()
        #     X_normalized = scaler.fit_transform(X)
        #     X_normalized_list.append(X_normalized)
        #     scalers.append(scaler)

        X_normalized_list = X_list

        # Step 2: Pre-train autoencoders
        # print("\nStep 2: Pre-training autoencoders...")
        self._pretrain_autoencoders(X_normalized_list)

        # Step 3: Extract features and generate initial pseudo-labels
        # print("\nStep 3: Feature extraction and initial clustering...")
        Z_list, Z_concat = self._extract_embedded_features(X_normalized_list)

        # Initial k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        initial_pseudo_labels = kmeans.fit_predict(Z_concat)

        initial_pseudo_labels = labelCluster_v2(initial_pseudo_labels)
        self.pseudo_labels = initial_pseudo_labels

        # Step 4: Build initial decision tree
        # print("\nStep 4: Building initial decision tree...")
        X_original_concat = np.concatenate(X_normalized_list, axis=1)
        self._construct_initial_decision_tree(X_original_concat, initial_pseudo_labels)

        # Step 5: Alternating optimization
        # print("\nStep 5: Alternating optimization...")
        # current_pseudo_labels = initial_pseudo_labels.copy()

        for iteration in range(max_iters):
            # print(f"\n--- Iteration {iteration + 1} ---")

            # Phase 1: Optimize feature representations (fix tree)
            # print("a) Optimizing feature representations...")
            tree_labels = self.decision_tree.predict(X_original_concat)
            self._optimize_feature_representation(X_normalized_list, tree_labels)

            # Update features and pseudo-labels
            Z_list, Z_concat = self._extract_embedded_features(X_normalized_list)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            new_pseudo_labels = kmeans.fit_predict(Z_concat)

            new_pseudo_labels = labelCluster_v2(new_pseudo_labels)

            # Phase 2: Optimize decision tree (fix features)
            # print("b) Optimizing decision tree...")
            self._optimize_decision_tree(X_original_concat, new_pseudo_labels)

            # Check convergence
            # label_change = np.sum(new_pseudo_labels != current_pseudo_labels) / n_samples
            # print(f"Label change rate: {label_change:.4f}")

            # if label_change < 0.001:  # Convergence condition
            #     print(f"Converged at iteration {iteration + 1}")
            #     break
            #
            # current_pseudo_labels = new_pseudo_labels

        # Final cluster assignments
        final_labels = self.decision_tree.predict(X_original_concat)
        self.pseudo_labels = final_labels

        return final_labels

    def get_interpretability_report(self):
        """
        Generate interpretability analysis report

        Returns:
            report: Dictionary containing interpretability metrics
        """
        if self.decision_tree is None:
            raise ValueError("Please call fit_predict() first to train the model")

        # Tree complexity metrics
        n_nodes = self.decision_tree.tree_.node_count
        max_depth = self.decision_tree.get_depth()

        # View importance analysis
        # n_features_per_view = [autoencoder.encoder[0].in_features for autoencoder in self.autoencoders]
        #
        # view_importances = []
        # start_idx = 0
        # for n_features in n_features_per_view:
        #     end_idx = start_idx + n_features
        #     view_importance = np.sum(self.feature_importances_[start_idx:end_idx])
        #     view_importances.append(view_importance)
        #     start_idx = end_idx

        report = {
            'tree_max_depth': max_depth,
            'tree_n_nodes': n_nodes,
            # 'view_importances': view_importances,
            'feature_importances': self.feature_importances_,
            'pseudo_labels': self.pseudo_labels
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
        if self.decision_tree is None:
            raise ValueError("Please call fit_predict() first to train the model")

        if feature_names is None:
            n_features = len(self.feature_importances_)
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        tree_rules = export_text(self.decision_tree, feature_names=feature_names)
        # print("Decision Tree Rules:")
        # print(tree_rules)

        return tree_rules