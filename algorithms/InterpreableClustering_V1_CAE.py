import numpy as np
import pandas
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.cluster import KMeans
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier

from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering


try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

from algorithms.SC import SC
from algorithms.Classifiers2 import CLF
from algorithms.labelingCluster import labelCluster


# ============================================================
# Contrastive Autoencoder
# ============================================================

class ContrastiveAutoencoder(nn.Module):
    """
    Contrastive Autoencoder (CAE)

    Encoder:
        X -> hidden layers -> latent representation Z

    Decoder:
        Z -> hidden layers -> reconstructed X

    The encoder is trained using:
        1. Reconstruction loss
        2. Contrastive loss
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(ContrastiveAutoencoder, self).__init__()

        # -------------------------
        # Encoder
        # -------------------------
        encoder_layers = []
        last_dim = input_dim

        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = h_dim

        self.encoder_backbone = nn.Sequential(*encoder_layers)

        # Projection head
        self.projection_head = nn.Linear(last_dim, latent_dim)

        # -------------------------
        # Decoder
        # -------------------------
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
        x_recon = self.decoder(z)

        return z, x_recon


# ============================================================
# Interpretable Clustering with CAE
# ============================================================

class InterpretableClustering:

    def __init__(
        self,
        n_clusters=2,
        hidden_dims=[128, 64],
        latent_dim=32,
        clf='DT',
        cluster_type='kmeans',
        epochs=200,
        e2_epochs=200,
        lr=1e-3,
        device='cpu',
        lambda_recon=0.7,
        lambda_ce=0.1,
        margin=1.0,
        random_state=42
    ):

        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.clf = clf
        self.clf_type = clf

        self.cluster_type = cluster_type.lower()

        # CAE parameters
        self.epochs = epochs
        self.e2_epochs = e2_epochs
        self.lr = lr

        self.device = device

        # CAE contrastive loss weight
        self.lambda_recon = lambda_recon

        # ICC weight
        self.lambda_ce = lambda_ce

        # Contrastive margin
        self.margin = margin

        self.random_state = random_state

        # Model components
        self.encoder = None
        self.tree = None

        self.pseudo_labels = None
        self.feature_importances_ = None

    # ========================================================
    # Contrastive Loss
    # ========================================================

    def _contrastive_loss(self, z1, z2, y_pair):
        """
        Contrastive loss.

        y_pair = 1:
            positive pair -> minimize distance

        y_pair = 0:
            negative pair -> enforce margin
        """

        dist = torch.norm(z1 - z2, dim=1)

        pos_loss = y_pair * (dist ** 2)

        neg_loss = (
            (1 - y_pair)
            * torch.pow(
                torch.clamp(self.margin - dist, min=0.0),
                2
            )
        )

        return torch.mean(pos_loss + neg_loss)

    # ========================================================
    # Global Density Pair Construction
    # ========================================================

    def _build_global_density_pairs(self, X_tensor):
        """
        Construct global self-supervised pairs.

        For each sample:
            1. Randomly select another sample.
            2. Calculate its distance.
            3. Use the sample-wise median distance as
               the density boundary.
            4. Assign positive/negative pair labels.
        """

        n_samples = X_tensor.size(0)

        # -----------------------------------------
        # 1. Random pairing
        # -----------------------------------------

        rand_idx = torch.randperm(n_samples).to(self.device)

        xi = X_tensor
        xj = X_tensor[rand_idx]

        # -----------------------------------------
        # 2. Global pairwise distance matrix
        # -----------------------------------------

        dist_matrix = torch.cdist(
            X_tensor,
            X_tensor,
            p=2.0
        )

        # -----------------------------------------
        # 3. Median distance as density threshold
        # -----------------------------------------

        thresholds = torch.quantile(
            dist_matrix,
            0.5,
            dim=1
        )

        # -----------------------------------------
        # 4. Pair distance
        # -----------------------------------------

        pair_dists = torch.norm(
            xi - xj,
            dim=1
        )

        y_pair = (
            pair_dists < thresholds
        ).float()

        return xi, xj, y_pair

    # ========================================================
    # CAE Pre-training
    # ========================================================

    def _train_encoder(self, X_tensor):

        input_dim = X_tensor.shape[1]

        X_tensor = X_tensor.to(self.device)

        # Create CAE
        self.encoder = ContrastiveAutoencoder(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.lr
        )

        criterion_recon = nn.MSELoss()

        # -----------------------------------------
        # CAE pre-training
        # -----------------------------------------

        for epoch in range(self.epochs):

            self.encoder.train()

            # Build global pairs
            xi, xj, y_pair = self._build_global_density_pairs(
                X_tensor
            )

            # Forward
            zi, xi_recon = self.encoder(xi)
            zj, _ = self.encoder(xj)

            # Contrastive loss
            loss_contrastive = self._contrastive_loss(
                zi,
                zj,
                y_pair
            )

            # Reconstruction loss
            loss_recon = criterion_recon(
                xi_recon,
                xi
            )

            # CAE loss
            loss = (
                loss_recon
                + self.lambda_recon * loss_contrastive
            )

            # Gradient update
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    # ========================================================
    # Extract latent features
    # ========================================================

    def _extract_embedded_features(self, X):

        if isinstance(X, pandas.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        X_tensor = torch.FloatTensor(
            X_arr
        ).to(self.device)

        self.encoder.eval()

        with torch.no_grad():

            z, _ = self.encoder(
                X_tensor
            )

            Z = z.cpu().numpy()

        return Z

    # ========================================================
    # Cluster router
    # ========================================================

    def _get_cluster_labels(self, z_np):

        if self.cluster_type == 'kmeans':

            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )

            return model.fit_predict(z_np)

        elif self.cluster_type == 'sc':

            return SC(z_np)

        elif self.cluster_type == 'gmm':

            model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state
            )

            return model.fit_predict(z_np)

        elif self.cluster_type == 'agglomerative':

            model = AgglomerativeClustering(
                n_clusters=self.n_clusters
            )

            return model.fit_predict(z_np)

        elif self.cluster_type == 'kmedoids':

            if KMedoids is None:

                raise ImportError(
                    "no scikit-learn-extra ，"
                    "'pip install scikit-learn-extra'。"
                )

            model = KMedoids(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )

            return model.fit_predict(z_np)

        else:

            raise ValueError(
                f"no: {self.cluster_type}，"
                f"please "
                f"['kmeans', 'sc', 'gmm', "
                f"'agglomerative', 'kmedoids'] 。"
            )

    # ========================================================
    # Initial Decision Tree
    # ========================================================

    def _construct_initial_tree(
        self,
        X_original,
        pseudo_labels
    ):
        """
        Train the decision tree on ORIGINAL software features.

        This is consistent with the original ICUSDP implementation.
        """

        current_clf = CLF(
            classifier=self.clf
        )

        self.tree = current_clf.getCLF()

        # IMPORTANT:
        # DT uses original features X,
        # NOT latent features Z.
        self.tree.fit(
            X_original,
            pseudo_labels
        )

        self.feature_importances_ = (
            self.tree.feature_importances_
        )

    # ========================================================
    # CAE + ICC Feature Optimization
    # ========================================================

    def _optimize_feature_representation(
        self,
        X,
        tree_labels
    ):
        """
        Optimize CAE feature representations while
        keeping the current decision tree fixed.

        The structure follows the original ICUSDP implementation:

            CAE loss
                +
            ICC loss
                ↓
            total loss
                ↓
            CAE update
        """

        X_tensor = torch.FloatTensor(
            X
        ).to(self.device)

        # -----------------------------------------
        # Tree labels -> one-hot
        # -----------------------------------------

        tree_labels_onehot = np.eye(
            self.n_clusters
        )[tree_labels]

        labels_tensor = torch.FloatTensor(
            tree_labels_onehot
        ).to(self.device)

        # -----------------------------------------
        # Optimizer
        # -----------------------------------------

        cae = self.encoder

        optimizer = torch.optim.Adam(
            cae.parameters(),
            lr=self.lr
        )

        criterion_recon = nn.MSELoss()

        cae.train()

        # -----------------------------------------
        # Iterative CAE optimization
        # -----------------------------------------

        for epoch in range(self.e2_epochs):

            optimizer.zero_grad()

            # CAE forward
            z, x_recon = cae(
                X_tensor
            )

            # -------------------------------------
            # CAE reconstruction loss
            # -------------------------------------

            recon_loss = criterion_recon(
                x_recon,
                X_tensor
            )

            # -------------------------------------
            # Current latent representation
            # -------------------------------------

            z_np = z.detach().cpu().numpy()

            # -------------------------------------
            # Current clustering
            # -------------------------------------

            current_labels = (
                self._get_cluster_labels(z_np)
            )

            # -------------------------------------
            # One-hot cluster assignment
            #
            # This intentionally follows the
            # original ICUSDP implementation.
            # -------------------------------------

            cluster_assignment = np.eye(
                self.n_clusters
            )[current_labels]

            cluster_assignment_tensor = torch.FloatTensor(
                cluster_assignment
            ).to(self.device)

            # -------------------------------------
            # ICC / consistency loss
            # -------------------------------------

            ce_loss = F.cross_entropy(
                cluster_assignment_tensor,
                labels_tensor
            )

            # -------------------------------------
            # CAE total loss
            #
            # CAE:
            # reconstruction + contrastive
            #
            # Here we keep the contrastive loss
            # from the CAE feature learner.
            # -------------------------------------

            # Reconstruct current pair structure
            xi, xj, y_pair = (
                self._build_global_density_pairs(
                    X_tensor
                )
            )

            zi, xi_recon = cae(xi)
            zj, _ = cae(xj)

            contrastive_loss = (
                self._contrastive_loss(
                    zi,
                    zj,
                    y_pair
                )
            )

            cae_loss = (
                recon_loss
                + self.lambda_recon * contrastive_loss
            )

            # -------------------------------------
            # Total loss
            # -------------------------------------

            total_loss = (
                cae_loss
                + self.lambda_ce * ce_loss
            )

            # -------------------------------------
            # Backward
            # -------------------------------------

            total_loss.backward()

            optimizer.step()

    # ========================================================
    # Optimize Decision Tree
    # ========================================================

    def _optimize_tree(
        self,
        X_original,
        pseudo_labels
    ):
        """
        Rebuild decision tree using ORIGINAL
        software features.
        """

        current_clf = CLF(
            classifier=self.clf
        )

        self.tree = current_clf.getCLF()

        # IMPORTANT:
        # Keep interpreter unchanged:
        # Decision Tree uses X_original.
        self.tree.fit(
            X_original,
            pseudo_labels
        )

        self.feature_importances_ = (
            self.tree.feature_importances_
        )

    # ========================================================
    # Main training procedure
    # ========================================================

    def fit_predict(
        self,
        X,
        max_iters=10,
        cluster_method='kmeans'
    ):
        """
        Main CAE-based ICUSDP procedure.

        Flow:

            X
            ↓
            CAE
            ↓
            Z
            ↓
            K-means
            ↓
            Pseudo-label
            ↓
            Decision Tree(X)
            ↓
            Iterative refinement
        """

        # -----------------------------------------
        # Convert input
        # -----------------------------------------

        if isinstance(X, pandas.DataFrame):

            X_normalized = X.values

        else:

            X_normalized = X

        X_tensor = torch.tensor(
            X_normalized,
            dtype=torch.float32
        )

        # -----------------------------------------
        # Step 1:
        # Pre-train CAE
        # -----------------------------------------

        self._train_encoder(
            X_tensor
        )

        # -----------------------------------------
        # Step 2:
        # Extract latent features
        # -----------------------------------------

        Z = self._extract_embedded_features(
            X_normalized
        )

        # -----------------------------------------
        # Step 3:
        # Initial clustering
        # -----------------------------------------

        # Use the requested clustering method.
        # In the component experiment,
        # this should be K-means.
        self.cluster_type = (
            cluster_method.lower()
        )

        initial_pseudo_labels = (
            self._get_cluster_labels(Z)
        )

        # -----------------------------------------
        # Label orientation
        # -----------------------------------------

        initial_pseudo_labels = labelCluster(
            X_normalized,
            initial_pseudo_labels
        )

        self.pseudo_labels = (
            initial_pseudo_labels
        )

        # -----------------------------------------
        # Step 4:
        # Initial Decision Tree
        #
        # IMPORTANT:
        # Tree uses original X.
        # -----------------------------------------

        self._construct_initial_tree(
            X_normalized,
            initial_pseudo_labels
        )

        # -----------------------------------------
        # Step 5:
        # Alternating optimization
        # -----------------------------------------

        try:

            if hasattr(max_iters, 'item'):

                safe_iters = int(
                    max_iters.item()
                )

            else:

                safe_iters = int(
                    max_iters
                )

        except Exception:

            safe_iters = 10

        for iteration in range(
            safe_iters
        ):

            # -------------------------------------
            # Phase 1:
            # Optimize CAE feature learner
            # while keeping tree fixed
            # -------------------------------------

            if self.lambda_ce > 0:

                tree_labels = (
                    self.tree.predict(
                        X_normalized
                    )
                )

                self._optimize_feature_representation(
                    X_normalized,
                    tree_labels
                )

            # -------------------------------------
            # Phase 2:
            # Extract new latent representation
            # -------------------------------------

            Z = self._extract_embedded_features(
                X_normalized
            )

            # -------------------------------------
            # Phase 3:
            # Re-cluster
            # -------------------------------------

            new_pseudo_labels = (
                self._get_cluster_labels(Z)
            )

            new_pseudo_labels = labelCluster(
                X_normalized,
                new_pseudo_labels
            )

            # -------------------------------------
            # Phase 4:
            # Update Decision Tree
            #
            # Again, use ORIGINAL X.
            # -------------------------------------

            self._optimize_tree(
                X_normalized,
                new_pseudo_labels
            )

        # -----------------------------------------
        # Final prediction
        # -----------------------------------------

        final_labels = (
            self.tree.predict(
                X_normalized
            )
        )

        self.pseudo_labels = (
            final_labels
        )

        return final_labels

    # ========================================================
    # Prediction
    # ========================================================

    def predict(self, X):

        if isinstance(
            X,
            pandas.DataFrame
        ):

            X_arr = X.values

        else:

            X_arr = X

        # Extract latent representation
        Z = self._extract_embedded_features(
            X_arr
        )

        # DT predicts using original X
        return self.tree.predict(
            X_arr
        )

    # ========================================================
    # Interpretability Report
    # ========================================================

    def get_report(
        self,
        feature_names
    ):

        if self.tree is None:

            raise ValueError(
                "请先调用 fit_predict()"
            )


        # Feature importance


        feature_importances_ = (
            self.tree.feature_importances_
        )

        fi = pandas.DataFrame(
            feature_importances_,
            index=feature_names,
            columns=[0]
        )

        sorted_fi = fi.sort_values(
            ascending=False,
            by=0
        )

        # -----------------------------------------
        # Report
        # -----------------------------------------

        report = {

            'feature_importances':
                sorted_fi,

            'pseudo_labels':
                self.pseudo_labels,

            'latent_dim':
                self.latent_dim
        }

        # -----------------------------------------
        # Decision tree information
        # -----------------------------------------

        if self.clf_type == 'DT':

            report[
                'tree_rules'
            ] = export_text(
                self.tree,
                feature_names=list(
                    feature_names
                )
            )

            report[
                'tree_depth'
            ] = self.tree.get_depth()

            report[
                'tree_n_nodes'
            ] = self.tree.tree_.node_count

        return report

    # ========================================================
    # Latent samples
    # ========================================================

    def generate_latent_samples(
        self,
        X,
        n_samples=10
    ):
        """
        Generate latent representations
        from the trained CAE.
        """

        if self.encoder is None:

            raise ValueError(
                "CAE not trained. "
                "Please call fit_predict() first."
            )

        if isinstance(
            X,
            pandas.DataFrame
        ):

            X_arr = X.values

        else:

            X_arr = X

        X_sample = X_arr[
            :n_samples
        ]

        X_tensor = torch.FloatTensor(
            X_sample
        ).to(self.device)

        self.encoder.eval()

        with torch.no_grad():

            z, _ = self.encoder(
                X_tensor
            )

        return [
            z.cpu().numpy()
        ]