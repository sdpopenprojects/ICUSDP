import numpy as np
import pandas
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.tree import export_text

# ==== 泛化性引入：新解释器依赖的外部库 ====
from interpret.glassbox import ExplainableBoostingClassifier  # EBM
from wittgenstein import RIPPER  # RIPPER

from algorithms.SC import SC
from algorithms.VAE import VariationalAutoencoder
from algorithms.Classifiers2 import CLF
from algorithms.labelingCluster import labelCluster


# ==================== 泛化组件：软决策树(SDT)的神经网络实现 ====================
class SoftDecisionTreeClassifier(torch.nn.Module):
    """
    可导的浅层软决策树(SDT)适配器，保持与 sklearn 接口一致
    """

    def __init__(self, input_dim, n_classes=2):
        super(SoftDecisionTreeClassifier, self).__init__()
        self.gate = torch.nn.Linear(input_dim, 1)
        self.leaf_prob1 = torch.nn.Parameter(torch.randn(n_classes))
        self.leaf_prob2 = torch.nn.Parameter(torch.randn(n_classes))

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        p1 = F.softmax(self.leaf_prob1, dim=0)
        p2 = F.softmax(self.leaf_prob2, dim=0)
        return g * p1 + (1 - g) * p2

    def fit(self, X, y):
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(50):
            optimizer.zero_grad()
            out = self.forward(X_t)
            loss = F.cross_entropy(torch.log(out + 1e-8), y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X):
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            out = self.forward(X_t)
        return out.numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ==================== 主模型类 ====================
class InterpretableClustering:
    """
    模块化无监督缺陷预测闭环框架 (ICUSDP)
    完美对齐 4-value VAE 输出，全面兼容 visualize_interpreter_rules 接口调用
    """

    def __init__(self, n_clusters=2, hidden_dims=[128, 64], latent_dim=32, clf='DT', cluster_type='sc',
                 e1_epochs=200, e2_epochs=200, lambda_ce=0.1, lambda_kld=0.01, random_state=42):

        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.clf = clf
        self.cluster_type = cluster_type.lower()
        self.interpreter_type = clf.upper()

        self.e1_epochs = e1_epochs
        self.e2_epochs = e2_epochs
        self.lambda_ce = lambda_ce
        self.lambda_kld = lambda_kld
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.vae = []
        self.interpreter = None
        self.tree = None
        self.cluster_centers = None
        self.pseudo_labels = None
        self.feature_importances_ = None

    def _init_interpreter(self, input_dim):
        if self.interpreter_type == 'DT':
            current_clf = CLF(classifier=self.clf)
            self.interpreter = current_clf.getCLF()
        elif self.interpreter_type == 'SDT':
            self.interpreter = SoftDecisionTreeClassifier(input_dim=input_dim, n_classes=self.n_clusters)
        elif self.interpreter_type == 'EBM':
            # self.interpreter = ExplainableBoostingClassifier(random_state=self.random_state)
            # ======= 【完美对齐老师给的 GA²M 参数配置】 =======
            self.interpreter = ExplainableBoostingClassifier(
                interactions=10,  # 最多保留 10 个最重要的交互项
                max_rounds=5000,  # 迭代轮数
                outer_bags=8,  # Bagging 数量，用来稳定特征重要性
                inner_bags=0,  # 加快单轮迭代速度
                learning_rate=0.01,  # 学习率控制
                random_state=self.random_state
            )
        elif self.interpreter_type == 'RIPPER':
            self.interpreter = RIPPER(random_state=self.random_state)
        else:
            current_clf = CLF(classifier=self.clf)
            self.interpreter = current_clf.getCLF()

        self.tree = self.interpreter

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
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.lambda_kld * kld_loss
        return total_loss, recon_loss, kld_loss

    def _pretrain_vae(self, X):
        X_tensor = torch.FloatTensor(X)
        input_dim = X.shape[1]
        self.vae = VariationalAutoencoder(input_dim, self.hidden_dims, self.latent_dim)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)

        self.vae.train()
        for epoch in range(self.e1_epochs):
            optimizer.zero_grad()
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
        if self.interpreter is None:
            self._init_interpreter(input_dim=X_original.shape[1])
        self.tree.fit(X_original, pseudo_labels)
        if hasattr(self.tree, 'feature_importances_'):
            self.feature_importances_ = self.tree.feature_importances_

    def _optimize_feature_representation(self, X, tree_labels):
        X_tensor = torch.FloatTensor(X)
        # tree_labels_onehot = np.eye(self.n_clusters)[tree_labels]
        # 【核心修复】：强转为 int 型，彻底消除 RIPPER 返回布尔数组导致的 IndexError 掩码报错
        tree_labels_int = np.array(tree_labels).astype(int)
        tree_labels_onehot = np.eye(self.n_clusters)[tree_labels_int]

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
        if self.interpreter is None:
            self._init_interpreter(input_dim=X_original.shape[1])
        self.tree.fit(X_original, pseudo_labels)
        if hasattr(self.tree, 'feature_importances_'):
            self.feature_importances_ = self.tree.feature_importances_

    def fit_predict(self, X, max_iters=10):
        X_normalized = X
        self._pretrain_vae(X_normalized)

        Z = self._extract_embedded_features(X_normalized)
        initial_pseudo_labels = self._get_cluster_labels(Z)
        initial_pseudo_labels = labelCluster(X_normalized, initial_pseudo_labels)
        self.pseudo_labels = initial_pseudo_labels

        self._init_interpreter(input_dim=X_normalized.shape[1])
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

        # 【核心修复点 1】：处理 EBM 没有 feature_importances_ 属性的问题。
        # 通过调用 explain_global().data() 自带的得分，将其映射对齐为特征重要性向量。
        if self.interpreter_type == 'EBM':
            ebm_global = self.tree.explain_global()
            ebm_data = ebm_global.data()
            importances_dict = {name: score for name, score in zip(ebm_data['names'], ebm_data['scores'])}
            importances = np.array([importances_dict.get(name, 0.0) for name in feature_names])
        elif hasattr(self.tree, 'feature_importances_'):
            importances = self.tree.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)

        feature_importances_ = pandas.DataFrame(importances)
        feature_importances_.index = feature_names
        sorted_feature_importances = feature_importances_.sort_values(ascending=False, by=0)

        if self.interpreter_type == 'DT':
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
            rules_str = self.visualize_decision_rules(feature_names)
            report = {
                'tree': self.tree,
                'feature_importances': sorted_feature_importances,
                'pseudo_labels': self.pseudo_labels,
                'latent_dim': self.latent_dim,
                'tree_rules': rules_str
            }
        return report

    def visualize_decision_rules(self, feature_names=None):
        if self.tree is None:
            raise ValueError("Please call fit_predict() first to train the model")

        if feature_names is None:
            n_features = len(self.feature_importances_) if self.feature_importances_ is not None else 1
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        if hasattr(feature_names, 'tolist'):
            feature_names = feature_names.tolist()

        if self.interpreter_type == 'DT':
            return export_text(self.tree, feature_names=feature_names)
        elif self.interpreter_type == 'RIPPER':
            return str(self.tree.ruleset_)
        elif self.interpreter_type == 'EBM':
            # 【核心修复点 2】：将方法获取修改为调用函数 ebm_global.data()，彻底根除 TypeError
            ebm_global = self.tree.explain_global()
            ebm_data = ebm_global.data()
            summary = "EBM Feature Contribution:\n"
            for name, weight in zip(ebm_data['names'], ebm_data['scores']):
                summary += f"  {name}: {weight:.4f}\n"
            return summary
        elif self.interpreter_type == 'SDT':
            summary = "Soft Decision Tree Gate Weights:\n"
            weights = self.tree.gate.weight.detach().numpy()[0]
            for i, w in enumerate(weights[:len(feature_names)]):
                summary += f"  {feature_names[i]}: {w:.4f}\n"
            return summary
        return "Unknown interpreter rules."

    # 【核心修复点】：新增外部脚本调用的映射函数，彻底解决该 AttributeError 阻塞
    def visualize_interpreter_rules(self, feature_names=None):
        """ 显式桥接映射，完全兼容外部 demo 脚本的 visualize_interpreter_rules 调用 """
        return self.visualize_decision_rules(feature_names=feature_names)

    def generate_latent_samples(self, X, n_samples=10):
        if not self.vae:
            raise ValueError("VAEs not trained. Please call fit_predict() first.")
        X_sample = X[:n_samples]
        X_tensor = torch.FloatTensor(X_sample)
        self.vae.eval()
        with torch.no_grad():
            _, _, _, z = self.vae(X_tensor)
            latent_samples = [z.numpy()]
        return latent_samples