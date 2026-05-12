"""LOGMAC-COLON: Omic Gradient Engine"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class GradientEngine:
    def __init__(self, expression_df: pd.DataFrame, metadata_df: pd.DataFrame = None):
        self.raw_data = expression_df.copy()
        self.metadata = metadata_df
        self.data_scaled = None
        self.pseudotime = None
        self.gradient_matrix = None
        self.feature_scores = None
        
    def preprocess(self) -> pd.DataFrame:
        log_data = np.log1p(self.raw_data.replace(0, 1e-6))
        scaler = StandardScaler()
        self.data_scaled = pd.DataFrame(
            scaler.fit_transform(log_data.T).T,
            index=log_data.index,
            columns=log_data.columns
        )
        gene_var = self.data_scaled.var(axis=1)
        keep_genes = gene_var > gene_var.quantile(0.2)
        self.data_scaled = self.data_scaled.loc[keep_genes]
        return self.data_scaled

    def compute_pseudotime(self) -> np.ndarray:
        if self.metadata is not None and 'clinical_stage' in self.metadata.columns:
            aligned = self.data_scaled.merge(self.metadata[['clinical_stage']], left_index=True, right_index=True)
            self.pseudotime = aligned['clinical_stage'].values
        else:
            pca = PCA(n_components=1)
            self.pseudotime = pca.fit_transform(self.data_scaled.T).flatten()
        p_min, p_max = self.pseudotime.min(), self.pseudotime.max()
        self.pseudotime = (self.pseudotime - p_min) / (p_max - p_min + 1e-8)
        return self.pseudotime

    def compute_omic_gradients(self, window: int = 7, polyorder: int = 2) -> pd.DataFrame:
        if self.data_scaled is None: self.preprocess()
        if self.pseudotime is None: self.compute_pseudotime()
        
        sort_idx = np.argsort(self.pseudotime)
        sorted_data = self.data_scaled.iloc[:, sort_idx].values
        sorted_pt = self.pseudotime[sort_idx]
        
        window = min(window, sorted_data.shape[1])
        if window % 2 == 0: window += 1
        smoothed = savgol_filter(sorted_data, window_length=window, polyorder=polyorder, axis=1)
        
        self.gradient_matrix = np.gradient(smoothed, sorted_pt, axis=1)
        
        grad_df = pd.DataFrame(
            np.zeros_like(self.data_scaled.values),
            index=self.data_scaled.index,
            columns=self.data_scaled.columns
        )
        grad_df.iloc[:, sort_idx] = self.gradient_matrix
        return grad_df

    def select_top_biomarkers(self, top_k: int = 50) -> tuple:
        if self.gradient_matrix is None: raise ValueError("Run compute_omic_gradients() first")
        grad_mag = np.abs(self.gradient_matrix).mean(axis=1)
        expr_var = self.data_scaled.var(axis=1)
        grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        var_norm = (expr_var - expr_var.min()) / (expr_var.max() - expr_var.min() + 1e-8)
        self.feature_scores = 0.7 * grad_norm + 0.3 * var_norm
        top_idx = np.argsort(self.feature_scores)[-top_k:][::-1]
        return self.data_scaled.index[top_idx], self.feature_scores[top_idx]

    def export_results(self) -> dict:
        return {
            "expression_matrix": self.data_scaled,
            "gradient_matrix": pd.DataFrame(self.gradient_matrix, index=self.data_scaled.index, columns=self.data_scaled.columns),
            "pseudotime": self.pseudotime,
            "selected_genes": self.feature_scores.sort_values(ascending=False).head(50).index.tolist(),
            "feature_scores": self.feature_scores
        }