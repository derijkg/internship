import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score


class StatisticalAnalyzer:
    """Computes Mann-Whitney U, Levene, Cohen's d, FDR control, and ROC-AUC metrics."""

    @staticmethod
    def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        if dof <= 0:
            return 0.0
        var_x = np.var(x, ddof=1) if nx > 1 else 0.0
        var_y = np.var(y, ddof=1) if ny > 1 else 0.0
        pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof)
        if pooled_std < 1e-8:
            return 0.0
        return float((np.mean(x) - np.mean(y)) / pooled_std)

    @staticmethod
    def format_p_value(p: float) -> str:
        if p < 1e-4:
            return f"{p:.2e} ***"
        elif p < 0.001:
            return f"{p:.2e} **"
        elif p < 0.05:
            return f"{p:.4f} *"
        else:
            return f"{p:.4f} (ns)"

    @classmethod
    def calculate_significance(cls, sent_df: pd.DataFrame) -> pd.DataFrame:
        if sent_df is None or sent_df.empty:
            return pd.DataFrame()

        human_df = sent_df[sent_df["is_llm"] == 0]
        llm_df = sent_df[sent_df["is_llm"] == 1]

        ignore_cols = [
            "sentence_id", "_id", "doc_id", "label", "generator_model",
            "is_llm", "text", "genre", "token_length"
        ]
        feature_cols = [col for col in sent_df.columns if col not in ignore_cols]
        results = []

        for feat in feature_cols:
            h_raw = pd.to_numeric(human_df[feat], errors='coerce').values
            l_raw = pd.to_numeric(llm_df[feat], errors='coerce').values

            h_vals = h_raw[np.isfinite(h_raw)]
            l_vals = l_raw[np.isfinite(l_raw)]

            if len(h_vals) < 2 or len(l_vals) < 2:
                continue

            try:
                _, p_mw = stats.mannwhitneyu(h_vals, l_vals, alternative='two-sided')
            except Exception:
                p_mw = 1.0

            try:
                _, p_lev = stats.levene(h_vals, l_vals)
            except Exception:
                p_lev = 1.0

            d_val = cls.compute_cohens_d(h_vals, l_vals)

            try:
                feat_series = pd.to_numeric(sent_df[feat], errors='coerce').values
                valid_mask = np.isfinite(feat_series) & np.isfinite(sent_df["is_llm"].values)

                if len(np.unique(sent_df.loc[valid_mask, "is_llm"])) > 1:
                    clean_y_true = sent_df.loc[valid_mask, "is_llm"].values
                    clean_y_scores = feat_series[valid_mask]
                    auc_val = float(roc_auc_score(clean_y_true, clean_y_scores))
                else:
                    auc_val = 0.5
            except Exception:
                auc_val = 0.5

            results.append({
                "feature": feat,
                "human_mean_std": f"{np.mean(h_vals):.3f} ± {np.std(h_vals):.3f}",
                "llm_mean_std": f"{np.mean(l_vals):.3f} ± {np.std(l_vals):.3f}",
                "cohens_d": round(d_val, 3),
                "roc_auc": round(auc_val, 3),
                "_raw_auc": auc_val,
                "_raw_p_mw": p_mw,
                "_raw_p_lev": p_lev
            })

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df["p_mw_fdr"] = stats.false_discovery_control(res_df["_raw_p_mw"].values)
            res_df["p_location (MW-U FDR)"] = res_df["p_mw_fdr"].apply(cls.format_p_value)
            res_df["p_variance (Levene)"] = res_df["_raw_p_lev"].apply(cls.format_p_value)

            res_df["_auc_dist"] = np.abs(res_df["_raw_auc"] - 0.5)
            res_df = res_df.sort_values(by="_auc_dist", ascending=False).drop(columns=["_auc_dist"]).reset_index(drop=True)
            res_df = res_df[res_df["p_mw_fdr"] < 0.05].reset_index(drop=True)

        return res_df