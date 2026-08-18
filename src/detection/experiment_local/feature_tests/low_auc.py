import pandas as pd
import numpy as np

# Path to one of your generated significance CSVs
csv_path = "clin_qwen/qwen2.5-0.5b-base_clin33_significance_results.csv"

# Load the significance results
df = pd.read_csv(csv_path)

# Ensure roc_auc is float
auc_col = "_raw_auc" if "_raw_auc" in df.columns else "roc_auc"
df[auc_col] = pd.to_numeric(df[auc_col], errors="coerce")

print("==================================================")
print("1. TOP 5 HUMAN INDICATORS (Lowest Raw AUC -> 0.0)")
print("==================================================")
# Features that strongly correlate with Human-written text
top_human = df.sort_values(by=auc_col, ascending=True).head(5)
print(top_human[["feature", auc_col, "human_mean_std", "llm_mean_std", "cohens_d"]])

print("\n==================================================")
print("2. TOP 5 LEAST PREDICTIVE FEATURES (AUC -> 0.5)")
print("==================================================")
# Features that perform no better than random guessing
df["auc_distance"] = np.abs(df[auc_col] - 0.5)
least_predictive = df.sort_values(by="auc_distance", ascending=True).head(5)
print(least_predictive[["feature", auc_col, "auc_distance", "cohens_d"]])

print("\n==================================================")
print("3. TOP 5 OVERALL PREDICTORS (Highest |AUC - 0.5|)")
print("==================================================")
# Features with strongest separation (both Human-leaning and LLM-leaning)
top_overall = df.sort_values(by="auc_distance", ascending=False).head(5)
print(top_overall[["feature", auc_col, "auc_distance", "cohens_d"]])