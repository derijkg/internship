import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import shap
from pathlib import Path

# -------------------------------------------------------------------
# 0. LOAD AND CONCATENATE ALL GENERATOR DATASETS
# -------------------------------------------------------------------
file_paths = [
    r'E:\code\dta\internship\src\detection\feature_tests\abstracts_qwen_qwen3.5_4b_full_501\qwen2.5-3b-base_abstracts_sentence_features.csv',
    r'E:\code\dta\internship\src\detection\feature_tests\abstracts_qwen_gemma4_e4b_full_500\qwen2.5-3b-base_abstracts_sentence_features.csv',
    r'E:\code\dta\internship\src\detection\feature_tests\abstracts_qwen_gemma4_26b_full_500\qwen2.5-3b-base_abstracts_sentence_features.csv',
    r'E:\code\dta\internship\src\detection\feature_tests\abstracts_qwen_qwen3.6_27b_full_500\qwen2.5-3b-base_abstracts_sentence_features.csv'
]

dfs = []
for fp in file_paths:
    path_obj = Path(fp)
    if path_obj.exists():
        print(f"Loading: {path_obj.parent.name}")
        dfs.append(pd.read_csv(path_obj))
    else:
        print(f"Warning: Path not found -> {fp}")

if not dfs:
    raise FileNotFoundError("No valid CSV files were loaded. Please check your file paths.")

# Concatenate all datasets
df_raw = pd.concat(dfs, ignore_index=True)

# Separate Human and AI rows
human_mask = (df_raw['is_llm'] == 0) if 'is_llm' in df_raw.columns else (df_raw['label'].astype(str).str.upper() == 'HUMAN')

df_ai = df_raw[~human_mask]
df_human = df_raw[human_mask]

# Deduplicate Human rows based on unique sentence identifier or doc ID
# (Use 'sentence_id' or '_id' depending on your dataframe column)
dedup_col = 'sentence_id' if 'sentence_id' in df_human.columns else '_id'
df_human_unique = df_human.drop_duplicates(subset=[dedup_col])

# Combine unique Humans + all AI samples
df = pd.concat([df_human_unique, df_ai], ignore_index=True).copy()

print(f"Dataset cleaned:")
print(f"  - Unique Human samples: {len(df_human_unique)}")
print(f"  - Total AI samples:     {len(df_ai)}")
print(f"  - Total unified dataset: {len(df)}")

# -------------------------------------------------------------------
# 1. CONVERT STRING LABELS TO BINARY (0 = Human, 1 = AI / LLM)
# -------------------------------------------------------------------
if 'is_llm' in df.columns:
    df['y_target'] = df['is_llm'].astype(int)
else:
    df['y_target'] = (df['label'].astype(str).str.upper() != 'HUMAN').astype(int)

# -------------------------------------------------------------------
# 2. SELECT ONLY NUMERIC FEATURE COLUMNS
# -------------------------------------------------------------------
ignore_cols = {
    '_id', 'sentence_id', 'label', 'generator', 'generator_model', 
    'domain', 'token_length', 'is_llm', 'y_target'
}

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ignore_cols]

# Clean Infs and NaNs
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Processing {len(feature_cols)} numeric features across all combined datasets.")

# -------------------------------------------------------------------
# 3. GLOBAL AUROC RANKING
# -------------------------------------------------------------------
global_auroc = {}
for col in feature_cols:
    try:
        if df[col].nunique() <= 1:
            continue
        score = roc_auc_score(df['y_target'], df[col])
        global_auroc[col] = max(score, 1.0 - score)
    except Exception as e:
        print(f"Skipping feature {col}: {e}")

auroc_df = pd.DataFrame(list(global_auroc.items()), columns=['Feature', 'Global_AUROC'])

# -------------------------------------------------------------------
# 4. GENERATOR SUBGROUP STABILITY ANALYSIS
# -------------------------------------------------------------------
gen_col = 'generator_model' if 'generator_model' in df.columns else ('generator' if 'generator' in df.columns else None)

if gen_col:
    ai_generators = [g for g in df[gen_col].unique() if str(g).upper() != 'HUMAN']
    print(f"Found {len(ai_generators)} AI generators for subgroup testing: {ai_generators}")
    
    for gen in ai_generators:
        sub_mask = (df['y_target'] == 0) | (df[gen_col] == gen)
        sub_df = df[sub_mask]
        
        if len(sub_df['y_target'].unique()) < 2:
            continue
            
        gen_scores = []
        for col in auroc_df['Feature']:
            try:
                s = roc_auc_score(sub_df['y_target'], sub_df[col])
                gen_scores.append(max(s, 1.0 - s))
            except Exception:
                gen_scores.append(0.5)
                
        auroc_df[f'AUROC_{gen}'] = gen_scores

    auroc_cols = [c for c in auroc_df.columns if c.startswith('AUROC_')]
    if auroc_cols:
        auroc_df['Min_Generator_AUROC'] = auroc_df[auroc_cols].min(axis=1)
    else:
        auroc_df['Min_Generator_AUROC'] = auroc_df['Global_AUROC']
else:
    auroc_df['Min_Generator_AUROC'] = auroc_df['Global_AUROC']

# -------------------------------------------------------------------
# 5. FILTER ROBUST FEATURES & RUN SHAP IMPORTANCE
# -------------------------------------------------------------------
robust_mask = auroc_df['Min_Generator_AUROC'] >= 0.65
robust_features = auroc_df[robust_mask]['Feature'].tolist()

if len(robust_features) == 0:
    print("Warning: No features passed >= 0.65 threshold. Relaxing threshold to 0.55.")
    robust_features = auroc_df[auroc_df['Min_Generator_AUROC'] >= 0.55]['Feature'].tolist()

print(f"Retained {len(robust_features)} robust features out of {len(feature_cols)}.")

# Train model
X = df[robust_features]
y = df['y_target']

model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
model.fit(X, y)

# Compute SHAP Importance (Disabling strict additivity check for GBDT histograms)
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)

# Extract SHAP matrix correctly whether 2D or 3D
shap_matrix = shap_values.values
if shap_matrix.ndim == 3:
    shap_matrix = shap_matrix[:, :, 1]

mean_shap = np.abs(shap_matrix).mean(axis=0)

final_ranking = pd.DataFrame({
    'Feature': robust_features,
    'SHAP_Importance': mean_shap
}).merge(auroc_df[['Feature', 'Global_AUROC', 'Min_Generator_AUROC']], on='Feature')

final_ranking = final_ranking.sort_values(by='SHAP_Importance', ascending=False)

print("\n" + "="*70)
print("TOP 20 MOST DISCRIMINATIVE & ROBUST FEATURES (COMBINED GENERATORS)")
print("="*70)
print(final_ranking.head(20).to_string(index=False))