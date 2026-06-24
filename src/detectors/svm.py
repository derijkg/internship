#first try full ai

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ===================================================================
# STEP 1: LOAD AND RESTRUCTURE THE PARQUET DATA
# ===================================================================
print("Loading Parquet dataset...")
df = pd.read_parquet('E:/code/dta/internship/data/ug_selected.parquet')

# We will build a flat, long-format dataset:
# Columns: 'abstract_id', 'text', 'label' (0 = Human, 1 = AI)
flat_data = []

# List of the AI model columns we want to evaluate against
ai_models = ['qwen3.6:27b', 'qwen3.5:4b', 'gemma4:e4b']  # Adjust based on your columns

for idx, row in df.iterrows():
    # Fallback to row index if _id is missing
    abs_id = row.get('_id', idx)
    human_text = row.get('text_dut')
    
    # 1. Add the Human sentence (Label 0)
    if isinstance(human_text, str) and human_text.strip():
        flat_data.append({
            'abstract_id': abs_id,
            'text': human_text,
            'label': 0
        })
        
        # 2. Add the corresponding AI rewrites (Label 1)
        for model in ai_models:
            if model in df.columns:
                ai_text = row[model]
                if isinstance(ai_text, str) and ai_text.strip():
                    flat_data.append({
                        'abstract_id': abs_id,
                        'text': ai_text,
                        'label': 1
                    })

dataset = pd.DataFrame(flat_data)
print(f"Dataset compiled. Total sentences: {len(dataset)}")
print(f"Class distribution:\n{dataset['label'].value_counts(normalize=True)}")

# ===================================================================
# STEP 2: GROUP-SAFE TRAIN/TEST SPLIT
# ===================================================================
# This ensures sentences from the same abstract never cross the train/test boundary
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(
    X=dataset['text'], 
    y=dataset['label'], 
    groups=dataset['abstract_id']
))

X_train, y_train = dataset['text'].iloc[train_idx], dataset['label'].iloc[train_idx]
X_test, y_test = dataset['text'].iloc[test_idx], dataset['label'].iloc[test_idx]

print(f"\nTraining set size: {len(X_train)} (from {dataset['abstract_id'].iloc[train_idx].nunique()} abstracts)")
print(f"Testing set size: {len(X_test)} (from {dataset['abstract_id'].iloc[test_idx].nunique()} abstracts)")

# ===================================================================
# STEP 3: CONSTRUCT THE FEATURE EXTRACTION & SVM PIPELINE
# ===================================================================
# We use Character-level TF-IDF (n-grams of 3 to 5 characters).
# 'sublinear_tf=True' squashes term frequency scaling to prevent long repetitive sentences from dominating.
vectorizer = TfidfVectorizer(
    analyzer='char', 
    ngram_range=(3, 5), 
    sublinear_tf=True
)

# LinearSVC is significantly faster and less memory-intensive than standard SVC 
# when dealing with high-dimensional sparse text vectors.
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('clf', LinearSVC(random_state=42, C=1.0, dual='auto'))
])

# ===================================================================
# STEP 4: TRAINING & EVALUATION
# ===================================================================
print("\nTraining Linear SVM...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# Make predictions
y_pred = pipeline.predict(X_test)

# Display evaluation reports
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Calculate AUC Score (LinearSVC decision function acts as the confidence score)
decision_scores = pipeline.decision_function(X_test)
auc_score = roc_auc_score(y_test, decision_scores)
print(f"\nROC-AUC Score: {auc_score:.4f}")


#analysis
# Extract the feature names (the character n-grams)
feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
# Get the SVM coefficients
coefs = pipeline.named_steps['clf'].coef_[0]

# Pair them and sort
features_coefs = sorted(zip(coefs, feature_names))

print("\nTop 10 strongest indicators of HUMAN text:")
for coef, feat in features_coefs[:10]:
    print(f"{feat:10} : {coef:.4f}")
    
print("\nTop 10 strongest indicators of AI text:")
for coef, feat in features_coefs[-10:]:
    print(f"{feat:10} : {coef:.4f}")