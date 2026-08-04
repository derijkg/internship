import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Try importing Binoculars (from official repo: https://github.com/ahans30/Binoculars)
try:
    from binoculars import Binoculars
    HAS_BINOCULARS = True
except ImportError:
    HAS_BINOCULARS = False
    print("Warning: 'binoculars' package not installed. Using dummy wrapper for demonstration.")

# ==========================================
# 1. MODEL WRAPPERS & EXTRACTORS
# ==========================================

class FullAbstractSVM:
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ('svc', SVC(kernel='linear', C=1.0))
        ])

    def fit(self, X_texts, y_labels):
        self.model.fit(X_texts, y_labels)

    def get_scores(self, X_texts):
        """Returns signed distance to hyperplane (higher = more likely LLM)."""
        return self.model.decision_function(X_texts)


class SentenceSVM:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ('svc', SVC(kernel='linear', C=1.0))
        ])

    def fit(self, X_sentences, y_sentence_labels):
        self.model.fit(X_sentences, y_sentence_labels)

    def predict_sentence_scores(self, sentences):
        if not sentences:
            return np.array([0.0])
        return self.model.decision_function(sentences)

    def get_document_scores(self, X_texts, aggregation='mean'):
        """
        Splits text into sentences, scores each sentence, and aggregates to document level.
        aggregation: 'mean', 'max', or 'ratio_above_zero'
        """
        doc_scores = []
        for text in X_texts:
            sents = sent_tokenize(text)
            if not sents:
                doc_scores.append(0.0)
                continue
            
            sent_scores = self.predict_sentence_scores(sents)
            
            if aggregation == 'mean':
                doc_scores.append(np.mean(sent_scores))
            elif aggregation == 'max':
                doc_scores.append(np.max(sent_scores))
            elif aggregation == 'ratio':
                doc_scores.append(np.mean(sent_scores > 0)) # fraction of AI sentences
        return np.array(doc_scores)


class BinocularsWrapper:
    def __init__(self, observer="tiiuae/falcon-7b-instruct", performer="tiiuae/falcon-7b"):
        if HAS_BINOCULARS:
            self.bino = Binoculars(observer_name_or_path=observer, performer_name_or_path=performer)
        else:
            self.bino = None

    def get_scores(self, X_texts):
        """
        Calculates raw Binoculars score B(s).
        CRITICAL SCORE ALIGNMENT: 
        Binoculars: Lower score = AI, Higher score = Human.
        We invert it (-B(s)) so that Higher = AI, matching the SVM convention.
        """
        raw_scores = []
        for text in X_texts:
            if self.bino:
                score = self.bino.compute_score(text)
            else:
                # Dummy placeholder for structure validation
                score = np.random.normal(0.9, 0.1) 
            raw_scores.append(score)
        
        # Invert so higher score = LLM text (aligning with SVM)
        return -np.array(raw_scores)


# ==========================================
# 2. METRIC EVALUATORS
# ==========================================

def compute_tpr_at_fixed_fpr(y_true, y_scores, target_fpr=0.001):
    """Computes True Positive Rate (TPR) at an ultra-low False Positive Rate (FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find the highest index where FPR <= target_fpr
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return tpr[idx[-1]]


def evaluate_pure_dataset(y_true, model_scores_dict):
    """
    Evaluates detectors on 0% vs 100% LLM data.
    model_scores_dict: {'Model Name': continuous_scores_array}
    """
    results = []
    
    for model_name, scores in model_scores_dict.items():
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        tpr_at_01_fpr = compute_tpr_at_fixed_fpr(y_true, scores, target_fpr=0.001) # 0.1% FPR
        tpr_at_001_fpr = compute_tpr_at_fixed_fpr(y_true, scores, target_fpr=0.0001) # 0.01% FPR
        
        results.append({
            'Model': model_name,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'TPR @ 0.1% FPR': tpr_at_01_fpr,
            'TPR @ 0.01% FPR': tpr_at_001_fpr
        })
        
    return pd.DataFrame(results)


def evaluate_mixed_dataset(llm_percentages, model_scores_dict):
    """
    Evaluates continuous output correlation against actual LLM content percentage.
    """
    results = []
    for model_name, scores in model_scores_dict.items():
        p_corr, p_val = pearsonr(llm_percentages, scores)
        s_corr, s_val = spearmanr(llm_percentages, scores)
        
        results.append({
            'Model': model_name,
            'Pearson r': p_corr,
            'Pearson p-val': p_val,
            'Spearman rho': s_corr,
            'Spearman p-val': s_val
        })
    return pd.DataFrame(results)


# ==========================================
# 3. PLOTTING UTILITIES FOR PAPER
# ==========================================

def plot_roc_curves(y_true, model_scores_dict, save_path="roc_curves.png"):
    plt.figure(figsize=(8, 6))
    for model_name, scores in model_scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xscale('log') # Log scale emphasizes low FPR performance
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves on Pure Human vs. LLM Text')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mixed_text_correlation(llm_percentages, model_scores_dict, save_path="mixed_correlation.png"):
    plt.figure(figsize=(9, 6))
    for model_name, scores in model_scores_dict.items():
        # Min-max scale scores to [0, 1] range for visual comparison
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        sns.regplot(x=llm_percentages, y=norm_scores, label=model_name, scatter_kws={'alpha':0.3}, lowess=True)
        
    plt.xlabel('True LLM Text Percentage')
    plt.ylabel('Normalized Detector Score')
    plt.title('Detector Sensitivity across Synthetic Mixed Texts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()