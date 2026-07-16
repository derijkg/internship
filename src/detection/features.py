# features.py
import re
import string
import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

DUTCH_TRANSITIONS = {"echter", "bovendien", "daarnaast", "desalniettemin", "kortom", "tevens", "daardoor", "derhalve", "bijgevolg", "namelijk"}

def calculate_ttr(words):
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def calculate_hapax_ratio(words):
    if not words:
        return 0.0
    counts = Counter(words)
    hapaxes = sum(1 for w, c in counts.items() if c == 1)
    return hapaxes / len(words)

def extract_stylometric_features(text, sentences):
    words = re.findall(r'\w+', text.lower())
    total_chars = len(text)
    
    if not words or not sentences:
        return np.zeros(12)
    
    sent_lengths = [len(re.findall(r'\w+', s)) for s in sentences if len(re.findall(r'\w+', s)) > 0]
    word_lengths = [len(w) for w in words]
    
    mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0.0
    var_sent_len = np.var(sent_lengths) if sent_lengths else 0.0
    burstiness = (np.std(sent_lengths) / mean_sent_len) if mean_sent_len > 0 else 0.0
    
    mean_word_len = np.mean(word_lengths)
    var_word_len = np.var(word_lengths)
    
    ttr = calculate_ttr(words)
    hapax_ratio = calculate_hapax_ratio(words)
    
    transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
    transition_ratio = transition_count / len(words)
    
    spaces_count = text.count(' ')
    double_spaces = text.count('  ')
    space_ratio = spaces_count / total_chars if total_chars > 0 else 0.0
    
    double_space_ratio = double_spaces / total_chars if total_chars > 0 else 0.0
    
    punc_count = sum(1 for c in text if c in string.punctuation)
    punc_ratio = punc_count / total_chars if total_chars > 0 else 0.0
    
    return np.array([
        mean_sent_len, var_sent_len, burstiness,
        mean_word_len, var_word_len,
        ttr, hapax_ratio,
        transition_ratio, space_ratio, double_space_ratio, punc_ratio,
        float(total_chars)
    ])


class TextExtractor(BaseEstimator, TransformerMixin):
    """Extracts text field from list of dictionary records."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [item['text'] for item in X]

class StylometricExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Prevent parallelization overhead on small validation folds
        if len(X) < 100 or self.n_jobs == 1:
            features = [extract_stylometric_features(item['text'], item['sentences']) for item in X]
        else:
            features = Parallel(n_jobs=self.n_jobs)(
                delayed(extract_stylometric_features)(item['text'], item['sentences']) 
                for item in X
            )
        return np.array(features)