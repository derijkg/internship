# features.py
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Common transition/signal words (customizable for Dutch or English)
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
    """
    Extracts numerical stylometric features from a single text and its pre-split sentences.
    """
    words = re.findall(r'\w+', text.lower())
    total_chars = len(text)
    
    if not words or not sentences:
        return np.zeros(12)  # Return zero vector if text is empty
    
    # Word & Sentence lengths
    sent_lengths = [len(re.findall(r'\w+', s)) for s in sentences if len(re.findall(r'\w+', s)) > 0]
    word_lengths = [len(w) for w in words]
    
    mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0.0
    var_sent_len = np.var(sent_lengths) if sent_lengths else 0.0
    # Burstiness is often represented as standard deviation / mean of sentence length
    burstiness = (np.std(sent_lengths) / mean_sent_len) if mean_sent_len > 0 else 0.0
    
    mean_word_len = np.mean(word_lengths)
    var_word_len = np.var(word_lengths)
    
    # Lexical Diversity
    ttr = calculate_ttr(words)
    hapax_ratio = calculate_hapax_ratio(words)
    
    # Transition word ratio
    transition_count = sum(1 for w in words if w in DUTCH_TRANSITIONS)
    transition_ratio = transition_count / len(words)
    
    # Formatting/Spacing features
    spaces_count = text.count(' ')
    double_spaces = text.count('  ')
    space_ratio = spaces_count / total_chars if total_chars > 0 else 0.0
    
    # Punctuation ratio
    punc_count = len(re.findall(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', text))
    punc_ratio = punc_count / total_chars if total_chars > 0 else 0.0
    
    return np.array([
        mean_sent_len, var_sent_len, burstiness,
        mean_word_len, var_word_len,
        ttr, hapax_ratio,
        transition_ratio, space_ratio, double_spaces, punc_ratio,
        float(total_chars)
    ])