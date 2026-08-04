# models/svm.py
from datetime import datetime
import joblib
import numpy as np
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import pandas as pd

from evaluation import run_full_evaluation
from features import (
    clear_optuna_cache,
    get_cached_split_features,
    get_dutch_stopwords_lemmatized,
    get_feature_extraction_pipeline,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

# Suppress repetitive Liblinear convergence warnings during Optuna search
warnings.filterwarnings('ignore', category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.INFO)


def extract_doc_ids(df: pd.DataFrame) -> np.ndarray:
  """Helper to extract unique abstract group IDs from DataFrame."""
  for col in ['_id', 'doc_id', 'id']:
    if col in df.columns:
      return df[col].values
  return np.arange(len(df))


def predict_pipeline(
    pipeline: Pipeline,
    X_raw: Union[List[str], str, List[Dict]],
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Inference helper for production deployment on raw strings or dicts.

  Applies text normalization, feature extraction, and optimal calibrated
  thresholding.

  Returns: (binary_predictions, probability_or_decision_scores)
  """
  is_calibrated = hasattr(pipeline, 'predict_proba')
  default_thresh = 0.5 if is_calibrated else 0.0

  if threshold is None:
    threshold = getattr(pipeline, 'optimal_threshold', default_thresh)

  items = [X_raw] if isinstance(X_raw, (str, dict)) else X_raw

  if is_calibrated:
    scores = pipeline.predict_proba(items)[:, 1]
  else:
    scores = pipeline.decision_function(items)

  preds = (scores >= threshold).astype(int)
  return preds, scores


# ==========================================
# 1. Classifier & Parameter Builders
# ==========================================
class ClassifierFactory:
  """Instantiates SVM classifiers with strict group-isolated calibration (zero data leakage) and configurable solver parameters."""

  @staticmethod
  def create(
      kernel: str,
      c_val: float,
      gamma: str = 'scale',
      coef0: float = 0.0,
      linear_loss: str = 'squared_hinge',
      class_weight: Any = 'balanced',
      calibrate: bool = False,
      cv: Optional[Any] = None,
      cv_groups: Optional[np.ndarray] = None,
      y: Optional[np.ndarray] = None,
      max_iter: int = 10000,
      tol: float = 1e-4,  # Standard high-precision tolerance
  ):
    # -------------------------------------------------------------------------
    # 1. Base Classifier Construction
    # -------------------------------------------------------------------------
    if kernel == 'linear':
      # For squared_hinge loss, dual=False is faster and avoids dual solver issues
      dual_mode = True if linear_loss == 'hinge' else False

      base_clf = LinearSVC(
          C=c_val,
          loss=linear_loss,
          dual=dual_mode,
          random_state=42,
          class_weight=class_weight,
          max_iter=max_iter,
          tol=tol,
      )
    else:
      base_clf = SVC(
          C=c_val,
          kernel=kernel,
          gamma=gamma,
          coef0=coef0,
          random_state=42,
          class_weight=class_weight,
          cache_size=1000,  # 1GB cache for faster non-linear SVMs
          tol=tol,
      )

    # -------------------------------------------------------------------------
    # 2. Probability Calibration (Strict Zero-Data-Leakage Group CV)
    # -------------------------------------------------------------------------
    if calibrate:
      final_cv_splits = cv

      # Fallback: If pre-computed cv is missing but cv_groups & y are provided, generate fold tuples
      if final_cv_splits is None and cv_groups is not None and y is not None:
        n_groups = len(np.unique(cv_groups))
        min_class_count = pd.Series(y).value_counts().min()
        n_splits = max(2, min(3, min_class_count, n_groups))
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )

        dummy_X = np.zeros((len(cv_groups), 1))
        final_cv_splits = list(sgkf.split(dummy_X, y, groups=cv_groups))
      elif final_cv_splits is None:
        final_cv_splits = 3

      return CalibratedClassifierCV(
          estimator=base_clf, cv=final_cv_splits, method='sigmoid'
      )

    return base_clf


class TFIDFParamBuilder:
  """Samples expanded TF-IDF, Stylometric, and Class Weight parameters from Optuna trials."""

  @staticmethod
  def sample_tfidf(
      trial: Trial, prefix: str, granularity: str = 'full'
  ) -> Dict[str, Any]:
    min_ngram_low = 1 if prefix == 'word' else 2
    min_ngram_high = 2 if prefix == 'word' else 3
    min_ngram = trial.suggest_int(
        f'{prefix}_min_ngram', min_ngram_low, min_ngram_high
    )

    max_ngram_limit = 3 if prefix == 'word' else 5
    max_ngram = trial.suggest_int(
        f'{prefix}_max_ngram', 2 if prefix == 'word' else 3, max_ngram_limit
    )

    if min_ngram > max_ngram:
      min_ngram = max_ngram

    # Tailor max_features and min_df search ranges to granularity
    if granularity == 'sentence':
      max_feat_low, max_feat_high = 10000, 50000
      min_df_low, min_df_high = (2, 8) if prefix == 'char' else (2, 5)
    else:  # 'full' or 'both'
      max_feat_low, max_feat_high = 20000, 100000
      min_df_low, min_df_high = 1, 5

    return {
        'ngram_range': (min_ngram, max_ngram),
        'max_features': trial.suggest_int(
            f'{prefix}_max_features', max_feat_low, max_feat_high, step=10000
        ),
        'min_df': trial.suggest_int(
            f'{prefix}_min_df', min_df_low, min_df_high
        ),
        'max_df': 0.95,
        'norm': 'l2',
        'sublinear_tf': trial.suggest_categorical(
            f'{prefix}_sublinear_tf', [True, False]
        ),
        'binary': trial.suggest_categorical(f'{prefix}_binary', [True, False]),
        'analyzer': 'word' if prefix == 'word' else 'char',
    }

  @staticmethod
  def sample_stylometrics(
      trial: Trial, granularity: str = 'full'
  ) -> Dict[str, Any]:
    use_sty = trial.suggest_categorical('use_stylometrics', [True, False])
    sty_weight = (
        trial.suggest_float('sty_weight', 0.01, 10.0, log=True)
        if use_sty
        else 0.0
    )
    return {'use_stylometrics': use_sty, 'sty_weight': sty_weight}

  @staticmethod
  def sample_model_params(trial: Trial, kernel_choice: str) -> Dict[str, Any]:
    kernel = (
        trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
        if kernel_choice == 'all'
        else kernel_choice
    )
    c_val = trial.suggest_float('C', 1e-2, 1e2, log=True)

    linear_loss = (
        trial.suggest_categorical('linear_loss', ['squared_hinge', 'hinge'])
        if kernel == 'linear'
        else 'squared_hinge'
    )
    gamma = (
        trial.suggest_float('gamma', 1e-4, 1e1, log=True)
        if kernel in ['rbf', 'sigmoid']
        else 'scale'
    )
    coef0 = (
        trial.suggest_float('coef0', -1.0, 1.0) if kernel == 'sigmoid' else 0.0
    )

    weight_mode = trial.suggest_categorical(
        'weight_mode', ['balanced', 'custom']
    )
    if weight_mode == 'custom':
      human_w = trial.suggest_float('human_class_weight', 1.0, 20.0)
      class_weight = {0: human_w, 1: 1.0}
    else:
      class_weight = 'balanced'

    return {
        'kernel': kernel,
        'C': c_val,
        'linear_loss': linear_loss,
        'gamma': gamma,
        'coef0': coef0,
        'class_weight': class_weight,
    }

  @staticmethod
  def from_best_params(
      best_params: Dict[str, Any], prefix: str
  ) -> Dict[str, Any]:
    params = {}

    if (
        f'{prefix}_min_ngram' in best_params
        and f'{prefix}_max_ngram' in best_params
    ):
      params['ngram_range'] = (
          int(best_params[f'{prefix}_min_ngram']),
          int(best_params[f'{prefix}_max_ngram']),
      )
    else:
      params['ngram_range'] = (1, 2) if prefix == 'word' else (3, 5)

    key_mapping = {
        'max_features': 'max_features',
        'min_df': 'min_df',
        'max_df': 'max_df',
        'norm': 'norm',
        'sublinear_tf': 'sublinear_tf',
        'binary': 'binary',
        'analyzer': 'analyzer',
    }

    for param_name, tfidf_arg in key_mapping.items():
      optuna_key = f'{prefix}_{param_name}'
      if optuna_key in best_params:
        params[tfidf_arg] = best_params[optuna_key]

    params.setdefault('max_features', 50000)
    params.setdefault('min_df', 2)
    params.setdefault('sublinear_tf', True)
    params.setdefault('max_df', 0.95)
    params.setdefault('norm', 'l2')
    params.setdefault('binary', False)

    if prefix == 'word':
      params.setdefault('analyzer', 'word')
      params.setdefault('stop_words', get_dutch_stopwords_lemmatized())
    else:
      params.setdefault('analyzer', 'char')

    return params

  @staticmethod
  def extract_sty_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'use_stylometrics': best_params.get('use_stylometrics', True),
        'sty_weight': best_params.get('sty_weight', 1.0),
    }


# ==========================================
# 2. Metric & Threshold Evaluators
# ==========================================
class ScoreEvaluator:
  """Calculates classification evaluation metrics cleanly and robustly."""

  @staticmethod
  def evaluate(
      y_true: np.ndarray,
      y_pred: np.ndarray,
      y_score: Optional[np.ndarray],
      metric_name: str,
      max_fpr: float = 0.01,
  ) -> float:
    metric = metric_name.lower().replace('-', '_')

    try:
      if metric in ['pauc', 'p_auc', 'partial_auc', 'partial_roc_auc']:
        if y_score is None or len(np.unique(y_true)) < 2:
          return 0.0
        return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))
      elif metric in ['roc_auc', 'rocauc']:
        if y_score is None or len(np.unique(y_true)) < 2:
          return 0.0
        return float(roc_auc_score(y_true, y_score))
      elif metric in ['pr_auc', 'prauc', 'average_precision']:
        if y_score is None or len(np.unique(y_true)) < 2:
          return 0.0
        return float(average_precision_score(y_true, y_score, pos_label=1))
      elif metric == 'precision':
        return float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        )
      elif metric == 'recall':
        return float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        )
      elif metric in ['f0.5', 'f0_5']:
        return float(
            fbeta_score(y_true, y_pred, beta=0.5, pos_label=1, zero_division=0)
        )
      elif metric == 'f1':
        return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
      elif metric == 'mcc':
        return float(matthews_corrcoef(y_true, y_pred))
      elif metric in ['brier_score', 'brier']:
        if y_score is None:
          return 1.0
        # Sigmoid normalization if scores are uncalibrated decision function margins
        if np.min(y_score) < 0.0 or np.max(y_score) > 1.0:
          y_score_prob = 1.0 / (1.0 + np.exp(-y_score))
        else:
          y_score_prob = y_score
        return float(brier_score_loss(y_true, y_score_prob))
      elif metric == 'set_fp':
        if y_score is None or len(np.unique(y_true)) < 2:
          return 0.0
        fpr, tpr, _ = roc_curve(y_true, y_score)
        valid_indices = np.where(fpr <= max_fpr)[0]
        return (
            float(tpr[valid_indices[-1]]) if len(valid_indices) > 0 else 0.0
        )
      else:
        return float(f1_score(y_true, y_pred, average='macro'))
    except Exception:
      return 0.0

  @staticmethod
  def find_threshold_for_max_fpr(
      y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01
  ) -> float:
    try:
      fpr, tpr, thresholds = roc_curve(y_true, y_score)
      valid_indices = np.where(fpr <= target_fpr)[0]
      if len(valid_indices) > 0:
        return float(thresholds[valid_indices[-1]])
    except Exception:
      pass
    return 0.5


# ==========================================
# 3. Optuna Objectives (Group-Aware CV)
# ==========================================
class Stage1Objective:
  """Stage 1 Optuna objective using StratifiedGroupKFold to prevent _id data leakage."""

  def __init__(
      self,
      X_raw: List[Dict],
      y: np.ndarray,
      groups: np.ndarray,
      cv: StratifiedGroupKFold,
      kernel_choice: str,
      metric_name: str,
      granularity: str = 'full',
      max_fpr: float = 0.01,
  ):
    self.X_raw = X_raw
    self.y = y
    self.groups = groups
    self.cv = cv
    self.kernel_choice = kernel_choice
    self.metric_name = metric_name
    self.granularity = granularity
    self.max_fpr = max_fpr

  def __call__(self, trial: Trial) -> float:
    word_params = TFIDFParamBuilder.sample_tfidf(
        trial, 'word', granularity=self.granularity
    )
    char_params = TFIDFParamBuilder.sample_tfidf(
        trial, 'char', granularity=self.granularity
    )
    sty_params = TFIDFParamBuilder.sample_stylometrics(
        trial, granularity=self.granularity
    )

    fold_scores = []
    eval_kernel = (
        self.kernel_choice if self.kernel_choice != 'all' else 'linear'
    )

    for fold, (train_idx, val_idx) in enumerate(
        self.cv.split(self.X_raw, self.y, groups=self.groups)
    ):
      X_tr_raw = [self.X_raw[i] for i in train_idx]
      X_va_raw = [self.X_raw[i] for i in val_idx]
      y_tr, y_va = self.y[train_idx], self.y[val_idx]

      X_tr, X_va = get_cached_split_features(
          X_tr_raw,
          X_va_raw,
          word_params,
          char_params,
          sty_params=sty_params,
          use_pre_lemmatized=True,
          granularity=self.granularity,
      )

      # Fast tuning tolerance (tol=1e-3, max_iter=10000)
      clf = ClassifierFactory.create(
          kernel=eval_kernel,
          c_val=1.0,
          calibrate=False,
          max_iter=10000,
          tol=1e-3,
      )
      clf.fit(X_tr, y_tr)

      preds = clf.predict(X_va)
      decision_scores = clf.decision_function(X_va)

      score = ScoreEvaluator.evaluate(
          y_va, preds, decision_scores, self.metric_name, max_fpr=self.max_fpr
      )
      fold_scores.append(score)

      trial.report(score, step=fold)
      if trial.should_prune():
        raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


class Stage2Objective:
  """Stage 2 Optuna objective using StratifiedGroupKFold to prevent _id data leakage."""

  def __init__(
      self,
      X_raw: List[Dict],
      y: np.ndarray,
      groups: np.ndarray,
      cv: StratifiedGroupKFold,
      kernel_choice: str,
      metric_name: str,
      tuning_strategy: str,
      best_tfidf_params: Optional[Dict] = None,
      max_fpr: float = 0.01,
      granularity: str = 'full',
  ):
    self.X_raw = X_raw
    self.y = y
    self.groups = groups
    self.cv = cv
    self.kernel_choice = kernel_choice
    self.metric_name = metric_name
    self.tuning_strategy = tuning_strategy
    self.best_tfidf_params = best_tfidf_params or {}
    self.max_fpr = max_fpr
    self.granularity = granularity

    self.precomputed_folds: List[Tuple[Any, Any, np.ndarray, np.ndarray]] = []
    if self.tuning_strategy in ['2stage', 'model']:
      word_params = TFIDFParamBuilder.from_best_params(
          self.best_tfidf_params, 'word'
      )
      char_params = TFIDFParamBuilder.from_best_params(
          self.best_tfidf_params, 'char'
      )
      sty_params = TFIDFParamBuilder.extract_sty_params(self.best_tfidf_params)

      for train_idx, val_idx in self.cv.split(
          self.X_raw, self.y, groups=self.groups
      ):
        X_tr_raw = [self.X_raw[i] for i in train_idx]
        X_va_raw = [self.X_raw[i] for i in val_idx]
        y_tr, y_va = self.y[train_idx], self.y[val_idx]

        X_tr, X_va = get_cached_split_features(
            X_tr_raw,
            X_va_raw,
            word_params,
            char_params,
            sty_params=sty_params,
            use_pre_lemmatized=True,
            granularity=self.granularity,
        )
        self.precomputed_folds.append((X_tr, X_va, y_tr, y_va))

  def __call__(self, trial: Trial) -> float:
    model_params = TFIDFParamBuilder.sample_model_params(
        trial, self.kernel_choice
    )
    fold_scores = []

    if self.tuning_strategy in ['2stage', 'model']:
      for fold, (X_tr, X_va, y_tr, y_va) in enumerate(self.precomputed_folds):
        # Fast tuning tolerance (tol=1e-3, max_iter=10000)
        clf = ClassifierFactory.create(
            kernel=model_params['kernel'],
            c_val=model_params['C'],
            gamma=model_params['gamma'],
            coef0=model_params['coef0'],
            linear_loss=model_params['linear_loss'],
            class_weight=model_params['class_weight'],
            calibrate=False,
            max_iter=10000,
            tol=1e-3,
        )
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_va)
        decision_scores = clf.decision_function(X_va)

        score = ScoreEvaluator.evaluate(
            y_va, preds, decision_scores, self.metric_name, max_fpr=self.max_fpr
        )
        fold_scores.append(score)

        trial.report(score, step=fold)
        if trial.should_prune():
          raise optuna.TrialPruned()
    else:
      word_params = TFIDFParamBuilder.sample_tfidf(
          trial, 'word', granularity=self.granularity
      )
      char_params = TFIDFParamBuilder.sample_tfidf(
          trial, 'char', granularity=self.granularity
      )
      sty_params = TFIDFParamBuilder.sample_stylometrics(
          trial, granularity=self.granularity
      )

      for fold, (train_idx, val_idx) in enumerate(
          self.cv.split(self.X_raw, self.y, groups=self.groups)
      ):
        X_tr_raw = [self.X_raw[i] for i in train_idx]
        X_va_raw = [self.X_raw[i] for i in val_idx]
        y_tr, y_va = self.y[train_idx], self.y[val_idx]

        X_tr, X_va = get_cached_split_features(
            X_tr_raw,
            X_va_raw,
            word_params,
            char_params,
            sty_params=sty_params,
            use_pre_lemmatized=True,
            granularity=self.granularity,
        )

        # Fast tuning tolerance (tol=1e-3, max_iter=10000)
        clf = ClassifierFactory.create(
            kernel=model_params['kernel'],
            c_val=model_params['C'],
            gamma=model_params['gamma'],
            coef0=model_params['coef0'],
            linear_loss=model_params['linear_loss'],
            class_weight=model_params['class_weight'],
            calibrate=False,
            max_iter=10000,
            tol=1e-3,
        )
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_va)
        decision_scores = clf.decision_function(X_va)

        score = ScoreEvaluator.evaluate(
            y_va, preds, decision_scores, self.metric_name, max_fpr=self.max_fpr
        )
        fold_scores.append(score)

        trial.report(score, step=fold)
        if trial.should_prune():
          raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# ==========================================
# 4. Optuna Tuning Orchestrator
# ==========================================
class OptunaTuner:

  @staticmethod
  def print_best_trial_callback(study: Study, trial: FrozenTrial):
    if trial.state == TrialState.COMPLETE:
      best = study.best_trial
      print(
          f'-> [Optuna Progress] Best Trial {best.number} | Score'
          f' ({study.direction.name}): {best.value:.4f}'
      )

  @classmethod
  def get_or_create_study(
      cls,
      study_name: str,
      storage: str,
      score_metric: str,
      reset: bool = False,
  ) -> Study:
    if reset:
      try:
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"-> Cleared existing Optuna study: '{study_name}'")
      except Exception:
        pass

    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, n_startup_trials=5, seed=42
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=1, interval_steps=1
    )

    direction = (
        'minimize'
        if score_metric.lower() in ['brier_score', 'brier']
        else 'maximize'
    )

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

  @classmethod
  def run(
      cls,
      train_df: pd.DataFrame,
      granularity: str,
      kernel_choice: str = 'linear',
      tuning_strategy: str = '2stage',
      tuning_sample_size: int = 3000,
      trials: int = 15,
      trials_stage1: int = 10,
      trials_stage2: int = 10,
      reset_study: bool = False,
      score_metric: str = 'roc_auc',
      max_fpr: float = 0.01,
      study_name: Optional[str] = None,
      n_jobs_optuna: int = 1,
  ) -> Dict[str, Any]:

    db_path = 'sqlite:///optuna_svm.db?timeout=60'
    clean_metric = score_metric.replace('-', '_').replace('.', '')
    if study_name is None:
      extra_metric = f'_{max_fpr}' if score_metric in ['pauc', 'set_fp'] else ''
      study_name = f'svm_{kernel_choice}_{granularity}_{clean_metric}{extra_metric}_{tuning_strategy}'

    unique_groups = (
        train_df['_id'].unique()
        if '_id' in train_df.columns
        else train_df['doc_id'].unique()
        if 'doc_id' in train_df.columns
        else train_df['id'].unique()
    )
    target_rows = (
        max(1, int(len(train_df) * tuning_sample_size))
        if isinstance(tuning_sample_size, float)
        else min(tuning_sample_size, len(train_df))
    )

    if len(train_df) > target_rows:
      avg_rows_per_group = len(train_df) / max(1, len(unique_groups))
      target_num_groups = max(1, int(target_rows / avg_rows_per_group))

      rng = np.random.default_rng(42)
      sampled_groups = set(
          rng.choice(
              unique_groups,
              size=min(target_num_groups, len(unique_groups)),
              replace=False,
          )
      )

      id_col = (
          '_id'
          if '_id' in train_df.columns
          else ('doc_id' if 'doc_id' in train_df.columns else 'id')
      )
      train_sub = train_df[train_df[id_col].isin(sampled_groups)].copy()
      print(
          f'Group-subsampled training set down to {len(train_sub)} rows'
          f' ({len(sampled_groups)} unique abstract IDs) for CV tuning...'
      )
    else:
      train_sub = train_df

    cols = [
        c
        for c in ['text', 'sentences', 'text_lemmatized']
        if c in train_sub.columns
    ]
    X_raw_all = train_sub[cols].to_dict(orient='records')
    y_all = train_sub['label'].values
    groups_all = extract_doc_ids(train_sub)

    min_class_count = pd.Series(y_all).value_counts().min()
    num_unique_groups = len(np.unique(groups_all))
    n_splits = max(2, min(3, min_class_count, num_unique_groups))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_tfidf_params = {}
    if tuning_strategy == '2stage':
      print(
          f'\n>>> [Stage 1] Tuning Preprocessing and TF-IDF via {n_splits}-Fold'
          f' Group CV ({trials_stage1} trials)...'
      )
      study_s1 = cls.get_or_create_study(
          f'{study_name}_stage1', db_path, score_metric, reset=reset_study
      )

      if granularity == 'sentence':
        s1_warmstart = {
            'word_min_ngram': 1,
            'word_max_ngram': 2,
            'word_max_features': 25000,
            'word_min_df': 2,
            'word_norm': 'l2',
            'word_sublinear_tf': True,
            'word_binary': False,
            'char_min_ngram': 2,
            'char_max_ngram': 4,
            'char_max_features': 25000,
            'char_min_df': 2,
            'char_norm': 'l2',
            'char_sublinear_tf': True,
            'char_binary': False,
            'use_stylometrics': True,
            'sty_weight': 0.1,
        }
      else:
        s1_warmstart = {
            'word_min_ngram': 1,
            'word_max_ngram': 2,
            'word_max_features': 50000,
            'word_min_df': 2,
            'word_norm': 'l2',
            'word_sublinear_tf': True,
            'word_binary': False,
            'char_min_ngram': 3,
            'char_max_ngram': 5,
            'char_max_features': 50000,
            'char_min_df': 2,
            'char_norm': 'l2',
            'char_sublinear_tf': True,
            'char_binary': False,
            'use_stylometrics': True,
            'sty_weight': 1.0,
        }

      study_s1.enqueue_trial(s1_warmstart)

      objective_s1 = Stage1Objective(
          X_raw_all,
          y_all,
          groups_all,
          cv,
          kernel_choice,
          score_metric,
          granularity=granularity,
          max_fpr=max_fpr,
      )
      study_s1.optimize(
          objective_s1,
          n_trials=trials_stage1,
          n_jobs=max(1, n_jobs_optuna),
          callbacks=[cls.print_best_trial_callback],
      )
      best_tfidf_params = study_s1.best_params
      print(f'-> Best Preprocessing parameters found: {best_tfidf_params}')

    stage2_trials = (
        trials_stage2 if tuning_strategy in ['2stage', 'model'] else trials
    )
    print(
        f'\n>>> [Stage 2] Tuning SVM Parameters via {n_splits}-Fold Group CV'
        f' ({stage2_trials} trials)...'
    )

    study_s2 = cls.get_or_create_study(
        study_name, db_path, score_metric, reset=reset_study
    )

    if tuning_strategy == 'merged':
      study_s2.enqueue_trial({
          'word_min_ngram': 1,
          'word_max_ngram': 2,
          'word_max_features': 50000 if granularity != 'sentence' else 25000,
          'word_min_df': 2,
          'word_norm': 'l2',
          'word_sublinear_tf': True,
          'word_binary': False,
          'char_min_ngram': 3 if granularity != 'sentence' else 2,
          'char_max_ngram': 5 if granularity != 'sentence' else 4,
          'char_max_features': 50000 if granularity != 'sentence' else 25000,
          'char_min_df': 2,
          'char_norm': 'l2',
          'char_sublinear_tf': True,
          'char_binary': False,
          'use_stylometrics': True,
          'sty_weight': 1.0 if granularity != 'sentence' else 0.1,
          'C': 1.0,
          'kernel': 'linear' if kernel_choice == 'all' else kernel_choice,
          'linear_loss': 'squared_hinge',
          'weight_mode': 'balanced',
      })
    else:
      study_s2.enqueue_trial({
          'C': 1.0,
          'kernel': 'linear' if kernel_choice == 'all' else kernel_choice,
          'linear_loss': 'squared_hinge',
          'weight_mode': 'balanced',
      })

    objective_s2 = Stage2Objective(
        X_raw_all,
        y_all,
        groups_all,
        cv,
        kernel_choice,
        score_metric,
        tuning_strategy,
        best_tfidf_params,
        max_fpr=max_fpr,
        granularity=granularity,
    )
    study_s2.optimize(
        objective_s2,
        n_trials=stage2_trials,
        n_jobs=max(1, n_jobs_optuna),
        callbacks=[cls.print_best_trial_callback],
    )

    completed_trials = [
        t for t in study_s2.trials if t.state == TrialState.COMPLETE
    ]
    if completed_trials:
      best_value = study_s2.best_value
      top_trials = [
          t for t in completed_trials if abs(t.value - best_value) < 5e-3
      ]
      best_trial = min(top_trials, key=lambda t: t.params.get('C', float('inf')))
      best_s2_params = best_trial.params
      print(
          '\n[Tie-Breaker Applied] Best trial chosen (lowest C within'
          f' tolerance of {best_value:.4f}): {best_s2_params}'
      )
    else:
      best_s2_params = study_s2.best_params

    best_overall = {}
    if tuning_strategy == '2stage':
      best_overall.update(best_tfidf_params)
    best_overall.update(best_s2_params)
    if kernel_choice != 'all':
      best_overall['kernel'] = kernel_choice

    return best_overall


# ==========================================
# 5. Out-of-Fold Score Calculation
# ==========================================
def compute_oof_scores(
    X_train_raw: List[Dict],
    y_train: np.ndarray,
    doc_ids: np.ndarray,
    word_params: Optional[Dict],
    char_params: Optional[Dict],
    sty_params: Optional[Dict],
    c_val: float,
    kernel: str,
    gamma: str,
    coef0: float = 0.0,
    linear_loss: str = 'squared_hinge',
    class_weight: Any = 'balanced',
    calibrate: bool = True,
    n_splits: int = 3,
    granularity: str = 'full',
) -> np.ndarray:
  """Computes unbiased Out-of-Fold (OOF) scores across training set using StratifiedGroupKFold."""
  min_class_count = pd.Series(y_train).value_counts().min()
  num_unique_groups = len(np.unique(doc_ids))
  actual_splits = max(2, min(n_splits, min_class_count, num_unique_groups))

  sgkf = StratifiedGroupKFold(
      n_splits=actual_splits, shuffle=True, random_state=42
  )
  oof_scores = np.zeros(len(y_train))

  for fold, (train_idx, val_idx) in enumerate(
      sgkf.split(X_train_raw, y_train, groups=doc_ids)
  ):
    X_tr_raw = [X_train_raw[i] for i in train_idx]
    X_va_raw = [X_train_raw[i] for i in val_idx]
    y_tr_fold = y_train[train_idx]
    doc_ids_fold = doc_ids[train_idx]

    X_tr, X_va = get_cached_split_features(
        X_tr_raw,
        X_va_raw,
        word_params,
        char_params,
        sty_params=sty_params,
        use_pre_lemmatized=True,
        granularity=granularity,
    )

    inner_cv = None
    if calibrate:
      min_class_count_fold = pd.Series(y_tr_fold).value_counts().min()
      num_unique_groups_fold = len(np.unique(doc_ids_fold))
      inner_splits = max(
          2, min(3, min_class_count_fold, num_unique_groups_fold)
      )

      inner_sgkf = StratifiedGroupKFold(
          n_splits=inner_splits, shuffle=True, random_state=42
      )
      # Explicit pre-computed index tuples using document groups
      inner_cv = list(inner_sgkf.split(X_tr, y_tr_fold, groups=doc_ids_fold))

    # Standard high-precision parameters for OOF threshold search (tol=1e-4, max_iter=10000)
    clf = ClassifierFactory.create(
        kernel=kernel,
        c_val=c_val,
        gamma=gamma,
        coef0=coef0,
        linear_loss=linear_loss,
        class_weight=class_weight,
        calibrate=calibrate,
        cv=inner_cv,
        cv_groups=doc_ids_fold,
        y=y_tr_fold,
        max_iter=10000,
        tol=1e-4,
    )

    clf.fit(X_tr, y_tr_fold)

    if calibrate or not hasattr(clf, 'decision_function'):
      oof_scores[val_idx] = clf.predict_proba(X_va)[:, 1]
    else:
      oof_scores[val_idx] = clf.decision_function(X_va)

  return oof_scores


# ==========================================
# 6. Main Pipeline Runner
# ==========================================
def train_svm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    c_val: float,
    kernel: str,
    save_path: str,
    granularity: str,
    test_raw_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    run_optuna: bool = False,
    reset_study: bool = False,
    trials: int = 15,
    trials_stage1: int = 10,
    trials_stage2: int = 10,
    tuning_strategy: str = '2stage',
    tuning_sample_size: int = 3000,
    score_metric: str = 'pauc',
    max_fpr: float = 0.01,
    study_name: Optional[str] = None,
    n_jobs_optuna: int = 1,
    eval_mode: str = 'both',
    mixed_ratios: list = [0.25, 0.50, 0.75],
    selected_models: list = [
        'qwen3.5:4b',
        'qwen3.6:27b',
        'gemma4:e4b',
        'gemma4:26b',
    ],
    calibrate: bool = True,
):

  word_params, char_params = None, None
  sty_params = {'use_stylometrics': True, 'sty_weight': 1.0}
  gamma = 'scale'
  coef0 = 0.0
  linear_loss = 'squared_hinge'
  class_weight = 'balanced'
  best_params = {}

  if run_optuna:
    print(
        'Running Hyperparameter Optimization via Optuna (Group-Aware CV,'
        f' Metric={score_metric}, Max FPR={max_fpr})...'
    )
    best_params = OptunaTuner.run(
        train_df=train_df,
        granularity=granularity,
        kernel_choice=kernel,
        tuning_strategy=tuning_strategy,
        tuning_sample_size=tuning_sample_size,
        trials=trials,
        trials_stage1=trials_stage1,
        trials_stage2=trials_stage2,
        reset_study=reset_study,
        score_metric=score_metric,
        max_fpr=max_fpr,
        study_name=study_name,
        n_jobs_optuna=n_jobs_optuna,
    )
    c_val = best_params.get('C', c_val)
    kernel = best_params.get('kernel', kernel)
    gamma = best_params.get('gamma', 'scale')
    coef0 = best_params.get('coef0', 0.0)
    linear_loss = best_params.get('linear_loss', 'squared_hinge')

    weight_mode = best_params.get('weight_mode', 'balanced')
    if weight_mode == 'custom':
      human_w = best_params.get('human_class_weight', 1.0)
      class_weight = {0: human_w, 1: 1.0}
    else:
      class_weight = 'balanced'

    word_params = TFIDFParamBuilder.from_best_params(best_params, 'word')
    char_params = TFIDFParamBuilder.from_best_params(best_params, 'char')
    sty_params = TFIDFParamBuilder.extract_sty_params(best_params)
  else:
    word_params = TFIDFParamBuilder.from_best_params({}, 'word')
    char_params = TFIDFParamBuilder.from_best_params({}, 'char')

  cols = [
      c for c in ['text', 'sentences', 'text_lemmatized'] if c in train_df.columns
  ]
  X_train_raw = train_df[cols].to_dict(orient='records')
  y_train = train_df['label'].values
  doc_ids = extract_doc_ids(train_df)

  feature_pipeline = get_feature_extraction_pipeline(
      word_tfidf_params=word_params,
      char_tfidf_params=char_params,
      sty_params=sty_params,
      stylometrics_n_jobs=1,
      use_pre_lemmatized=True,
      granularity=granularity,
  )

  # Pre-compute group-isolated fold splits for full training set probability calibration
  final_cv = None
  if calibrate:
    min_class_count = pd.Series(y_train).value_counts().min()
    num_unique_groups = len(np.unique(doc_ids))
    n_splits = max(2, min(3, min_class_count, num_unique_groups))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    final_cv = list(sgkf.split(X_train_raw, y_train, groups=doc_ids))

  # Final production classifier (tol=1e-4, max_iter=10000)
  clf = ClassifierFactory.create(
      kernel=kernel,
      c_val=c_val,
      gamma=gamma,
      coef0=coef0,
      linear_loss=linear_loss,
      class_weight=class_weight,
      calibrate=calibrate,
      cv=final_cv,
      cv_groups=doc_ids,
      y=y_train,
      max_iter=10000,
      tol=1e-4,
  )

  full_pipeline = Pipeline(
      [('features', feature_pipeline), ('classifier', clf)]
  )

  optimal_threshold = 0.5
  if score_metric == 'set_fp':
    print(
        '\nCalculating Out-of-Fold (OOF) probability scores using'
        ' StratifiedGroupKFold...'
    )
    oof_scores = compute_oof_scores(
        X_train_raw=X_train_raw,
        y_train=y_train,
        doc_ids=doc_ids,
        word_params=word_params,
        char_params=char_params,
        sty_params=sty_params,
        c_val=c_val,
        kernel=kernel,
        gamma=gamma,
        coef0=coef0,
        linear_loss=linear_loss,
        class_weight=class_weight,
        calibrate=calibrate,
        granularity=granularity,
    )
    optimal_threshold = ScoreEvaluator.find_threshold_for_max_fpr(
        y_train, oof_scores, target_fpr=0.01
    )
    print(
        '-> Calibrated Threshold (OOF 1% Max FPR Probability):'
        f' {optimal_threshold:.6f}'
    )

  full_pipeline.optimal_threshold = optimal_threshold

  print('Training final probability-calibrated SVM pipeline on 100% of data...')
  full_pipeline.fit(X_train_raw, y_train)

  metadata = {
      'study_name': study_name or f'svm_{granularity}',
      'save_path': save_path,
      'granularity': granularity,
      'tuning_strategy': tuning_strategy,
      'kernel': kernel,
      'score_metric': score_metric,
      'tuning_sample_size': tuning_sample_size,
      'C': c_val,
      'linear_loss': linear_loss,
      'weight_mode': best_params.get('weight_mode', 'balanced'),
      'human_class_weight': best_params.get('human_class_weight', 1.0),
      'use_stylometrics': sty_params.get('use_stylometrics', True),
      'sty_weight': sty_params.get('sty_weight', 1.0),
      'word_ngram': (
          f"({word_params.get('ngram_range', (1,2))[0]},{word_params.get('ngram_range', (1,2))[1]})"
      ),
      'word_max_features': word_params.get('max_features', 50000),
      'word_min_df': word_params.get('min_df', 2),
      'char_ngram': (
          f"({char_params.get('ngram_range', (3,5))[0]},{char_params.get('ngram_range', (3,5))[1]})"
      ),
      'char_max_features': char_params.get('max_features', 50000),
      'char_min_df': char_params.get('min_df', 2),
  }

  run_full_evaluation(
      model_pipeline=full_pipeline,
      test_raw_df=test_raw_df if test_raw_df is not None else test_df,
      test_df=test_df,
      metadata=metadata,
      selected_models=selected_models,
      mixed_ratios=mixed_ratios,
      eval_mode=eval_mode,
      experiments_dir='experiments',
  )

  if os.path.dirname(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

  joblib.dump(full_pipeline, save_path)
  print(f'Deployable pipeline saved successfully to {save_path}')

  clear_optuna_cache()