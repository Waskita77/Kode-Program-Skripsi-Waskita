# -*- coding: utf-8 -*-

import os
import json
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


class Config:
    TRAIN_CSV_PATH = 'Dataset/Splitted/target_train.csv'
    BASE_DIR = "Model_FineTune"
    TRAIN_SUBDIR = "Train"
    RANDOM_STATE_SEED = 42


def save_csv(df, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["Model", "Best Parameter", "Best F1 (CV)"]
    df = df[cols]
    df["Best Parameter"] = df["Best Parameter"].astype(str).fillna("")
    df["Best F1 (CV)"] = pd.to_numeric(df["Best F1 (CV)"], errors="coerce").round(3)
    df = df.sort_values(by="Best F1 (CV)", ascending=False, na_position="last").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, filename), index=False)


def _make_cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE_SEED)


def tune(model, param_grid, X, y, name, fit_params=None):
    cv = _make_cv()
    gs = GridSearchCV(model, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1)
    if fit_params:
        gs.fit(X, y, **fit_params)
    else:
        gs.fit(X, y)
    best_param_str = json.dumps(gs.best_params_, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    result = {"Model": name, "Best Parameter": best_param_str, "Best F1 (CV)": gs.best_score_}
    return gs.best_estimator_, result


def _fit_ensembles(m1, m2, m3, m4, X_t, y):
    meta = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        C=0.1,
        max_iter=5000,
        tol=1e-4,
        class_weight='balanced',
        random_state=Config.RANDOM_STATE_SEED,
        multi_class='auto'
    )
    cv = _make_cv()
    e1 = StackingClassifier(
        estimators=[('svm_linear', m1), ('svm_rbf', m2)],
        final_estimator=meta,
        stack_method='predict_proba',
        cv=cv,
        n_jobs=-1
    )
    e2 = StackingClassifier(
        estimators=[('rf', m3), ('xgb', m4)],
        final_estimator=meta,
        stack_method='predict_proba',
        cv=cv,
        n_jobs=-1
    )
    e3 = StackingClassifier(
        estimators=[('svm_linear', m1), ('svm_rbf', m2), ('rf', m3), ('xgb', m4)],
        final_estimator=meta,
        stack_method='predict_proba',
        cv=cv,
        n_jobs=-1
    )
    w = compute_sample_weight(class_weight='balanced', y=y)
    e1.fit(X_t, y, sample_weight=w)
    e2.fit(X_t, y, sample_weight=w)
    e3.fit(X_t, y, sample_weight=w)
    return e1, e2, e3


def fine_tune():
    train_dir = os.path.join(Config.BASE_DIR, Config.TRAIN_SUBDIR)
    os.makedirs(train_dir, exist_ok=True)

    df = pd.read_csv(Config.TRAIN_CSV_PATH)
    X, y = df.drop(columns=['VALUE']), df['VALUE']

    pre = Pipeline([('scaler', StandardScaler())])
    pre.fit(X, y)
    X_t = pre.transform(X)

    grid_t1 = {'C': [0.1, 1, 10, 100]}
    grid_t2 = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]}
    grid_t3 = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'max_features': ['sqrt', 'log2']}
    grid_t4 = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}

    m1, r1 = tune(
        SVC(kernel='linear', probability=True, random_state=Config.RANDOM_STATE_SEED, class_weight='balanced'),
        grid_t1, X_t, y, "T-1_SVC_Linear"
    )
    m2, r2 = tune(
        SVC(kernel='rbf', probability=True, random_state=Config.RANDOM_STATE_SEED, class_weight='balanced'),
        grid_t2, X_t, y, "T-2_SVC_RBF"
    )
    m3, r3 = tune(
        RandomForestClassifier(random_state=Config.RANDOM_STATE_SEED, class_weight='balanced'),
        grid_t3, X_t, y, "T-3_RandomForest"
    )

    w = compute_sample_weight(class_weight='balanced', y=y)
    m4, r4 = tune(
        XGBClassifier(eval_metric='mlogloss', random_state=Config.RANDOM_STATE_SEED, tree_method='hist', n_jobs=-1),
        grid_t4, X_t, y, "T-4_XGBoost",
        fit_params={'sample_weight': w}
    )

    e1, e2, e3 = _fit_ensembles(m1, m2, m3, m4, X_t, y)

    joblib.dump(pre, os.path.join(train_dir, "preprocessing_pipeline.pkl"))
    joblib.dump(m1, os.path.join(train_dir, "T-1_SVC_Linear.pkl"))
    joblib.dump(m2, os.path.join(train_dir, "T-2_SVC_RBF.pkl"))
    joblib.dump(m3, os.path.join(train_dir, "T-3_RandomForest.pkl"))
    joblib.dump(m4, os.path.join(train_dir, "T-4_XGBoost.pkl"))
    joblib.dump(e1, os.path.join(train_dir, "E-1.pkl"))
    joblib.dump(e2, os.path.join(train_dir, "E-2.pkl"))
    joblib.dump(e3, os.path.join(train_dir, "E-3.pkl"))

    tuning_df = pd.DataFrame([r1, r2, r3, r4])
    save_csv(tuning_df, train_dir, "hyperparameter_tuning_results.csv")


if __name__ == "__main__":
    fine_tune()
