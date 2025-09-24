# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_score,
    recall_score, f1_score
)

class Config:
    SOURCE_TEST_CSV_PATH = 'Dataset/Splitted/source_test.csv'
    TARGET_TEST_CSV_PATH = 'Dataset/Splitted/target_test.csv'
    BASE_DIR = "Model"
    TRAIN_SUBDIR = "Train"
    EVAL_SUBDIR = "Evaluation"
    MODELS = ["T-1","T-2","T-3","T-4","E-1","E-2","E-3"]

def save_csv(df, out_dir, filename):
    df.to_csv(os.path.join(out_dir, filename), index=True)

def metrics_only(model, X, y, model_name):
    y_pred = model.predict(X)
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Kappa": cohen_kappa_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y, y_pred, average='macro', zero_division=0),
        "F1-Score": f1_score(y, y_pred, average='macro', zero_division=0)
    }

def evaluate():
    train_dir = os.path.join(Config.BASE_DIR, Config.TRAIN_SUBDIR)
    eval_dir  = os.path.join(Config.BASE_DIR, Config.EVAL_SUBDIR)
    os.makedirs(eval_dir, exist_ok=True)

    pre = joblib.load(os.path.join(train_dir, "preprocessing_pipeline.pkl"))

    # --- Source domain ---
    df_src = pd.read_csv(Config.SOURCE_TEST_CSV_PATH)
    X_src, y_src = df_src.drop(columns=['VALUE']), df_src['VALUE']
    X_src_t = pre.transform(X_src)

    # --- Target domain ---
    df_tgt = pd.read_csv(Config.TARGET_TEST_CSV_PATH)
    X_tgt, y_tgt = df_tgt.drop(columns=['VALUE']), df_tgt['VALUE']
    X_tgt_t = pre.transform(X_tgt)

    results_src, results_tgt = [], []
    for name in Config.MODELS:
        model = joblib.load(os.path.join(train_dir, f"{name}.pkl"))
        results_src.append(metrics_only(model, X_src_t, y_src, name))
        results_tgt.append(metrics_only(model, X_tgt_t, y_tgt, name))

    df_src_out = pd.DataFrame(results_src).set_index("Model").round(4)
    df_tgt_out = pd.DataFrame(results_tgt).set_index("Model").round(4)

    save_csv(df_src_out, eval_dir, "evaluation_results_source_test.csv")
    save_csv(df_tgt_out, eval_dir, "evaluation_results_target_test.csv")

if __name__ == "__main__":
    evaluate()
