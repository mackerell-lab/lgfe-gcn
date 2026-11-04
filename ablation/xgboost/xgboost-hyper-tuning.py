import optuna
import numpy as np
import pandas as pd
import random
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

seed = 121314
N_TRIALS = 100
np.random.seed(seed)
random.seed(seed)

FEATURE_FILE = "features.txt"   
ACTIVITY_FILE = "exp.csv"      
FEATURE_PREFIXES = ("GFE_",)

def load_data(feature_file, activity_file):
    feats = pd.read_csv(feature_file, sep=r"\s+")
    feats["CMPD"] = feats["SDF_name"].astype(str).str.replace(r"\.sdf$", "", regex=True)
    feats = feats.drop(columns=["SDF_name"])

    act = pd.read_csv(activity_file)
    df = act.merge(feats, on="CMPD", how="inner")

    n_missing = len(act) - len(df)
    if n_missing > 0:
        print(f"Warning: {n_missing} compounds in {activity_file} have no matching features and were dropped.")

    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    X = df[feature_cols].values.astype(np.float32)
    y = df["DG"].values.astype(np.float32)
    groups = df["KGRP"].values
    return df, X, y, groups, feature_cols


df, X, y, groups, feature_cols = load_data(FEATURE_FILE, ACTIVITY_FILE)
print(f"Loaded {len(df)} compounds, {len(feature_cols)} features, folds (KGRP): {sorted(np.unique(groups))}")


def kgrp_folds(groups):
    folds = []
    for g in sorted(np.unique(groups)):
        val_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        folds.append((train_idx, val_idx))
    return folds


folds = kgrp_folds(groups)

def build_params(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
    )


def make_model(params):
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def objective(trial):
    params = build_params(trial)
    val_errors = []
    for train_idx, val_idx in folds:
        model = make_model(params)
        model.fit(X[train_idx], y[train_idx])  
        pred = model.predict(X[val_idx])
        val_errors.append(mean_absolute_error(y[val_idx], pred))
    return float(np.mean(val_errors))

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=N_TRIALS)
print("Best trial:")
print(study.best_trial)

best_params = study.best_params
oof_pred = np.zeros(len(y))
for i, (train_idx, val_idx) in enumerate(folds):
    model = make_model(best_params)
    model.fit(X[train_idx], y[train_idx])
    oof_pred[val_idx] = model.predict(X[val_idx])
    print(f"Fold {i} (KGRP={sorted(np.unique(groups))[i]}) MAE: {mean_absolute_error(y[val_idx], oof_pred[val_idx]):.4f}")

print(f"Overall CV MAE: {mean_absolute_error(y, oof_pred):.4f}")

final_model = make_model(best_params)
final_model.fit(X, y)
joblib.dump({"model": final_model, "features": feature_cols}, "xgb_final.joblib")

