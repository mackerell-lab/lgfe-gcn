import numpy as np
import pandas as pd
import random
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

seed = 326187
np.random.seed(seed)
random.seed(seed)

FEATURE_FILE = "features.txt"   
ACTIVITY_FILE = "exp.csv"      
FEATURE_PREFIXES = ("GFE_",)

BEST_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.016032150508650086,
    max_depth=6,
    min_child_weight=2,
    subsample=0.9251383302688995,
    colsample_bytree=0.44068775540927396,
    reg_lambda=0.16036308767296775,
    reg_alpha=0.02436192720533176,
    gamma=0.45577556403490344,
)


def make_model(params):
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def compute_pc(pred, exp):
    n = len(exp)
    if n <= 1:
        return 1.0

    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):

            exp_diff = exp[j] - exp[i]
            pred_diff = pred[j] - pred[i]

            if exp_diff == 0:
                continue

            if exp_diff * pred_diff > 0:
                correct += 1

            total += 1

    return correct / total if total > 0 else np.nan


def load_data(feature_file, activity_file):
    feats = pd.read_csv(feature_file, sep=r"\s+")
    feats["CMPD"] = feats["SDF_name"].astype(str).str.replace(r"\.sdf$", "", regex=True)
    feats = feats.drop(columns=["SDF_name"])

    act = pd.read_csv(activity_file)
    df = act.merge(feats, on="CMPD", how="inner")
    if len(df) < len(act):
        print(f"Warning: {len(act) - len(df)} compounds had no matching features and were dropped.")

    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    X = df[feature_cols].values.astype(np.float32)
    y = df["DG"].values.astype(np.float32)
    groups = df["KGRP"].values
    return df, X, y, groups, feature_cols


df, X, y, groups, feature_cols = load_data(FEATURE_FILE, ACTIVITY_FILE)
print(f"Loaded {len(df)} compounds, {len(feature_cols)} features")

oof_pred = np.zeros(len(y))
fold_mae, fold_rmse, fold_r, fold_pc = [], [], [], []
for g in sorted(np.unique(groups)):
    val = groups == g
    m = make_model(BEST_PARAMS)
    m.fit(X[~val], y[~val])
    oof_pred[val] = m.predict(X[val])
    mae = mean_absolute_error(y[val], oof_pred[val])
    rmse = root_mean_squared_error(y[val], oof_pred[val])
    r = pearsonr(y[val], oof_pred[val])[0]
    pc = compute_pc(oof_pred[val], y[val])
    fold_mae.append(mae)
    fold_rmse.append(rmse)
    fold_r.append(r)
    fold_pc.append(pc)
    print(f"KGRP {g}: MAE {mae:.4f}  RMSE {rmse:.4f}  Pearson r {r:.4f}  PC {pc:.4f}")

# Mean +/- sample standard deviation (ddof=1) across the folds
print(f"CV MAE : {np.mean(fold_mae):.4f} +/- {np.std(fold_mae, ddof=1):.4f}")
print(f"CV RMSE: {np.mean(fold_rmse):.4f} +/- {np.std(fold_rmse, ddof=1):.4f}")
print(f"CV Pearson r: {np.mean(fold_r):.4f} +/- {np.std(fold_r, ddof=1):.4f}")
print(f"CV PC: {np.mean(fold_pc):.4f} +/- {np.std(fold_pc, ddof=1):.4f}")

model = make_model(BEST_PARAMS)
model.fit(X, y)
joblib.dump({"model": model, "features": feature_cols}, "xgb_model.joblib")

pred = model.predict(X)
print(f"Train MAE : {mean_absolute_error(y, pred):.4f}")
print(f"Train RMSE: {root_mean_squared_error(y, pred):.4f}")
print(f"Train Pearson r: {pearsonr(y, pred)[0]:.4f}")
print(f"Train PC: {compute_pc(pred, y):.4f}")

