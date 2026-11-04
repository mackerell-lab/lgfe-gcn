import csv
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

MODEL_FILE = "../xgb_model.joblib"   
FEATURE_FILE = "test.txt"       
ACTIVITY_FILE = "exp.csv"          
OUT_FILE = "pred_xgb.csv"         

TARGET_ALIASES = {
}

def get_target(cmpd_id):
    name = str(cmpd_id).split("_")[0].lower()
    return TARGET_ALIASES.get(name, name)

def compute_pc_per_compound(pred, exp):
    pred = np.asarray(pred, dtype=float)
    exp = np.asarray(exp, dtype=float)
    exp_diff = exp[None, :] - exp[:, None]      
    pred_diff = pred[None, :] - pred[:, None]
    valid = exp_diff != 0                       
    correct = (exp_diff * pred_diff > 0) & valid
    total = valid.sum(axis=1)
    return np.where(total > 0, correct.sum(axis=1) / np.maximum(total, 1), np.nan)

saved = joblib.load(MODEL_FILE)
model = saved["model"]
feature_cols = saved["features"]  

feats = pd.read_csv(FEATURE_FILE, sep=r"\s+")
feats["CMPD"] = feats["SDF_name"].astype(str).str.replace(r"\.sdf$", "", regex=True)

act = pd.read_csv(ACTIVITY_FILE)
if "CMPD" not in act.columns:   
    act = pd.read_csv(ACTIVITY_FILE, header=None, names=["CMPD", "DG"])
df = act.merge(feats, on="CMPD", how="inner")
if len(df) < len(act):
    print(f"Warning: {len(act) - len(df)} compounds had no matching features and were dropped.")

X = df[feature_cols].values.astype(np.float32)
true = df["DG"].values.astype(np.float32)
ids = df["CMPD"].tolist()
targets = np.array([get_target(i) for i in ids])

pred = model.predict(X)

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    for row in zip(true.tolist(), pred.tolist(), ids):
        writer.writerow(row)

rows = []
for tgt in sorted(np.unique(targets).tolist()):
    m = targets == tgt
    t, p = true[m], pred[m]
    pcs = compute_pc_per_compound(p, t)
    rows.append(dict(
        target=tgt,
        n=int(m.sum()),
        mae=mean_absolute_error(t, p),
        rmse=root_mean_squared_error(t, p),
        r=pearsonr(t, p)[0] if len(t) > 2 else np.nan,
        pc=float(np.nanmean(pcs)) if np.any(~np.isnan(pcs)) else np.nan, 
    ))
res = pd.DataFrame(rows)

print(f"Compounds : {len(true)}   Targets : {len(res)}")
print("\nPer-target results:")
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

