#!/usr/bin/env python3

import numpy as np
import argparse
from scipy.stats import pearsonr


def top_k(x, k):
    return set(np.argsort(x, kind="stable")[-k:])


def bottom_k(x, k):
    return set(np.argsort(x, kind="stable")[:k])


def ef(hits, k, n):
    return (hits * n) / (k * k) if k > 0 else 0.0


def enrichment_metrics(pred, exp, k_values):

    n = len(pred)
    results = {}

    for k in k_values:

        k_eff = min(k, n)

        pred_A = bottom_k(pred, k_eff)
        exp_A = bottom_k(exp, k_eff)
        A_hits = len(pred_A & exp_A)

        pred_B = top_k(pred, k_eff)
        exp_B = top_k(exp, k_eff)
        B_hits = len(pred_B & exp_B)

        results[k] = {
            "A_EF": ef(A_hits, k_eff, n),
            "A_hits": A_hits,
            "B_EF": ef(B_hits, k_eff, n),
            "B_hits": B_hits,
        }

    return results

def classification_metrics(pred, exp):

    # Exclude experimentally neutral compounds
    mask = exp != 0
    pred = pred[mask]
    exp = exp[mask]

    true_A = exp < 0
    true_B = exp > 0

    pred_A = pred < 0
    pred_B = pred >= 0

    TP = np.sum(true_A & pred_A)
    FN = np.sum(true_A & pred_B)
    TN = np.sum(true_B & pred_B)
    FP = np.sum(true_B & pred_A)

    n_A = np.sum(true_A)
    n_B = np.sum(true_B)
    n_class = n_A + n_B

    accuracy = (
        (TP + TN) / n_class
        if n_class > 0 else np.nan
    )

    TPR = TP / n_A if n_A > 0 else np.nan
    TNR = TN / n_B if n_B > 0 else np.nan

    BA = (
        (TPR + TNR) / 2
        if n_A > 0 and n_B > 0
        else np.nan
    )

    majority_accuracy = (
        max(n_A, n_B) / n_class
        if n_class > 0
        else np.nan
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "n_A": n_A,
        "n_B": n_B,
        "n_class": n_class,
        "accuracy": accuracy,
        "TPR": TPR,
        "TNR": TNR,
        "BA": BA,
        "majority_accuracy": majority_accuracy,
    }


def bootstrap_classification(pred, exp, n_boot, rng):

    idx_A = np.where(exp < 0)[0]
    idx_B = np.where(exp > 0)[0]

    metrics = {
        "TPR": [],
        "TNR": [],
        "Accuracy": [],
        "BA": [],
    }

    for _ in range(n_boot):

        boot_indices = []

        if len(idx_A) > 0:
            boot_A = rng.choice(
                idx_A, size=len(idx_A), replace=True
            )
            boot_indices.extend(boot_A)

        if len(idx_B) > 0:
            boot_B = rng.choice(
                idx_B, size=len(idx_B), replace=True
            )
            boot_indices.extend(boot_B)

        idx = np.asarray(boot_indices, dtype=int)

        result = classification_metrics(
            pred[idx], exp[idx]
        )

        if np.isfinite(result["TPR"]):
            metrics["TPR"].append(result["TPR"])

        if np.isfinite(result["TNR"]):
            metrics["TNR"].append(result["TNR"])

        if np.isfinite(result["accuracy"]):
            metrics["Accuracy"].append(result["accuracy"])

        if np.isfinite(result["BA"]):
            metrics["BA"].append(result["BA"])

    return {
        name: np.percentile(values, [2.5, 97.5])
        for name, values in metrics.items()
        if len(values) > 0
    }


def bootstrap_regression(pred, exp, n_boot, rng):

    rmse_values = []
    pearson_values = []

    n = len(pred)

    for _ in range(n_boot):

        idx = rng.choice(n, size=n, replace=True)

        p = pred[idx]
        e = exp[idx]

        rmse = np.sqrt(np.mean((p - e) ** 2))
        rmse_values.append(rmse)

        if np.std(p) > 0 and np.std(e) > 0:

            pr = pearsonr(p, e)[0]

            if np.isfinite(pr):
                pearson_values.append(pr)

    return {
        "RMSE": np.percentile(
            rmse_values, [2.5, 97.5]
        ),
        "Pearson": (
            np.percentile(pearson_values, [2.5, 97.5])
            if pearson_values
            else np.array([np.nan, np.nan])
        )
    }


def bootstrap_enrichment(pred, exp, k_values, n_boot, rng):

    metrics = {
        (k, direction): []
        for k in k_values
        for direction in ["A_EF", "B_EF"]
    }

    n = len(pred)

    for _ in range(n_boot):

        idx = rng.choice(n, size=n, replace=True)

        boot_results = enrichment_metrics(
            pred[idx], exp[idx], k_values
        )

        for k in k_values:

            metrics[(k, "A_EF")].append(
                boot_results[k]["A_EF"]
            )

            metrics[(k, "B_EF")].append(
                boot_results[k]["B_EF"]
            )

    return {
        key: np.percentile(values, [2.5, 97.5])
        for key, values in metrics.items()
    }

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data = np.atleast_2d(np.loadtxt(args.input_file))

    if data.shape[1] < 2:
        raise ValueError("Need two columns: pred exp")

    pred = data[:, 0]
    exp = data[:, 1]

    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(exp)):
        raise ValueError("Input contains NaN or infinite values.")

    n = len(pred)

    if n == 0:
        raise ValueError("Input file contains no samples.")

    k_values = [1, 5, 10]

    cls = classification_metrics(pred, exp)

    n_A = cls["n_A"]
    n_B = cls["n_B"]

    if n_A > 0 and n_B > 0:
        dataset_type = "MIXED"
    elif n_A > 0:
        dataset_type = "A-ONLY"
    elif n_B > 0:
        dataset_type = "B-ONLY"
    else:
        dataset_type = "UNDETERMINED"

    print("\n=== MODEL EVALUATION ===\n")

    print(f"Dataset type            : {dataset_type}")
    print(f"Total samples            : {n}")
    print(f"Positive (A-selective)   : {n_A}")
    print(f"Negative (B-selective)   : {n_B}")
    print(f"Neutral experimental     : {n - n_A - n_B}")

    print("\nConfusion Matrix")
    print("----------------")
    print(f"TP : {cls['TP']}")
    print(f"TN : {cls['TN']}")
    print(f"FP : {cls['FP']}")
    print(f"FN : {cls['FN']}")

    print("\nClassification Metrics")
    print("----------------------")

    print(f"Accuracy                : {cls['accuracy']:.4f}")
    print(f"Majority-class baseline : {cls['majority_accuracy']:.4f}")

    if n_A > 0:
        print(f"TPR                     : {cls['TPR']:.4f}")
    else:
        print("TPR                     : N/A")

    if n_B > 0:
        print(f"TNR                     : {cls['TNR']:.4f}")
    else:
        print("TNR                     : N/A")

    if np.isfinite(cls["BA"]):
        print(f"Balanced accuracy       : {cls['BA']:.4f}")
    else:
        print("Balanced accuracy       : N/A")

    cls_ci = bootstrap_classification(
        pred, exp, args.bootstrap, rng
    )

    print("\nClassification 95% Bootstrap CI")
    print("--------------------------------")

    value_map = {
        "TPR": "TPR",
        "TNR": "TNR",
        "Accuracy": "accuracy",
        "BA": "BA"
    }

    for metric in ["TPR", "TNR", "Accuracy", "BA"]:

        if metric in cls_ci:

            ci = cls_ci[metric]
            value = cls[value_map[metric]]

            print(
                f"{metric:<12}: {value:.4f} "
                f"(95% CI: {ci[0]:.4f}-{ci[1]:.4f})"
            )

        else:
            print(f"{metric:<12}: N/A")

    rmse = np.sqrt(np.mean((pred - exp) ** 2))

    if np.std(pred) > 0 and np.std(exp) > 0:
        pearson_corr, pearson_p = pearsonr(pred, exp)
    else:
        pearson_corr, pearson_p = np.nan, np.nan

    reg_ci = bootstrap_regression(
        pred, exp, args.bootstrap, rng
    )

    print("\nRegression and Correlation Metrics")
    print("-----------------------------------")

    print(
        f"RMSE       : {rmse:.4f} "
        f"(95% CI: {reg_ci['RMSE'][0]:.4f}-"
        f"{reg_ci['RMSE'][1]:.4f})"
    )

    print(
        f"Pearson R  : {pearson_corr:.4f} "
        f"(95% CI: {reg_ci['Pearson'][0]:.4f}-"
        f"{reg_ci['Pearson'][1]:.4f})"
    )

    print(f"Pearson p-value: {pearson_p:.4e}")

    results = enrichment_metrics(
        pred, exp, k_values
    )

    ef_ci = bootstrap_enrichment(
        pred, exp, k_values, args.bootstrap, rng
    )

    print("\nEnrichment (EF@k + Hits)")
    print("------------------------")

    # Only report enrichment for experimentally
    # represented selectivity classes.

    directions = []

    if n_A > 0:
        directions.append("A")

    if n_B > 0:
        directions.append("B")

    if not directions:
        print("Not calculated: no selective compounds.")

    for direction in directions:

        print(f"\n{direction}-selective enrichment")

        for k in k_values:

            ef_value = results[k][f"{direction}_EF"]
            hits = results[k][f"{direction}_hits"]

            ci = ef_ci[(k, f"{direction}_EF")]

            print(
                f"EF@{k:<2} = {ef_value:.3f} "
                f"(95% CI: {ci[0]:.3f}-{ci[1]:.3f}), "
                f"hits = {hits}/{min(k,n)}"
            )

    print("\n")


if __name__ == "__main__":
    main()

