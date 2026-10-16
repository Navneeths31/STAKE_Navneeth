import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# =========================
# CONFIG - EDIT THESE
# =========================

# Path to STAKE's per-video score CSV (the reference model everything is compared against)
STAKE_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Original\Resnet\STAKE_Resnet_test_per_video_scores.csv"

# Folder containing all baseline/ablation CSVs (one file per model).
# The script will pick up every .csv in this folder EXCEPT the STAKE file itself.
COMPARISON_DIR = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Ablation"

# Metrics to test. Must be column names present in every CSV.
# Set "higher_is_better" per metric so effect size sign is interpretable
# (F1/Precision/Recall: higher = better. MATE/DTW: lower = better -> distance-type metrics).
METRICS = {
    "f1":         {"higher_is_better": True},
    "precision9": {"higher_is_better": True},
    "recall9":    {"higher_is_better": True},
    "mate":       {"higher_is_better": False},
    "dtw":        {"higher_is_better": False},
}

OUTPUT_CSV = os.path.join(COMPARISON_DIR, "stake_significance_results.csv")


# =========================
# HELPERS
# =========================

def load_scores(csv_path):
    """Load a per-video score CSV, keep only 'video' + metric columns, drop rows with NaNs
    in any tested metric (Wilcoxon can't handle NaNs)."""
    df = pd.read_csv(csv_path)
    if "video" not in df.columns:
        raise ValueError(f"'video' column missing in {csv_path}")
    return df


def rank_biserial_effect_size(diffs):
    """Matched-pairs rank-biserial correlation for Wilcoxon signed-rank test.
    diffs: array of paired differences (a - b), zeros already removed.
    Returns a value in [-1, 1]; sign follows the sign of the differences passed in.
    """
    diffs = np.asarray(diffs)
    n = len(diffs)
    if n == 0:
        return np.nan

    abs_diffs = np.abs(diffs)
    ranks = pd.Series(abs_diffs).rank(method="average").values

    pos_sum = ranks[diffs > 0].sum()
    neg_sum = ranks[diffs < 0].sum()
    total = pos_sum + neg_sum

    if total == 0:
        return 0.0

    return (pos_sum - neg_sum) / total


def compare_pair(stake_df, other_df, model_name, metrics):
    """Merge STAKE and one comparison model on 'video', run Wilcoxon per metric."""
    merged = pd.merge(
        stake_df, other_df,
        on="video", suffixes=("_stake", "_other")
    )

    results = []

    for metric, cfg in metrics.items():
        col_stake = f"{metric}_stake"
        col_other = f"{metric}_other"

        if col_stake not in merged.columns or col_other not in merged.columns:
            print(f"  [skip] '{metric}' missing for {model_name}")
            continue

        sub = merged[[col_stake, col_other]].dropna()
        n_paired = len(sub)

        if n_paired < 3:
            print(f"  [skip] '{metric}' for {model_name}: only {n_paired} paired videos (need >=3)")
            continue

        a = sub[col_stake].values
        b = sub[col_other].values
        diffs = a - b

        # If all differences are zero, Wilcoxon errors out -> handle explicitly
        if np.all(diffs == 0):
            stat, p = np.nan, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b, zero_method="wilcox", correction=False)
            except ValueError as e:
                print(f"  [error] Wilcoxon failed for {model_name} / {metric}: {e}")
                continue

        nonzero_diffs = diffs[diffs != 0]
        effect = rank_biserial_effect_size(nonzero_diffs)

        median_diff = np.median(diffs)
        stake_mean = a.mean()
        other_mean = b.mean()

        # Interpret direction relative to "higher/lower is better"
        higher_is_better = cfg["higher_is_better"]
        if higher_is_better:
            stake_better = stake_mean > other_mean
        else:
            stake_better = stake_mean < other_mean

        results.append({
            "model": model_name,
            "metric": metric,
            "n_videos": n_paired,
            "stake_mean": stake_mean,
            "other_mean": other_mean,
            "median_diff": median_diff,
            "wilcoxon_stat": stat,
            "p_value": p,
            "effect_size_rank_biserial": effect,
            "stake_better": stake_better,
        })

    return results


# =========================
# MAIN
# =========================

def main():
    if not os.path.exists(STAKE_CSV):
        raise FileNotFoundError(f"STAKE CSV not found: {STAKE_CSV}")

    stake_df = load_scores(STAKE_CSV)

    csv_files = glob.glob(os.path.join(COMPARISON_DIR, "*.csv"))
    csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(STAKE_CSV)]

    if not csv_files:
        raise FileNotFoundError(f"No comparison CSVs found in {COMPARISON_DIR}")

    print(f"Found {len(csv_files)} comparison models:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    all_results = []

    for csv_path in csv_files:
        model_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\nComparing STAKE vs {model_name} ...")

        other_df = load_scores(csv_path)
        pair_results = compare_pair(stake_df, other_df, model_name, METRICS)
        all_results.extend(pair_results)

    if not all_results:
        print("No valid comparisons produced. Check your CSVs and metric names.")
        return

    results_df = pd.DataFrame(all_results)

    # =========================
    # Multiple comparison correction (Holm-Bonferroni)
    # Correct WITHIN each metric across all models
    # (STAKE vs baseline_1 on F1, STAKE vs baseline_2 on F1, ... all correced together per metric)
    # =========================
    corrected_frames = []
    for metric in results_df["metric"].unique():
        sub = results_df[results_df["metric"] == metric].copy()
        reject, p_corrected, _, _ = multipletests(sub["p_value"].values, method="holm")
        sub["p_value_holm"] = p_corrected
        sub["significant_holm_0.05"] = reject
        corrected_frames.append(sub)

    final_df = pd.concat(corrected_frames, ignore_index=True)
    final_df = final_df.sort_values(["metric", "model"]).reset_index(drop=True)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved full results to: {OUTPUT_CSV}")
    print("\n===== Summary (significant after Holm-Bonferroni correction) =====")
    sig = final_df[final_df["significant_holm_0.05"]]
    if len(sig) == 0:
        print("No comparisons reached significance after correction.")
    else:
        print(sig[["model", "metric", "p_value", "p_value_holm",
                    "effect_size_rank_biserial", "stake_better"]].to_string(index=False))


if __name__ == "__main__":
    main()