# -*- coding: utf-8 -*-


from __future__ import annotations

import ast
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# Shared experiment defaults.
RANDOM_SEED = 20260513
NESTED_CV_REPEATS = 5
N_PERMUTATIONS = 1000
FIGURE_FORMATS = ("png", "pdf")

# Shared SVM pipeline and hyperparameter grid.
SVM_PIPELINE = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(class_weight="balanced", random_state=RANDOM_SEED)),
    ]
)
_SVM_C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
_SVM_GAMMA_GRID = ["scale", "auto", 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

SVM_PARAM_GRID = [
    {"svc__kernel": ["linear"], "svc__C": _SVM_C_GRID},
    {"svc__kernel": ["rbf"], "svc__C": _SVM_C_GRID, "svc__gamma": _SVM_GAMMA_GRID},
]


# ──────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def make_result_dirs(base_dir: str) -> dict[str, str]:
    dirs = {
        "root": base_dir,
        "svm_tables": base_dir,
        "svm_figures": base_dir,
        "svm_arrays": base_dir,
    }
    ensure_dirs(*dirs.values())
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# SVM GridSearch helper
# ──────────────────────────────────────────────────────────────────────────────

def make_svm_search(inner_splits: int, cv_seed: int, n_jobs: int = -1) -> GridSearchCV:
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=cv_seed)
    return GridSearchCV(
        estimator=SVM_PIPELINE,
        param_grid=SVM_PARAM_GRID,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Primary nested CV unit used by repeated nested CV and permutation tests
# ──────────────────────────────────────────────────────────────────────────────

def run_primary_nested_cv(
    X_df: pd.DataFrame,
    y: pd.Series,
    sample_ids: pd.Series,
    label_order: list[str],
    outer_splits: int,
    cv_seed: int,
    run_label: str,
    n_jobs: int = -1,
) -> dict:
    """
    Run one complete nested-CV pass.

    Outer loop: out-of-fold evaluation.
    Inner loop: SVM hyperparameter selection by GridSearchCV using macro-F1.

    Returns pooled outer-test predictions and metrics across all outer folds.
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=cv_seed)
    y_true, y_pred, records = [], [], []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y), start=1):
        X_train = X_df.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        inner_splits = min(5, int(y_train.value_counts().min()))
        if inner_splits < 2:
            raise ValueError(f"Inner CV cannot be built in {run_label}, fold {fold_id}.")

        search = make_svm_search(inner_splits, cv_seed + fold_id, n_jobs=n_jobs)
        search.fit(X_train, y_train)
        fold_pred = search.best_estimator_.predict(X_test)

        y_true.extend(y_test)
        y_pred.extend(fold_pred)

        for local_i, sample_pos in enumerate(test_idx):
            records.append(
                {
                    "run_label": run_label,
                    "fold": fold_id,
                    "sample_id": sample_ids.iloc[sample_pos],
                    "true_label": y_test.iloc[local_i],
                    "pred_label": fold_pred[local_i],
                    "correct": y_test.iloc[local_i] == fold_pred[local_i],
                    "inner_cv_splits": inner_splits,
                    "inner_cv_best_f1_macro": search.best_score_,
                    "best_params": str(search.best_params_),
                }
            )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "run_label": run_label,
        "cv_seed": cv_seed,
        "y_true": y_true,
        "y_pred": y_pred,
        "records": pd.DataFrame(records),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=label_order,
            average="macro",
            zero_division=0,
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# nested-CV macro-F1 permutation test
# ──────────────────────────────────────────────────────────────────────────────

def run_nested_cv_macro_f1_permutation_test(
    X_df: pd.DataFrame,
    y: pd.Series,
    label_order: list[str],
    observed_macro_f1: float,
    outer_splits: int,
    n_permutations: int,
    random_state: int,
    n_jobs: int = -1,
) -> tuple[float, np.ndarray, float]:
    """
    Permutation test for single-pass nested-CV pooled macro-F1.

    For each permutation:
        1. Shuffle labels.
        2. Run one primary nested CV.
        3. Record pooled macro-F1.

    The p-value is computed by comparing permuted macro-F1 scores
    against the observed single-pass nested-CV macro-F1.
    """
    rng = np.random.default_rng(random_state)
    y_values = y.to_numpy()
    perm_seeds = rng.integers(
        0,
        np.iinfo(np.int32).max,
        size=n_permutations,
        endpoint=False,
    )

    def score_one_permutation(seed: int) -> float:
        local_rng = np.random.default_rng(int(seed))
        perm_y = pd.Series(
            local_rng.permutation(y_values),
            index=y.index,
        )

        result = run_primary_nested_cv(
            X_df=X_df,
            y=perm_y,
            sample_ids=pd.Series(perm_y.index.astype(str), index=perm_y.index),
            outer_splits=outer_splits,
            cv_seed=random_state,
            run_label="permutation",
            label_order=label_order,
            n_jobs=1,
        )

        return result["macro_f1"]

    perm_scores = Parallel(n_jobs=n_jobs)(
        delayed(score_one_permutation)(seed)
        for seed in perm_seeds
    )

    perm_scores = np.asarray(perm_scores, dtype=float)
    p_value = (1 + np.sum(perm_scores >= observed_macro_f1)) / (len(perm_scores) + 1)
    return observed_macro_f1, perm_scores, float(p_value) 

# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix plotting helper
# ──────────────────────────────────────────────────────────────────────────────

def draw_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: list[str],
    save_stem: str,
    title: str,
    formats: tuple[str, ...] = FIGURE_FORMATS,
) -> dict[str, float]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_labels = set(pd.Series(y_true).astype(str).unique().tolist())
    labels = [lab for lab in label_order if lab in unique_labels]
    labels += [lab for lab in sorted(unique_labels, key=str) if lab not in set(labels)]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_labels = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, n_labels * 1.9), max(3.6, n_labels * 1.7)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.6,
        linecolor="gray",
        annot_kws={"size": 14, "fontweight": "bold"},
        ax=ax,
        cbar=False,
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"{title}\nAccuracy={acc:.3f}; Macro-F1={macro_f1:.3f}", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    for fmt in formats:
        path = f"{save_stem}.{fmt}"
        plt.savefig(path, dpi=300, bbox_inches="tight", format=fmt)
        print(f"Saved {path}")
    plt.close(fig)

    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


# ──────────────────────────────────────────────────────────────────────────────
# Parameter stability, fixed-parameter evaluation, and final model
# ──────────────────────────────────────────────────────────────────────────────

def _param_stability_and_final_model(
    all_records: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    output_dir: str,
    prefix: str,
    cv_configs: list[tuple[int, int]] | None = None,
    perm_p: float | None = None,
    label_order: list[str] | None = None,
    n_permutations: int = 1000,
    sample_ids: pd.Series | np.ndarray | None = None,
):
    """
    Post-analysis and final model module.

    Outputs:
    - Table 1: per outer-fold detail across repeated nested CV.
    - Table 2: parameter stability summary.
    - Final model: retrained on all data using the most-selected parameter set.
    - Fixed-params repeated outer CV.
    - Fixed-params permutation test.
    - Representative fixed-params confusion matrix.
    - Final model summary CSV and .pkl model.
    """

    def _parse_params(s: str) -> dict:
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return {}

    if label_order is None:
        label_order = sorted(pd.Series(y).astype(str).unique().tolist())

    # ── Table 1: per-fold detail ──────────────────────────────────────────────
    print(f"Running Table1")
    rows = []
    for (rl, fid), g in all_records.groupby(["run_label", "fold"], sort=True):
        yt = g["true_label"].values
        yp = g["pred_label"].values
        rows.append(
            {
                "repeat": rl,
                "fold": int(fid),
                "n_test": len(g),
                "fold_accuracy": round(accuracy_score(yt, yp), 4),
                "fold_macro_f1": round(
                    f1_score(yt, yp, labels=label_order, average="macro", zero_division=0),
                    4,
                ),
                "inner_cv_best_f1_macro": round(float(g["inner_cv_best_f1_macro"].iloc[0]), 4),
                "best_params": g["best_params"].iloc[0],
            }
        )
    t1 = pd.DataFrame(rows)
    t1_path = os.path.join(output_dir, f"{prefix}_Table1_PerFoldDetail.csv")
    t1.to_csv(t1_path, index=False)

    # ── Table 2: parameter stability ─────────────────────────────────────────
    print(f"Running Table2")
    t1["_pk"] = t1["best_params"].apply(
        lambda s: json.dumps(_parse_params(s), sort_keys=True, default=str)
    )
    t2 = (
        t1.groupby("_pk")
        .agg(
            selection_count=("_pk", "count"),
            mean_fold_macro_f1=("fold_macro_f1", "mean"),
            mean_inner_cv_f1=("inner_cv_best_f1_macro", "mean"),
        )
        .reset_index()
        .rename(columns={"_pk": "params"})
    )
    t2["selection_frequency"] = (t2["selection_count"] / len(t1)).round(4)
    t2["mean_fold_macro_f1"] = t2["mean_fold_macro_f1"].round(4)
    t2["mean_inner_cv_f1"] = t2["mean_inner_cv_f1"].round(4)
    t2 = t2.sort_values(["selection_count", "mean_fold_macro_f1"], ascending=False).reset_index(drop=True)
    t2_path = os.path.join(output_dir, f"{prefix}_Table2_ParamStability.csv")
    t2.to_csv(t2_path, index=False)

    print("─" * 70)
    print(f"[Param Stability] {prefix}  ({len(t1)} total CV folds across all repeats)")
    print(f"  Table 1 (per-fold detail)  → {t1_path}")
    print(f"  Table 2 (param stability)  → {t2_path}")
    print("\n  Top parameter combinations:")
    for _, r in t2.head(5).iterrows():
        print(
            f"    {int(r.selection_count):3d}/{len(t1)} ({r.selection_frequency:.0%})"
            f"  fold_F1={r.mean_fold_macro_f1:.3f}"
            f"  inner_F1={r.mean_inner_cv_f1:.3f}"
            f"  → {r.params}"
        )

    # ── Final model: retrain on all data with most-selected params ───────────
    print("\n" + "─" * 70)
    print(f"Running for final model")

    best_p = _parse_params(t2.iloc[0]["params"])
    final_pipe = clone(pipeline)
    final_pipe.set_params(**best_p)

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    final_pipe.fit(X_arr, y_arr)

    if sample_ids is not None:
        sid_arr = np.asarray(sample_ids)
    elif hasattr(X, "index"):
        sid_arr = np.asarray(X.index)
    else:
        sid_arr = np.arange(len(y_arr))

    # ── Fixed-params repeated outer CV ───────────────────────────────────────
    print(f"Running Fixed-params repeated outer CV")
    mean_acc = std_acc = mean_f1 = std_f1 = rep_idx = None
    fixed_perm_p = None
    all_pred_records = []

    if cv_configs is not None:
        rep_accs, rep_f1s, rep_yt_list, rep_yp_list = [], [], [], []

        for repeat_id, (n_splits, seed) in enumerate(cv_configs, start=1):
            outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            r_yt, r_yp, r_fold_ids, r_test_idxs = [], [], [], []

            for fold_id, (tr_idx, te_idx) in enumerate(outer.split(X_arr, y_arr), start=1):
                pipe_i = clone(pipeline)
                pipe_i.set_params(**best_p)
                pipe_i.fit(X_arr[tr_idx], y_arr[tr_idx])
                fold_preds = pipe_i.predict(X_arr[te_idx])

                r_yt.extend(y_arr[te_idx].tolist())
                r_yp.extend(fold_preds.tolist())
                r_fold_ids.extend([fold_id] * len(te_idx))
                r_test_idxs.extend(te_idx.tolist())

            r_yt = np.asarray(r_yt)
            r_yp = np.asarray(r_yp)
            rep_accs.append(round(accuracy_score(r_yt, r_yp), 4))
            rep_f1s.append(round(f1_score(r_yt, r_yp, labels=label_order, average="macro", zero_division=0), 4))
            rep_yt_list.append(r_yt)
            rep_yp_list.append(r_yp)

            for pos, (te_i, fold_i) in enumerate(zip(r_test_idxs, r_fold_ids)):
                all_pred_records.append(
                    {
                        "sample_id": sid_arr[te_i],
                        "true_label": r_yt[pos],
                        "pred_label": r_yp[pos],
                        "correct": r_yt[pos] == r_yp[pos],
                        "repeat_id": repeat_id,
                        "fold_id": fold_i,
                        "is_representative_repeat": False,
                    }
                )

        mean_acc = float(np.mean(rep_accs))
        std_acc = float(np.std(rep_accs, ddof=1) if len(rep_accs) > 1 else 0.0)
        mean_f1 = float(np.mean(rep_f1s))
        std_f1 = float(np.std(rep_f1s, ddof=1) if len(rep_f1s) > 1 else 0.0)
        rep_idx = int(np.argmin(np.abs(np.asarray(rep_f1s) - mean_f1)))

        ev_yt = rep_yt_list[rep_idx]
        ev_yp = rep_yp_list[rep_idx]
        ev_acc = rep_accs[rep_idx]
        ev_f1 = rep_f1s[rep_idx]
        ev_n_rep = len(cv_configs)
        ev_n_folds = cv_configs[0][0]
        ev_tag = (
            f"Fixed-params {ev_n_rep}-repeat {ev_n_folds}-fold outer CV "
            f"(representative repeat {rep_idx + 1})"
        )

        rep_number = rep_idx + 1
        for rec in all_pred_records:
            rec["is_representative_repeat"] = rec["repeat_id"] == rep_number

        pred_csv = os.path.join(output_dir, f"{prefix}_FixedParams_RepeatedOuterCV_predictions.csv")
        pd.DataFrame(all_pred_records).to_csv(pred_csv, index=False)
        print(f"[{prefix}] Fixed-params per-sample predictions → {pred_csv}")

        # ── Fixed-params permutation test ─────────────────────────────────────
        if n_permutations > 0:
            print(f"[{prefix}] Running fixed-params permutation test (n={n_permutations})...")
            fp_rng = np.random.default_rng(RANDOM_SEED)
            fp_seeds = fp_rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, endpoint=False)

            def score_one_fixed_perm(fp_seed: int) -> float:
                local_rng = np.random.default_rng(int(fp_seed))
                perm_y = local_rng.permutation(y_arr)
                perm_rep_f1s = []

                for n_splits2, seed_cv2 in cv_configs:
                    outer2 = StratifiedKFold(n_splits=n_splits2, shuffle=True, random_state=seed_cv2)
                    r_yt2, r_yp2 = [], []
                    for tr2, te2 in outer2.split(X_arr, perm_y):
                        pipe2 = clone(pipeline)
                        pipe2.set_params(**best_p)
                        pipe2.fit(X_arr[tr2], perm_y[tr2])
                        r_yt2.extend(perm_y[te2].tolist())
                        r_yp2.extend(pipe2.predict(X_arr[te2]).tolist())
                    perm_rep_f1s.append(
                        f1_score(
                            np.asarray(r_yt2),
                            np.asarray(r_yp2),
                            labels=label_order,
                            average="macro",
                            zero_division=0,
                        )
                    )
                return float(np.mean(perm_rep_f1s))

            fp_scores = Parallel(n_jobs=-1)(
                delayed(score_one_fixed_perm)(seed) for seed in fp_seeds
            )
            fp_scores = np.asarray(fp_scores, dtype=float)
            fixed_perm_p = float((1 + np.sum(fp_scores >= mean_f1)) / (n_permutations + 1))
            fp_null_path = os.path.join(output_dir, f"{prefix}_FixedParams_permutation_null.npy")
            np.save(fp_null_path, fp_scores)
            print(f"[{prefix}] Fixed-params perm p={fixed_perm_p:.4f}  →  {fp_null_path}")

        # ── Representative fixed-params confusion matrix ─────────────────────
        cm_stem = os.path.join(output_dir, f"{prefix}_FixedParams_OuterCV_ConfusionMatrix")
        cm_title = (
            f"{prefix} · {ev_tag}\n"
            f"This repeat: Acc={ev_acc:.3f}; F1={ev_f1:.3f}; "
            f"Repeated CV: Acc={mean_acc:.3f}±{std_acc:.3f}; F1={mean_f1:.3f}±{std_f1:.3f}"
        )
        if fixed_perm_p is not None:
            cm_title += f"; fixed-param perm p={fixed_perm_p:.4f}"
        draw_confusion_matrix(ev_yt, ev_yp, label_order, cm_stem, cm_title)

    else:
        # Fallback only; this is optimistic because it predicts the training data.
        ev_yp = final_pipe.predict(X_arr)
        ev_yt = y_arr
        ev_acc = round(accuracy_score(ev_yt, ev_yp), 4)
        ev_f1 = round(f1_score(ev_yt, ev_yp, labels=label_order, average="macro", zero_division=0), 4)
        ev_tag = "Full-data in-sample (optimistic)"
        cm_stem = os.path.join(output_dir, f"{prefix}_FinalModel_FullData_ConfusionMatrix")
        draw_confusion_matrix(ev_yt, ev_yp, label_order, cm_stem, f"{prefix} · {ev_tag}")

    # ── Representative/final classification report ──────────────────────────
    report = classification_report(
        ev_yt,
        ev_yp,
        labels=label_order,
        target_names=label_order,
        zero_division=0,
        output_dict=True,
    )
    report_path = os.path.join(output_dir, f"{prefix}_FinalModel_Representative_ClassificationReport.csv")
    pd.DataFrame(report).T.to_csv(report_path)
    print(f"[{prefix}] Representative classification report → {report_path}")

    # ── Final summary CSV ────────────────────────────────────────────────────
    final_summary = {
                "selected_params": t2.iloc[0]["params"],
                "selection_count": int(t2.iloc[0]["selection_count"]),
                "selection_frequency": t2.iloc[0]["selection_frequency"],
                "cv_mean_fold_macro_f1": t2.iloc[0]["mean_fold_macro_f1"],
                "total_cv_folds": len(t1),
                "n_training_samples": int(len(y_arr)),
                "class_counts": str(pd.Series(y_arr).value_counts().to_dict()),
                "nested_cv_permutation_p": round(perm_p, 4) if perm_p is not None else None,
                "fixed_params_representative_repeat": int(rep_idx + 1) if rep_idx is not None else None,
                "fixed_params_observed_mean_macro_f1": round(mean_f1, 4) if mean_f1 is not None else None,
                "fixed_params_observed_std_macro_f1": round(std_f1, 4) if std_f1 is not None else None,
                "fixed_params_observed_mean_accuracy": round(mean_acc, 4) if mean_acc is not None else None,
                "fixed_params_observed_std_accuracy": round(std_acc, 4) if std_acc is not None else None,
                "fixed_params_representative_repeat_macro_f1": ev_f1,
                "fixed_params_representative_repeat_accuracy": ev_acc,
                "fixed_params_repeated_outer_cv_permutation_p": round(fixed_perm_p, 4) if fixed_perm_p is not None else None,
                "fixed_params_permutation_statistic": "mean macro-F1 across repeated fixed-parameter outer CV",
                "eval_method": ev_tag,
            }
    fm_csv = os.path.join(output_dir, f"{prefix}_FinalModel_Summary.txt")
    with open(fm_csv, "w", encoding="utf-8") as f:
        for key, value in final_summary.items():
            f.write(f"{key}: {value}\n")

    pkl = os.path.join(output_dir, f"{prefix}_FinalModel.pkl")
    try:
        joblib.dump(final_pipe, pkl)
        print(f"\n  Final model .pkl  → {pkl}")
    except Exception as exc:
        print(f"\n  [joblib save skipped: {exc}]")

    print(f"  Final model CSV   → {fm_csv}")
    print(f"  Params selected:  {best_p}")
    print(f"  Eval accuracy  ({ev_tag}): {ev_acc:.2%}")
    print(f"  Eval macro-F1  ({ev_tag}): {ev_f1:.4f}")

    return t1, t2, final_pipe


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_svm_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    order: list[str],
    analysis_name: str,
    output_dir: str,
    output_prefix: str,
    outer_repeats: int = NESTED_CV_REPEATS,
    n_permutations: int = N_PERMUTATIONS,
    note: str = "",
) -> dict:
    """
    Run the refined SVM evaluation pipeline.

    Main outputs:
    - repeated nested-CV stability table and summary;
    - repeated nested-CV permutation null distribution;
    - parameter-stability tables;
    - fixed-parameter repeated outer-CV prediction table;
    - final representative confusion matrix;
    - final classification report;
    - final model summary;
    - final trained model .pkl.
    """
    X_df = df[feature_cols].copy()
    y = df[label_col].astype(str).copy()
    sample_ids = df["sample_id"].astype(str).copy()
    label_order = [str(x) for x in order]

    min_class_n = int(y.value_counts().min())
    if min_class_n < 2:
        raise ValueError(f"{analysis_name}: each class needs at least two samples.")
    outer_splits = min(5, min_class_n)
    result_dirs = make_result_dirs(output_dir)

    print("\n" + "=" * 80)
    print(f"Running repeated nested-CV SVM: {analysis_name}")
    print("=" * 80)
    print(f"Class counts:  {y.value_counts().reindex(label_order).to_dict()}")
    print(f"Features used: {len(feature_cols)} ({', '.join(feature_cols)})")
    print(f"Outer CV:      {outer_splits}-fold × {outer_repeats} repeated nested CV")

    # ── Repeated nested CV ───────────────────────────────────────────────────
    nested_results = []
    for repeat_id in range(1, outer_repeats + 1):
        nested_results.append(
            run_primary_nested_cv(
                X_df,
                y,
                sample_ids=sample_ids,
                label_order=label_order,
                outer_splits=outer_splits,
                cv_seed=RANDOM_SEED + repeat_id - 1,
                run_label=f"repeat_{repeat_id:02d}",
                n_jobs=-1,
            )
        )

    stability_df = pd.DataFrame(
        [
            {
                "run_label": item["run_label"],
                "cv_seed": item["cv_seed"],
                "accuracy": item["accuracy"],
                "balanced_accuracy": item["balanced_accuracy"],
                "macro_f1": item["macro_f1"],
            }
            for item in nested_results
        ]
    )
    stability_out = os.path.join(result_dirs["svm_tables"], f"{output_prefix}_RepeatedNestedCV_stability.csv")
    stability_df.to_csv(stability_out, index=False)

    observed_mean_macro_f1 = float(stability_df["macro_f1"].mean())

    # ── Repeated nested-CV permutation test ──────────────────────────────────
    print(f"Running nested-CV macro-F1 permutation test (n={n_permutations})...")
    perm_score, perm_scores, perm_p = run_nested_cv_macro_f1_permutation_test(
        X_df=X_df,
        y=y,
        label_order=label_order,
        observed_macro_f1=observed_mean_macro_f1,
        outer_splits=outer_splits,
        n_permutations=n_permutations,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    perm_null_out = os.path.join(result_dirs["svm_arrays"], f"{output_prefix}_RepeatedNestedCV_permutation_null.npy")
    np.save(perm_null_out, perm_scores)
    print("P value of nested-CV macro-F1 permutation test: "+ str(perm_p))

    # ── Main repeated nested-CV summary ──────────────────────────────────────
    summary = {
                "analysis": analysis_name,
                "n_features": len(feature_cols),
                "features": ", ".join(feature_cols),
                "repeated_nested_cv": f"{outer_splits}-fold × {outer_repeats} repeated nested CV",
                "stability_accuracy_mean": stability_df["accuracy"].mean(),
                "stability_accuracy_std": stability_df["accuracy"].std(ddof=1),
                "stability_balanced_accuracy_mean": stability_df["balanced_accuracy"].mean(),
                "stability_balanced_accuracy_std": stability_df["balanced_accuracy"].std(ddof=1),
                "stability_macro_f1_mean": stability_df["macro_f1"].mean(),
                "stability_macro_f1_std": stability_df["macro_f1"].std(ddof=1),
                "permutation_score_mean_macro_f1": perm_score,
                "permutation_p_value": perm_p,
                "n_permutations": n_permutations,
                "permutation_statistic": "mean macro-F1 across repeated nested CV",
                "note": note,
            }
    summary_out = os.path.join(result_dirs["svm_tables"], f"{output_prefix}_RepeatedNestedCV_summary.txt")
    with open(summary_out, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    # ── Parameter stability + final model ────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Running Parameter stability testing")
    print("=" * 80)

    all_records = pd.concat([r["records"] for r in nested_results], ignore_index=True)
    cv_configs = [(outer_splits, RANDOM_SEED + i) for i in range(outer_repeats)]
    _param_stability_and_final_model(
        all_records,
        X_df,
        y,
        SVM_PIPELINE,
        output_dir,
        output_prefix,
        cv_configs=cv_configs,
        perm_p=perm_p,
        label_order=label_order,
        n_permutations=n_permutations,
        sample_ids=sample_ids,
    )
