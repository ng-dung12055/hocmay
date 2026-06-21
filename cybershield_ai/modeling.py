from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import FEATURE_COLUMNS

matplotlib.use("Agg")

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - compatibility with older sklearn
    try:
        from sklearn.calibration import FrozenEstimator
    except ImportError:  # pragma: no cover
        FrozenEstimator = None


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Tạo bộ mô hình ứng viên để so sánh hiệu suất.

    Trả về ba pipeline: Logistic Regression (có StandardScaler),
    SVM-RBF (có StandardScaler), và Random Forest (không cần scaler
    vì thuật toán dựa trên cây). LR và SVM dùng ``class_weight='balanced'``
    để xử lý dữ liệu mất cân bằng nhãn.

    Args:
        random_state: Seed cho reproducibility.

    Returns:
        Dictionary mapping tên mô hình đến đối tượng estimator/pipeline.
    """
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
    }


def build_baseline_model(random_state: int = 42) -> dict[str, Any]:
    """Tạo các mô hình baseline để so sánh với candidate models.
    
    Sử dụng DummyClassifier với các chiến lược đơn giản (most_frequent,
    stratified) để thiết lập ngưỡng hiệu suất tối thiểu mà bất kỳ mô
    hình ML nào cũng phải vượt qua.
    
    Args:
        random_state: Seed cho reproducibility.
    
    Returns:
        Dictionary mapping tên baseline đến model object.
    """
    return {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent", random_state=random_state),
        "Baseline (Stratified)": DummyClassifier(strategy="stratified", random_state=random_state),
    }


def build_cv(random_state: int = 42) -> StratifiedKFold:
    """Tạo đối tượng Stratified 5-Fold Cross-Validation.

    Dùng Stratified K-Fold để đảm bảo mỗi fold giữ nguyên tỷ lệ
    phishing/legitimate giống tập gốc.

    Args:
        random_state: Seed cho thứ tự shuffle.

    Returns:
        Đối tượng ``StratifiedKFold`` với 5 folds.
    """
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def cross_validate_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """Đánh giá chéo nhiều mô hình với 5 metrics chuẩn.

    Chạy cross-validation cho mỗi mô hình với Accuracy, Precision,
    Recall, F1-Score và ROC-AUC. Kết quả được sắp xếp theo F1 giảm dần.

    Args:
        models: Dictionary mapping tên mô hình đến estimator.
        X_train: Ma trận đặc trưng tập huấn luyện.
        y_train: Nhãn nhị phân tập huấn luyện.
        cv: Đối tượng StratifiedKFold đã khởi tạo.

    Returns:
        DataFrame với mỗi hàng là một mô hình, các cột gồm mean và
        std của từng metric, sắp xếp theo ``f1_mean`` giảm dần.
    """
    rows: list[dict[str, float | str]] = []
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    for name, model in models.items():
        scores = cross_validate(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        row: dict[str, float | str] = {"model": name}
        for metric_name in scoring:
            row[f"{metric_name}_mean"] = float(np.mean(scores[f"test_{metric_name}"]))
            row[f"{metric_name}_std"] = float(np.std(scores[f"test_{metric_name}"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    quick: bool = False,
    random_state: int = 42,
) -> GridSearchCV:
    """Tinh chỉnh siêu tham số Random Forest bằng GridSearchCV.

    Tìm kiếm tổ hợp tối ưu cho ``n_estimators``, ``max_depth``,
    ``min_samples_split``, ``min_samples_leaf``, ``max_features`` và
    ``class_weight``. Scoring theo F1-Score.

    Args:
        X_train: Ma trận đặc trưng tập huấn luyện.
        y_train: Nhãn nhị phân tập huấn luyện.
        cv: Đối tượng StratifiedKFold cho cross-validation.
        quick: Nếu ``True``, dùng param grid nhỏ hơn để smoke-test.
        random_state: Seed cho reproducibility.

    Returns:
        Đối tượng ``GridSearchCV`` đã fit, truy cập ``best_estimator_``
        và ``best_params_`` để lấy kết quả.
    """
    estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }
    if quick:
        param_grid = {
            "n_estimators": [100],
            "max_depth": [None, 12],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
            "class_weight": [None, "balanced"],
        }

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid


def build_tuning_candidates(random_state: int = 42, quick: bool = False) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Build comparable hyperparameter search spaces for all candidate models."""
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    if quick:
        return {
            "Logistic Regression": (
                logistic,
                {
                    "model__C": [0.1, 1.0, 10.0],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "SVM": (
                svm,
                {
                    "model__C": [1.0, 2.0],
                    "model__gamma": ["scale", 0.1],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "Random Forest": (
                random_forest,
                {
                    "n_estimators": [100],
                    "max_depth": [None, 12],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt"],
                    "class_weight": [None, "balanced"],
                },
            ),
        }

    return {
        "Logistic Regression": (
            logistic,
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "SVM": (
            svm,
            {
                "model__C": [0.5, 1.0, 2.0, 5.0],
                "model__gamma": ["scale", 0.01, 0.1],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            random_forest,
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
        ),
    }


def tune_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    quick: bool = False,
    random_state: int = 42,
) -> tuple[dict[str, GridSearchCV], pd.DataFrame]:
    """Tune Logistic Regression, SVM and Random Forest with comparable CV search."""
    tuned: dict[str, GridSearchCV] = {}
    rows: list[dict[str, Any]] = []
    for name, (estimator, param_grid) in build_tuning_candidates(random_state=random_state, quick=quick).items():
        candidate_count = int(np.prod([len(values) for values in param_grid.values()]))
        print(f"Tuning {name}: {candidate_count} candidates x {cv.get_n_splits()} folds", flush=True)
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        tuned[name] = grid
        rows.append(
            {
                "model": name,
                "best_f1_cv": float(grid.best_score_),
                "best_params": grid.best_params_,
                "candidate_count": candidate_count,
            }
        )
    return tuned, pd.DataFrame(rows).sort_values("best_f1_cv", ascending=False).reset_index(drop=True)


def rank_features_by_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank features by Mutual Information against the binary target."""
    scores = mutual_info_classif(X_train, y_train, discrete_features=True, random_state=random_state)
    return (
        pd.DataFrame({"feature": list(X_train.columns), "mutual_information": scores})
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )


def select_top_features(feature_ranking: pd.DataFrame, k: int) -> list[str]:
    """Return the top-k feature names from a Mutual Information ranking."""
    return feature_ranking.head(k)["feature"].tolist()


def compare_feature_subsets(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    feature_ranking: pd.DataFrame,
    k_values: tuple[int, ...] = (10, 15, 20, 30),
) -> pd.DataFrame:
    """Compare top-k Mutual Information feature subsets with cross-validation."""
    rows: list[dict[str, Any]] = []
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    unique_k_values = sorted({min(max(1, int(k)), X_train.shape[1]) for k in k_values})
    for k in unique_k_values:
        selected_features = select_top_features(feature_ranking, k)
        scores = cross_validate(
            clone(model),
            X_train[selected_features],
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        row: dict[str, Any] = {
            "k": k,
            "features": ",".join(selected_features),
        }
        for metric_name in scoring:
            row[f"{metric_name}_mean"] = float(np.mean(scores[f"test_{metric_name}"]))
            row[f"{metric_name}_std"] = float(np.std(scores[f"test_{metric_name}"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def calibrate_classifier(
    model: Any,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrate probability estimates for a fitted classifier.

    The base estimator is expected to be already fitted. Calibration is learned
    only on the calibration split, which keeps the final test set untouched.
    """
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    else:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_calibration, y_calibration)
    return calibrated


def tune_decision_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    metric: str = "f1",
    min_recall: float | None = None,
) -> tuple[dict[str, float | str | None], pd.DataFrame]:
    """Find a decision threshold from calibrated probabilities.

    Args:
        y_true: Ground-truth binary labels.
        probabilities: Probability of the phishing class.
        metric: Metric used to rank thresholds. Supported: ``f1``,
            ``precision``, ``recall``.
        min_recall: Optional recall floor. Useful when reducing false
            negatives is more important than maximizing F1.

    Returns:
        A summary dictionary for the selected threshold and a full DataFrame
        with every candidate threshold.
    """
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                np.asarray(probabilities, dtype=float),
            ]
        )
    )
    rows: list[dict[str, float]] = []
    y_array = np.asarray(y_true, dtype=int)
    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_array, predictions, labels=[0, 1]).ravel()
        precision = precision_score(y_array, predictions, zero_division=0)
        recall = recall_score(y_array, predictions, zero_division=0)
        f1 = f1_score(y_array, predictions, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
                "tp": float(tp),
                "fp": float(fp),
                "tn": float(tn),
                "fn": float(fn),
            }
        )

    table = pd.DataFrame(rows)
    eligible = table
    if min_recall is not None:
        eligible = table[table["recall"] >= min_recall]
        if eligible.empty:
            eligible = table

    if metric not in {"f1", "precision", "recall"}:
        raise ValueError("metric must be one of: f1, precision, recall")
    selected = (
        eligible.sort_values([metric, "f1", "recall", "precision"], ascending=False)
        .iloc[0]
        .to_dict()
    )
    selected["metric"] = metric
    selected["min_recall"] = min_recall
    return selected, table


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Đánh giá mô hình trên tập kiểm tra và trả về toàn bộ metrics.

    Tính toán: Confusion Matrix, Classification Report, ROC-AUC,
    ROC Curve, Precision-Recall Curve và Average Precision.

    Args:
        model: Mô hình đã huấn luyện (cần có ``predict_proba``).
        X_test: Ma trận đặc trưng tập kiểm tra.
        y_test: Nhãn thực tế tập kiểm tra.
        threshold: Ngưỡng xác suất để gán nhãn ``phishing``.

    Returns:
        Dictionary chứa ``predictions``, ``probabilities``, ``confusion_matrix``,
        ``classification_report``, ``roc_auc``, ``roc_curve``, ``pr_curve``,
        và ``average_precision``.
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    report = classification_report(
        y_test,
        predictions,
        target_names=["legitimate", "phishing"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        target_names=["legitimate", "phishing"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
    ap_score = float(average_precision_score(y_test, probabilities))
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "threshold": float(threshold),
        "confusion_matrix": matrix,
        "classification_report": report,
        "classification_report_text": report_text,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "pr_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "average_precision": ap_score,
    }


def save_confusion_matrix_plot(matrix: np.ndarray, title: str, path: Path) -> None:
    """Vẽ và lưu biểu đồ Confusion Matrix dạng heatmap.

    Args:
        matrix: Ma trận nhầm lẫn từ ``sklearn.metrics.confusion_matrix``.
        title: Tiêu đề biểu đồ.
        path: Đường dẫn file PNG đầu ra.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Legitimate", "Pred Phishing"],
        yticklabels=["Actual Legitimate", "Actual Phishing"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Du doan")
    ax.set_ylabel("Thuc te")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_roc_curve_plot(results: dict[str, dict[str, Any]], path: Path) -> None:
    """Vẽ và lưu biểu đồ ROC-AUC so sánh nhiều mô hình.

    Args:
        results: Dictionary mapping tên model đến evaluation results,
                 mỗi result cần có key ``roc_curve`` và ``roc_auc``.
        path: Đường dẫn file PNG đầu ra.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, result in results.items():
        curve = result["roc_curve"]
        ax.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC={result['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC so sanh mo hinh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_precision_recall_plot(results: dict[str, dict[str, Any]], path: Path) -> None:
    """Vẽ và lưu biểu đồ Precision-Recall curve cho tất cả các mô hình.
    
    Precision-Recall curve đặc biệt hữu ích cho bài toán imbalanced,
    nơi mà ROC-AUC có thể cho kết quả quá lạc quan.
    
    Args:
        results: Dictionary mapping tên model đến evaluation results,
                 mỗi result cần có key 'probabilities' và 'pr_curve'.
        path: Đường dẫn file PNG đầu ra.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, result in results.items():
        if "pr_curve" not in result:
            continue
        curve = result["pr_curve"]
        ap = result.get("average_precision", 0)
        ax.plot(curve["recall"], curve["precision"], label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve so sanh mo hinh")
    ax.legend(loc="lower left")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_feature_importance_plot(
    model: RandomForestClassifier,
    path: Path,
    top_n: int = 15,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Vẽ biểu đồ Feature Importance và trả về DataFrame đầy đủ.

    Args:
        model: Random Forest đã huấn luyện (cần có ``feature_importances_``).
        path: Đường dẫn file PNG đầu ra.
        top_n: Số lượng đặc trưng quan trọng nhất hiển thị trên biểu đồ.
        feature_names: Danh sách tên đặc trưng tương ứng với model. Nếu
            ``None`` thì dùng schema UCI 30 đặc trưng.

    Returns:
        DataFrame gồm cột ``feature`` và ``importance``, sắp xếp giảm dần.
    """
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names or FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top_df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["importance"], color="#0B6E4F")
    ax.set_title("Top dac trung quan trong cua Random Forest")
    ax.set_xlabel("Feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return importance_df


def save_learning_curve_plot(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    path: Path,
    csv_path: Path,
    scoring: str = "f1",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute and save a learning curve for overfit/underfit diagnosis."""
    train_sizes, train_scores, validation_scores = learning_curve(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=random_state,
    )
    curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_score_mean": np.mean(train_scores, axis=1),
            "train_score_std": np.std(train_scores, axis=1),
            "validation_score_mean": np.mean(validation_scores, axis=1),
            "validation_score_std": np.std(validation_scores, axis=1),
        }
    )
    curve_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_df["train_size"], curve_df["train_score_mean"], marker="o", label="Train F1")
    ax.fill_between(
        curve_df["train_size"],
        curve_df["train_score_mean"] - curve_df["train_score_std"],
        curve_df["train_score_mean"] + curve_df["train_score_std"],
        alpha=0.15,
    )
    ax.plot(curve_df["train_size"], curve_df["validation_score_mean"], marker="o", label="Validation F1")
    ax.fill_between(
        curve_df["train_size"],
        curve_df["validation_score_mean"] - curve_df["validation_score_std"],
        curve_df["validation_score_mean"] + curve_df["validation_score_std"],
        alpha=0.15,
    )
    ax.set_title("Learning Curve - Best Random Forest")
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring.upper())
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return curve_df


def save_json(data: dict[str, Any], path: Path) -> None:
    """Ghi dictionary ra file JSON với định dạng đẹp và hỗ trợ Unicode.

    Args:
        data: Dữ liệu cần lưu.
        path: Đường dẫn file JSON đầu ra.
    """
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def precision_f1_explanation() -> str:
    """Trả về đoạn giải thích tầm quan trọng của Precision và F1 trong an ninh mạng.

    Returns:
        Chuỗi text giải thích tại sao F1-Score phù hợp hơn Accuracy
        cho bài toán phát hiện phishing.
    """
    return (
        "Trong bai toan chan website lua dao, Precision cao giup giam False Positive, "
        "tuc giam viec chan nham cac website hop le va tranh lam nguoi dung mat niem tin vao he thong. "
        "Tuy nhien, neu chi toi uu Precision ma Recall thap thi ta se bo sot nhieu trang phishing. "
        "F1-Score can bang giua Precision va Recall, vi vay phu hop hon Accuracy trong ngu canh an ninh mang, "
        "noi ma ca chan nham web tot (False Positive) lan bo lot web doc (False Negative) deu co chi phi rat lon."
    )
