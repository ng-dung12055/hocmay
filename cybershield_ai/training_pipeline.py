"""Pipeline huấn luyện end-to-end cho CyberShield AI.

Module này điều phối toàn bộ quy trình: tải dữ liệu UCI, chia
train/test, cross-validate nhiều mô hình (bao gồm baseline),
tinh chỉnh Random Forest, đánh giá trên tập test, sinh biểu đồ
(Confusion Matrix, ROC-AUC, Precision-Recall, SHAP), và lưu
model artifact dạng joblib.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, MODEL_DIR, REPORT_DIR
from .data_loader import compute_feature_defaults, ensure_project_dirs, load_uci_dataset
from .feature_extraction import configure_feature_defaults
from .modeling import (
    build_baseline_model,
    build_cv,
    calibrate_classifier,
    compare_feature_subsets,
    cross_validate_models,
    evaluate_model,
    precision_f1_explanation,
    rank_features_by_mutual_information,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_json,
    save_learning_curve_plot,
    save_precision_recall_plot,
    save_roc_curve_plot,
    select_top_features,
    tune_candidate_models,
    tune_decision_threshold,
)
from .xai import build_tree_explainer, get_positive_class_explanation

matplotlib.use("Agg")


def run_training_pipeline(
    quick: bool = False,
    sample_size: int | None = None,
    random_state: int = 42,
    output_model_path: Path | None = None,
    calibration_size: float = 0.15,
    calibration_method: str = "sigmoid",
    threshold_metric: str = "f1",
    threshold_min_recall: float | None = None,
    output_report_dir: Path | None = None,
    feature_selection_k_values: tuple[int, ...] = (10, 15, 20, 30),
    generate_learning_curves: bool = True,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline huấn luyện và đánh giá CyberShield AI.

    Quy trình:
        1. Tải bộ dữ liệu UCI Phishing Websites.
        2. Chia train/test (80/20, stratified).
        3. Cross-validate baseline + 3 candidate models.
        4. Tinh chỉnh Random Forest bằng GridSearchCV (scoring=F1).
        5. Đánh giá tất cả models trên tập test.
        6. Xuất biểu đồ và báo cáo JSON/CSV.
        7. Tính SHAP values và vẽ summary plot.
        8. Lưu model artifact (joblib).

    Args:
        quick: Nếu ``True``, dùng param grid nhỏ cho GridSearchCV.
        sample_size: Giới hạn số mẫu (cân bằng nhãn). ``None`` = dùng hết.
        random_state: Seed cho reproducibility.
        output_model_path: Đường dẫn tuỳ chỉnh cho file model.
            ``None`` = lưu tại ``artifacts/models/cybershield_ai_model.joblib``.
        calibration_size: Tỷ lệ của tập train dùng để calibrate xác suất và
            tune threshold. Tập test cuối cùng vẫn được giữ độc lập.
        calibration_method: Phương pháp calibration của scikit-learn
            (``sigmoid`` hoặc ``isotonic``).
        threshold_metric: Metric dùng để chọn ngưỡng quyết định
            (``f1``, ``precision`` hoặc ``recall``).
        threshold_min_recall: Ràng buộc recall tối thiểu khi chọn threshold.
        output_report_dir: Thư mục report tuỳ chỉnh. Hữu ích cho test để
            không ghi đè artifact chính.
        feature_selection_k_values: Các số lượng feature top-k cần so sánh
            bằng Mutual Information.
        generate_learning_curves: Nếu ``True``, xuất learning curve cho
            Random Forest tốt nhất.

    Returns:
        Dictionary chứa ``artifact_path``, ``cv_results`` (DataFrame),
        ``best_params``, ``evaluation_results``, và ``reports_dir``.
    """
    ensure_project_dirs()
    if output_model_path is not None:
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = output_report_dir or REPORT_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_uci_dataset(force_download=False)
    X = dataset.X.copy()
    y = dataset.y.copy()

    if sample_size is not None and sample_size < len(X):
        per_class = max(1, sample_size // 2)
        combined = pd.concat([X, y], axis=1)
        sampled_parts = [
            frame.sample(min(len(frame), per_class), random_state=random_state)
            for _, frame in combined.groupby("is_phishing")
        ]
        sampled = pd.concat(sampled_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        X = sampled[FEATURE_COLUMNS]
        y = sampled["is_phishing"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    X_model_train, X_calibration, y_model_train, y_calibration = train_test_split(
        X_train,
        y_train,
        test_size=calibration_size,
        stratify=y_train,
        random_state=random_state,
    )

    feature_defaults = compute_feature_defaults(X_train)
    configure_feature_defaults(feature_defaults)

    baseline_models = build_baseline_model(random_state=random_state)
    cv = build_cv(random_state=random_state)

    feature_ranking = rank_features_by_mutual_information(
        X_model_train,
        y_model_train,
        random_state=random_state,
    )
    feature_ranking.to_csv(reports_dir / "mutual_information_feature_ranking.csv", index=False)

    feature_selection_model = RandomForestClassifier(
        n_estimators=100 if quick else 200,
        random_state=random_state,
        n_jobs=-1,
    )
    feature_selection_results = compare_feature_subsets(
        feature_selection_model,
        X_model_train,
        y_model_train,
        cv=cv,
        feature_ranking=feature_ranking,
        k_values=feature_selection_k_values,
    )
    feature_selection_results.to_csv(reports_dir / "feature_selection_comparison.csv", index=False)
    selected_feature_count = int(feature_selection_results.iloc[0]["k"])
    selected_features = select_top_features(feature_ranking, selected_feature_count)

    X_model_train_selected = X_model_train[selected_features]
    X_calibration_selected = X_calibration[selected_features]
    X_test_selected = X_test[selected_features]
    X_train_selected = X_train[selected_features]

    tuned_models, tuning_summary = tune_candidate_models(
        X_model_train_selected,
        y_model_train,
        cv=cv,
        quick=quick,
        random_state=random_state,
    )
    tuning_summary.to_csv(reports_dir / "tuned_model_summary.csv", index=False)

    tuned_estimators = {name: grid.best_estimator_ for name, grid in tuned_models.items()}
    all_models_for_cv = {**baseline_models, **tuned_estimators}
    cv_results = cross_validate_models(all_models_for_cv, X_model_train_selected, y_model_train, cv)

    tuned_rf = tuned_models["Random Forest"]
    best_rf = tuned_rf.best_estimator_
    calibrated_rf = calibrate_classifier(
        best_rf,
        X_calibration_selected,
        y_calibration,
        method=calibration_method,
    )
    calibration_probabilities = calibrated_rf.predict_proba(X_calibration_selected)[:, 1]
    threshold_summary, threshold_table = tune_decision_threshold(
        y_calibration,
        calibration_probabilities,
        metric=threshold_metric,
        min_recall=threshold_min_recall,
    )
    decision_threshold = float(threshold_summary["threshold"])
    threshold_table.to_csv(reports_dir / "threshold_tuning.csv", index=False)

    fitted_models: dict[str, Any] = {}
    for name, model in tuned_estimators.items():
        if name == "Random Forest":
            fitted_models[name] = calibrated_rf
        else:
            fitted_models[name] = model
    for name, model in baseline_models.items():
        fitted_models[name] = model.fit(X_model_train_selected, y_model_train)

    evaluation_results = {
        name: evaluate_model(
            model,
            X_test_selected,
            y_test,
            threshold=decision_threshold if name == "Random Forest" else 0.5,
        )
        for name, model in fitted_models.items()
    }

    for name, result in evaluation_results.items():
        slug = name.lower().replace(" ", "_")
        save_confusion_matrix_plot(
            result["confusion_matrix"],
            title=f"Confusion Matrix - {name}",
            path=reports_dir / f"confusion_matrix_{slug}.png",
        )

    save_roc_curve_plot(evaluation_results, reports_dir / "roc_auc_comparison.png")
    save_precision_recall_plot(evaluation_results, reports_dir / "precision_recall_comparison.png")
    importance_df = save_feature_importance_plot(
        best_rf,
        reports_dir / "random_forest_feature_importance.png",
        feature_names=selected_features,
    )
    importance_df.to_csv(reports_dir / "random_forest_feature_importance.csv", index=False)

    learning_curve_df = None
    if generate_learning_curves:
        learning_curve_df = save_learning_curve_plot(
            best_rf,
            X_model_train_selected,
            y_model_train,
            cv=cv,
            path=reports_dir / "learning_curve_random_forest.png",
            csv_path=reports_dir / "learning_curve_random_forest.csv",
            random_state=random_state,
        )

    background = X_model_train_selected.sample(
        min(200, len(X_model_train_selected)), random_state=random_state
    ).reset_index(drop=True)
    shap_sample = X_test_selected.sample(min(250, len(X_test_selected)), random_state=random_state).reset_index(drop=True)
    explainer = build_tree_explainer(best_rf, background)
    shap_explanation = get_positive_class_explanation(explainer, shap_sample)
    shap.summary_plot(shap_explanation, shap_sample, show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    artifact = {
        "model": calibrated_rf,
        "raw_model": best_rf,
        "feature_columns": selected_features,
        "all_feature_columns": FEATURE_COLUMNS,
        "feature_defaults": feature_defaults,
        "label_mapping": {"legitimate": 0, "phishing": 1},
        "background_data": background,
        "best_params": tuned_rf.best_params_,
        "tuned_model_params": {name: grid.best_params_ for name, grid in tuned_models.items()},
        "selected_feature_count": selected_feature_count,
        "selected_features": selected_features,
        "feature_selection_method": "mutual_information_top_k",
        "calibration_method": calibration_method,
        "decision_threshold": decision_threshold,
        "threshold_summary": threshold_summary,
        "threshold_metric": threshold_metric,
        "threshold_min_recall": threshold_min_recall,
        "precision_f1_explanation": precision_f1_explanation(),
    }
    model_path = output_model_path or (MODEL_DIR / "cybershield_ai_model.joblib")
    joblib.dump(artifact, model_path)

    metrics_payload = {
        "dataset": {
            "sample_size": int(len(X)),
            "train_size": int(len(X_train)),
            "model_train_size": int(len(X_model_train)),
            "calibration_size": int(len(X_calibration)),
            "test_size": int(len(X_test)),
            "class_counts": {str(label): int(count) for label, count in y.value_counts().sort_index().items()},
        },
        "feature_selection": {
            "method": "mutual_information_top_k",
            "selected_feature_count": selected_feature_count,
            "selected_features": selected_features,
            "comparison": feature_selection_results.to_dict(orient="records"),
            "ranking": feature_ranking.to_dict(orient="records"),
        },
        "cv_results": cv_results.to_dict(orient="records"),
        "tuned_model_summary": tuning_summary.to_dict(orient="records"),
        "best_random_forest_params": tuned_rf.best_params_,
        "tuned_model_params": {name: grid.best_params_ for name, grid in tuned_models.items()},
        "calibration_method": calibration_method,
        "decision_threshold": decision_threshold,
        "threshold_summary": threshold_summary,
        "learning_curve": learning_curve_df.to_dict(orient="records") if learning_curve_df is not None else None,
        "evaluation_results": {
            name: {
                "roc_auc": result["roc_auc"],
                "average_precision": result.get("average_precision", None),
                "threshold": result.get("threshold", 0.5),
                "classification_report": result["classification_report"],
            }
            for name, result in evaluation_results.items()
        },
        "precision_f1_explanation": precision_f1_explanation(),
    }
    cv_results.to_csv(reports_dir / "cv_results.csv", index=False)
    save_json(metrics_payload, reports_dir / "metrics_summary.json")

    for name, result in evaluation_results.items():
        slug = name.lower().replace(" ", "_")
        (reports_dir / f"classification_report_{slug}.txt").write_text(
            result["classification_report_text"],
            encoding="utf-8",
        )

    return {
        "artifact_path": str(model_path),
        "cv_results": cv_results,
        "best_params": tuned_rf.best_params_,
        "selected_features": selected_features,
        "selected_feature_count": selected_feature_count,
        "feature_selection_results": feature_selection_results,
        "tuning_summary": tuning_summary,
        "decision_threshold": decision_threshold,
        "threshold_summary": threshold_summary,
        "evaluation_results": evaluation_results,
        "reports_dir": str(reports_dir),
    }
