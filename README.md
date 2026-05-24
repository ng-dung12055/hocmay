# CyberShield AI

CyberShield AI là đồ án học máy cho bài toán **Phishing Website Detection**. Hệ thống huấn luyện mô hình trên UCI Phishing Websites Dataset, bóc tách 30 đặc trưng theo schema UCI từ URL thực tế, dự đoán xác suất phishing, dùng threshold đã tune để ra cảnh báo và giải thích bằng SHAP.

## Điểm nổi bật

- Dùng **UCI Phishing Websites Dataset** với 11,055 mẫu và 30 đặc trưng.
- So sánh **baseline**, **Logistic Regression**, **SVM** và **Random Forest** bằng Stratified 5-Fold CV.
- Tinh chỉnh công bằng cả **Logistic Regression**, **SVM** và **Random Forest** bằng GridSearchCV.
- Chạy **Feature Selection** bằng Mutual Information, so sánh Top-K feature sets trước khi chọn feature set cuối.
- Xuất **Learning Curve** để phân tích overfitting/underfitting.
- Calibrate xác suất bằng calibration split và tune decision threshold thay vì cố định 0.5.
- Xuất artifact đồng bộ: model, raw model, feature defaults, threshold, metrics, plots và SHAP background.
- Có live benchmark end-to-end từ URL thật qua `scan_url -> model -> threshold`.
- Có Streamlit demo để nhập URL và xem xác suất, nhãn, risk tag, SHAP và bằng chứng kỹ thuật.

## Cấu trúc chính

- `cybershield_ai/data_loader.py`: tải và chuẩn hóa dữ liệu UCI.
- `cybershield_ai/feature_extraction.py`: public API `scan_url()` và `extract_features_from_url()`.
- `cybershield_ai/url_features.py`: đặc trưng hình thái URL và domain.
- `cybershield_ai/html_features.py`: đặc trưng HTML/DOM.
- `cybershield_ai/network_clients.py`: HTTP, DNS và hydrate context live.
- `cybershield_ai/whois_ssl.py`: WHOIS, SSL, tuổi domain.
- `cybershield_ai/reputation.py`: Tranco, search proxy và phishing feeds.
- `cybershield_ai/modeling.py`: mô hình, metrics, calibration, threshold tuning và biểu đồ.
- `cybershield_ai/training_pipeline.py`: pipeline train/evaluate/export artifact.
- `cybershield_ai/live_benchmark.py`: benchmark URL live riêng.
- `app.py`: giao diện Streamlit.
- `tests/`: unit test và integration test.

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

## Huấn luyện

Chạy trên toàn bộ dataset với grid nhanh:

```bash
python train.py --quick
```

Chạy đầy đủ grid lớn:

```bash
python train.py
```

Tùy chỉnh calibration và threshold:

```bash
python train.py --quick --threshold-metric f1
python train.py --quick --threshold-metric recall --threshold-min-recall 0.97
```

Tùy chỉnh feature selection và bỏ qua learning curve khi cần chạy nhanh:

```bash
python train.py --quick --feature-selection-k-values 10,15,20,30
python train.py --quick --no-learning-curves
```

Artifact model được lưu tại:

```text
artifacts/models/cybershield_ai_model.joblib
```

Report được lưu tại:

```text
artifacts/reports/
```

Các report quan trọng:

- `cv_results.csv`: so sánh baseline và các mô hình đã tune.
- `tuned_model_summary.csv`: best params của Logistic Regression, SVM, Random Forest.
- `mutual_information_feature_ranking.csv`: xếp hạng feature theo Mutual Information.
- `feature_selection_comparison.csv`: so sánh Top-K feature sets.
- `learning_curve_random_forest.png`: learning curve của Random Forest tốt nhất.
- `threshold_tuning.csv`: bảng chọn ngưỡng cảnh báo.

## Live Benchmark

Chạy benchmark 250 phishing URL và 250 legitimate URL:

```bash
python tools/run_live_benchmark.py --phishing-count 250 --legitimate-count 250
```

Chạy smoke benchmark nhanh:

```bash
python tools/run_live_benchmark.py --phishing-count 10 --legitimate-count 10 --max-urls 20
```

Kết quả benchmark được lưu tại:

```text
artifacts/reports/live_benchmark/
```

## Chạy Ứng Dụng

```bash
streamlit run app.py
```

## Kiểm Thử

Chạy unit test nhanh:

```bash
pytest -q -m "not slow"
```

Chạy toàn bộ test, gồm integration test train nhanh:

```bash
pytest -q
```

## Quy Ước Nhãn

- Nhãn gốc UCI: `Result = -1` là phishing, `Result = 1` là legitimate.
- Trong project:
  - `1 = phishing`
  - `0 = legitimate`


