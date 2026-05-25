"""Giao diện Streamlit cho CyberShield AI.

Ứng dụng web cho phép người dùng nhập URL, quét đặc trưng live,
dự đoán phishing bằng mô hình Random Forest đã huấn luyện, và
hiển thị giải thích SHAP waterfall kèm đánh giá loại rủi ro.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from cybershield_ai.config import FEATURE_FRIENDLY_NAMES, MODEL_DIR
from cybershield_ai.feature_extraction import configure_feature_defaults, scan_url
from cybershield_ai.xai import build_tree_explainer, get_positive_class_explanation

MODEL_PATH = MODEL_DIR / "cybershield_ai_model.joblib"

_HOSTNAME_LABEL_RE = re.compile(r"^(?=.{1,63}$)[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
_ALLOWED_SCHEMES = {"http", "https"}


def validate_url_input(text: str) -> str | None:
    """Kiểm tra URL người dùng nhập.

    Trả về thông báo lỗi tiếng Việt nếu URL không hợp lệ,
    hoặc ``None`` nếu URL có thể đem đi quét.
    """
    if not text or not text.strip():
        return "Hãy nhập một URL để quét."
    candidate = text.strip()
    if any(ch.isspace() for ch in candidate):
        return "URL không được chứa khoảng trắng."

    if "://" in candidate:
        scheme = candidate.split("://", 1)[0].lower()
        if scheme not in _ALLOWED_SCHEMES:
            return f"Chỉ hỗ trợ giao thức http hoặc https (đã nhận: '{scheme}')."
        normalized = candidate
    else:
        normalized = "http://" + candidate

    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return "Không phân tích được URL. Hãy kiểm tra lại định dạng."

    host = (parsed.hostname or "").strip()
    if not host:
        return "Không xác định được hostname trong URL."

    try:
        ipaddress.ip_address(host)
        return None
    except ValueError:
        pass

    if "." not in host:
        return (
            f"'{host}' không phải URL hợp lệ: thiếu tên miền cấp cao (TLD) như .com, .vn, .net..."
        )

    labels = host.split(".")
    for label in labels:
        if not _HOSTNAME_LABEL_RE.match(label):
            return f"Hostname '{host}' chứa nhãn không hợp lệ: '{label}'."

    tld = labels[-1]
    if len(tld) < 2 or not tld.isalpha():
        return f"Phần đuôi tên miền '.{tld}' không hợp lệ."

    return None


@st.cache_resource(show_spinner=False)
def load_artifact(model_path: str) -> dict:
    """Nạp model artifact từ file joblib và cấu hình feature defaults.

    Kết quả được cache bởi Streamlit để không nạp lại mỗi lần rerun.

    Args:
        model_path: Đường dẫn tuyệt đối tới file ``.joblib``.

    Returns:
        Dictionary chứa model, feature_columns, feature_defaults,
        background_data và các metadata khác.
    """
    artifact = joblib.load(model_path)
    configure_feature_defaults(artifact["feature_defaults"])
    return artifact


@st.cache_resource(show_spinner=False)
def build_explainer(model_path: str):
    """Tạo SHAP TreeExplainer từ model artifact (cached).

    Args:
        model_path: Đường dẫn tới file model ``.joblib``.

    Returns:
        Đối tượng ``shap.TreeExplainer`` sẵn sàng giải thích.
    """
    artifact = load_artifact(model_path)
    background = artifact.get("background_data")
    return build_tree_explainer(artifact.get("raw_model", artifact["model"]), background)


def render_shap_waterfall(explanation: shap.Explanation) -> None:
    """Vẽ biểu đồ SHAP waterfall cho mẫu đầu tiên và hiển thị trên Streamlit.

    Args:
        explanation: Đối tượng ``shap.Explanation`` chứa SHAP values.
    """
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)


def summarize_shap(explanation: shap.Explanation, feature_row: pd.DataFrame) -> str:
    """Tóm tắt top 3 đặc trưng đẩy rủi ro phishing lên cao nhất.

    Lấy 3 đặc trưng có SHAP value dương lớn nhất, chuyển thành tên
    thân thiện tiếng Việt từ ``FEATURE_FRIENDLY_NAMES``.

    Args:
        explanation: Đối tượng ``shap.Explanation`` cho mẫu cần tóm tắt.
        feature_row: DataFrame 1 dòng chứa giá trị đặc trưng.

    Returns:
        Câu mô tả bằng tiếng Việt, ví dụ:
        *"Rủi ro tăng do URL quá dài, tên miền có dấu gạch ngang, ..."*
    """
    values = pd.Series(explanation.values[0], index=feature_row.columns)
    positive = values.sort_values(ascending=False)
    top_positive = [feature for feature, value in positive.items() if value > 0][:3]
    if not top_positive:
        return "Rủi ro hiện tại chưa có đặc trưng nào nổi trội theo SHAP."
    phrases = [FEATURE_FRIENDLY_NAMES.get(feature, feature) for feature in top_positive]
    return "Rủi ro tăng do " + ", ".join(phrases) + "."


def main() -> None:
    """Điểm vào chính của ứng dụng Streamlit CyberShield AI.

    Khởi tạo giao diện, nạp model, nhận URL từ người dùng, quét
    đặc trưng, dự đoán, và hiển thị kết quả kèm giải thích SHAP.
    """
    st.set_page_config(page_title="CyberShield AI", page_icon="🔒", layout="wide")
    st.markdown(
        """
        <style>
        .stTextInput input {
            font-size: 1.2rem;
            padding: 0.85rem 1rem;
        }
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("CyberShield AI")

    if not MODEL_PATH.exists():
        st.error("Chưa tìm thấy model artifact. Hãy chạy `python train.py` trước.")
        st.stop()

    artifact = load_artifact(str(MODEL_PATH))
    explainer = build_explainer(str(MODEL_PATH))

    url_input = st.text_input(
        "🔒 Nhập đường link (URL) cần kiểm tra vào đây...",
        placeholder="https://example.com/login",
    )
    run_scan = st.button("Quét Radar", type="primary", use_container_width=True)

    if not run_scan:
        return

    validation_error = validate_url_input(url_input)
    if validation_error:
        st.warning(validation_error)
        st.caption("Ví dụ URL hợp lệ: `https://example.com`")
        return

    with st.spinner("Đang quét URL, bóc tách đặc trưng và giải thích bằng SHAP..."):
        scan_result = scan_url(url_input, feature_defaults=artifact["feature_defaults"])
        feature_row = pd.DataFrame([scan_result["features"]], columns=artifact["feature_columns"])
        probability = float(artifact["model"].predict_proba(feature_row)[0, 1])
        threshold = float(artifact.get("decision_threshold", 0.5))
        prediction = int(probability >= threshold)
        explanation = get_positive_class_explanation(explainer, feature_row)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        if prediction == 1:
            st.error("🚨 NGUY HIỂM: Phát hiện dấu hiệu Phishing!")
        else:
            st.success("✅ Web An toàn")

        st.metric("Điểm rủi ro (xác suất phishing)", f"{probability:.2%}")
        st.caption(f"Ngưỡng cảnh báo hiện dùng: {threshold:.2%}")
        if prediction == 1:
            st.info(f"Loại rủi ro nghi ngờ: **{scan_result['risk_category_label']}**")
            st.caption(scan_result["risk_summary"])
            if scan_result["risk_signals"]:
                for signal in scan_result["risk_signals"]:
                    st.caption(f"- {signal}")
        else:
            st.info(f"Tín hiệu quan sát được: **{scan_result['risk_category_label']}**")
            st.caption(
                "Các tín hiệu này chỉ dùng để mô tả ngữ cảnh. "
                "Kết luận cuối cùng vẫn là Web An toàn vì điểm rủi ro thấp hơn ngưỡng cảnh báo."
            )
            if scan_result["risk_signals"]:
                with st.expander("Xem tín hiệu mô tả đã ghi nhận"):
                    st.caption(scan_result["risk_summary"])
                    for signal in scan_result["risk_signals"]:
                        st.caption(f"- {signal}")
        st.write(summarize_shap(explanation, feature_row))

        if scan_result["warnings"]:
            st.warning("Một số tín hiệu live không truy vấn được. Hệ thống đã dùng fallback để vẫn trả kết quả.")
            for warning in scan_result["warnings"][:6]:
                st.caption(f"- {warning}")

        with st.expander("Xem bộ đặc trưng đã bóc tách"):
            st.dataframe(feature_row.T.rename(columns={0: "value"}), use_container_width=True)

    with col_right:
        st.subheader("Giải thích SHAP cho URL này")
        render_shap_waterfall(explanation)

    with st.expander("Bằng chứng kỹ thuật từ quá trình quét"):
        if scan_result["evidence"]:
            for key, value in scan_result["evidence"].items():
                st.write(f"- **{key}**: {value}")
        else:
            st.write("Không có bằng chứng chi tiết nào được ghi nhận.")


if __name__ == "__main__":
    main()
