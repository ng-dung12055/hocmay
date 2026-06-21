from __future__ import annotations

import socket
import ssl
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from .config import REQUEST_TIMEOUT, TRUSTED_ISSUER_HINTS
from .scan_context import ScanContext

try:
    import whois
except ImportError:  # pragma: no cover
    whois = None

_WHOIS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whois-lookup")


def _feature_ssl_final_state(context: ScanContext) -> int | None:
    if context.parsed.scheme != "https":
        context.evidence["SSLfinal_State"] = "URL không dùng HTTPS"
        return -1
    if context.ssl_info is None:
        return None
    if context.ssl_info.get("valid") and context.ssl_info.get("issuer_trusted"):
        days_remaining = int(context.ssl_info.get("days_remaining", 0))
        if days_remaining < 30:
            context.evidence["SSLfinal_State"] = (
                f"HTTPS hợp lệ, issuer đáng tin; chứng chỉ còn {days_remaining} ngày."
            )
        else:
            context.evidence["SSLfinal_State"] = "HTTPS hợp lệ với issuer đáng tin."
        return 1
    if context.ssl_info.get("valid"):
        context.evidence["SSLfinal_State"] = "HTTPS có mặt nhưng tín hiệu SSL chưa đủ mạnh"
        return 0
    context.evidence["SSLfinal_State"] = context.ssl_info.get("error", "Không xác minh được chứng chỉ")
    return 0


def _feature_domain_registration_length(context: ScanContext) -> int | None:
    expiration = _extract_whois_expiration(context.whois_record)
    if expiration is None:
        return None
    remaining_days = (expiration - _now_utc()).days
    context.evidence["Domain_registeration_length"] = f"Số ngày còn hạn đăng ký = {remaining_days}"
    return -1 if remaining_days <= 365 else 1


def _feature_abnormal_url(context: ScanContext) -> int | None:
    if context.whois_record is None:
        return 1 if context.hostname and context.hostname in context.normalized_url else None
    domain_names = _extract_whois_domain_names(context.whois_record)
    if not domain_names:
        return 1 if context.hostname and context.hostname in context.normalized_url else None
    if any(context.registered_domain in domain_name for domain_name in domain_names):
        return 1
    context.evidence["Abnormal_URL"] = "WHOIS domain không khớp với URL"
    return -1


def _feature_age_of_domain(context: ScanContext) -> int | None:
    creation = _extract_whois_creation(context.whois_record)
    if creation is None:
        return None
    age_days = (_now_utc() - creation).days
    context.evidence["age_of_domain"] = f"Số ngày tồn tại tên miền = {age_days}"
    return 1 if age_days >= 180 else -1


def _feature_dns_record(context: ScanContext) -> int | None:
    if context.dns_resolved is None:
        return None
    return 1 if context.dns_resolved else -1


@lru_cache(maxsize=256)
def _lookup_whois(domain: str) -> Any | None:
    if whois is None or not domain:
        return None
    future = _WHOIS_EXECUTOR.submit(whois.whois, domain)
    try:
        return future.result(timeout=REQUEST_TIMEOUT)
    except FuturesTimeoutError:
        future.cancel()
        return None
    except Exception:
        return None


@lru_cache(maxsize=256)
def _lookup_ssl_info(hostname: str, port: int) -> dict[str, Any] | None:
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=REQUEST_TIMEOUT) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as secured:
                cert = secured.getpeercert()
        issuer_parts = []
        for item in cert.get("issuer", ()):
            for key, value in item:
                issuer_parts.append(f"{key}={value}")
        issuer_text = " ".join(issuer_parts).lower()
        not_after = cert.get("notAfter")
        if not_after:
            expires_at = datetime.utcfromtimestamp(ssl.cert_time_to_seconds(not_after)).replace(tzinfo=timezone.utc)
            days_remaining = (expires_at - _now_utc()).days
        else:
            days_remaining = 0
        return {
            "valid": True,
            "issuer": issuer_text,
            "issuer_trusted": any(hint in issuer_text for hint in TRUSTED_ISSUER_HINTS),
            "days_remaining": days_remaining,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc), "days_remaining": 0, "issuer_trusted": False}


def _extract_whois_domain_names(record: Any) -> list[str]:
    if record is None:
        return []
    raw = None
    if isinstance(record, dict):
        raw = record.get("domain_name")
    else:
        raw = getattr(record, "domain_name", None)
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return [str(value).strip().lower() for value in values if value]


def _extract_whois_creation(record: Any) -> datetime | None:
    raw = _extract_record_field(record, "creation_date")
    return _normalize_datetime(raw, pick="min")


def _extract_whois_expiration(record: Any) -> datetime | None:
    raw = _extract_record_field(record, "expiration_date")
    return _normalize_datetime(raw, pick="max")


def _extract_record_field(record: Any, field_name: str) -> Any:
    if record is None:
        return None
    if isinstance(record, dict):
        return record.get(field_name)
    return getattr(record, field_name, None)


def _normalize_datetime(value: Any, pick: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        normalized = [item for item in (_normalize_datetime(item, pick="min") for item in value) if item is not None]
        if not normalized:
            return None
        return min(normalized) if pick == "min" else max(normalized)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
