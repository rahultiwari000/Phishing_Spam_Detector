# phishing_model/url_features.py

import re
from urllib.parse import urlparse

# Some lists for pattern-based detection
SUSPICIOUS_TLDS = {"zip", "mov", "top", "gq", "ml", "cf", "tk", "work", "click", "country", "stream", "men", "party"}
PHISH_KEYWORDS = {"login", "signin", "verify", "update", "secure", "account", "bank", "confirm", "webscr", "paypal"}
SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "buff.ly", "is.gd", "cutt.ly", "rebrand.ly"}


def count_chars(s, chars):
    """Count how many times any character in 'chars' appears in 's'."""
    return sum(s.count(c) for c in chars)


def has_ip(host):
    """Check if a hostname is an IP address."""
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))


def extract_features(url: str) -> dict:
    """
    Extract about 20 useful features from a given URL.
    Returns a dictionary of feature_name → value.
    """
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        parsed = urlparse("http://" + url)

    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full = (url or "")

    # Basic structure
    url_len = len(full)
    host_len = len(netloc)
    path_len = len(path)
    num_dots = netloc.count(".")
    num_hyphens_host = netloc.count("-")
    num_hyphens_path = path.count("-")
    num_digits = sum(ch.isdigit() for ch in full)
    num_params = query.count("&") + (1 if query else 0)

    # Symbol-related
    special_chars = count_chars(full, ['@', '?', '%', '=', '&', '#', '.', '-', '_', '/'])
    has_at = int("@" in full)
    has_double_slash_in_path = int("//" in path.strip("/"))

    # Host info
    host_parts = netloc.split(".")
    tld = host_parts[-1] if len(host_parts) >= 2 else ""
    sld = host_parts[-2] if len(host_parts) >= 2 else netloc

    contains_ip = int(has_ip(netloc))
    is_shortener = int(netloc in SHORTENERS)
    suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    # Connection and port
    is_https = int(scheme == "https")
    has_non_default_port = int(":" in netloc and not netloc.endswith(":80") and not netloc.endswith(":443"))

    # Keyword patterns
    lower_full = full.lower()
    keyword_hits = sum(1 for k in PHISH_KEYWORDS if k in lower_full)

    # Ratios
    digit_ratio = num_digits / max(1, url_len)
    path_depth = len([p for p in path.split("/") if p])

    # Suspicious signs
    has_https_in_host = int("https" in netloc and scheme != "https")
    starts_with_ip = contains_ip

    return {
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,
        "num_dots": num_dots,
        "num_hyphens_host": num_hyphens_host,
        "num_hyphens_path": num_hyphens_path,
        "num_digits": num_digits,
        "num_params": num_params,
        "special_chars": special_chars,
        "has_at": has_at,
        "has_double_slash_in_path": has_double_slash_in_path,
        "contains_ip": contains_ip,
        "is_shortener": is_shortener,
        "suspicious_tld": suspicious_tld,
        "is_https": is_https,
        "has_non_default_port": has_non_default_port,
        "keyword_hits": keyword_hits,
        "digit_ratio": digit_ratio,
        "path_depth": path_depth,
        "has_https_in_host": has_https_in_host,
        "starts_with_ip": starts_with_ip,
        "sld_len": len(sld),
    }
