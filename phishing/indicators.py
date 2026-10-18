"""
Phishing indicator extraction for Beast.

Hand-engineered signals that phishing emails and URLs share regardless
of wording: suspicious URLs (raw IPs, shorteners, punycode, deep
subdomains, odd TLDs), urgency / credential / money language, HTML
forms, and shouting. These are complementary to Attack's character
n-grams, which see surface text but not structure.
"""

import re
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np

FEATURE_NAMES = [
    "url_count",
    "ip_url_ratio",
    "shortener_url_ratio",
    "punycode_url_ratio",
    "suspicious_tld_ratio",
    "at_in_url_ratio",
    "https_ratio",
    "max_url_length",
    "max_subdomain_depth",
    "max_domain_hyphens",
    "url_digit_ratio",
    "urgency_density",
    "credential_density",
    "money_density",
    "exclamation_density",
    "uppercase_ratio",
    "has_html_form",
    "log_length",
]

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_BARE_DOMAIN_RE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}(?:[/:?#]\S*)?$", re.IGNORECASE)
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_WORD_RE = re.compile(r"[A-Za-z']+")
_FORM_RE = re.compile(r"<\s*(?:form|input)\b", re.IGNORECASE)

_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy",
}
_SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "click", "country", "kim", "work", "gq",
    "tk", "ml", "cf", "ga", "rest", "support", "cam", "icu",
}
_URGENCY_TERMS = (
    "urgent", "immediately", "suspended", "verify", "within 24 hours",
    "action required", "final notice", "expire", "unusual activity",
    "locked", "limited time", "act now", "confirm",
)
_CREDENTIAL_TERMS = (
    "password", "login", "log in", "sign in", "username", "account",
    "credentials", "ssn", "social security", "pin", "security question",
)
_MONEY_TERMS = (
    "bank", "invoice", "payment", "refund", "wire", "transfer", "bitcoin",
    "gift card", "prize", "won", "lottery", "credit card", "paypal",
)


def _term_density(text_lower: str, terms, word_count: int) -> float:
    if word_count == 0:
        return 0.0
    return sum(text_lower.count(t) for t in terms) / word_count


class IndicatorExtractor:
    """Computes structural phishing indicators from raw email/URL text."""

    def extract_urls(self, text: str) -> List[str]:
        urls = _URL_RE.findall(text)
        # A bare URL/domain sample (PhishTank-style) has no scheme prefix.
        stripped = text.strip()
        if not urls and " " not in stripped and _BARE_DOMAIN_RE.match(stripped):
            urls = [stripped]
        return urls

    @staticmethod
    def _host(url: str) -> str:
        if not re.match(r"^[a-z]+://", url, re.IGNORECASE):
            url = "http://" + url
        try:
            host = urlparse(url).hostname or ""
        except ValueError:
            return ""
        return host.lower()

    def extract(self, text: str) -> Dict[str, float]:
        """Compute the full feature dict for a single sample."""
        features = {name: 0.0 for name in FEATURE_NAMES}
        text_lower = text.lower()
        words = _WORD_RE.findall(text)
        word_count = len(words)

        urls = self.extract_urls(text)
        if urls:
            n = len(urls)
            hosts = [self._host(u) for u in urls]
            features["url_count"] = float(n)
            features["ip_url_ratio"] = sum(1 for h in hosts if _IP_RE.match(h)) / n
            features["shortener_url_ratio"] = sum(1 for h in hosts if h in _SHORTENERS) / n
            features["punycode_url_ratio"] = sum(1 for h in hosts if "xn--" in h) / n
            features["suspicious_tld_ratio"] = sum(
                1 for h in hosts if h.rsplit(".", 1)[-1] in _SUSPICIOUS_TLDS
            ) / n
            features["at_in_url_ratio"] = sum(1 for u in urls if "@" in u) / n
            features["https_ratio"] = sum(1 for u in urls if u.lower().startswith("https://")) / n
            features["max_url_length"] = float(max(len(u) for u in urls))
            features["max_subdomain_depth"] = float(max(h.count(".") for h in hosts))
            features["max_domain_hyphens"] = float(max(h.count("-") for h in hosts))
            url_chars = "".join(urls)
            features["url_digit_ratio"] = sum(c.isdigit() for c in url_chars) / len(url_chars)

        features["urgency_density"] = _term_density(text_lower, _URGENCY_TERMS, word_count)
        features["credential_density"] = _term_density(text_lower, _CREDENTIAL_TERMS, word_count)
        features["money_density"] = _term_density(text_lower, _MONEY_TERMS, word_count)
        features["exclamation_density"] = text.count("!") / word_count if word_count else 0.0

        letters = [c for c in text if c.isalpha()]
        features["uppercase_ratio"] = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
        features["has_html_form"] = 1.0 if _FORM_RE.search(text) else 0.0
        features["log_length"] = float(np.log1p(len(text)))

        return features

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Vectorize a batch of texts into a (N, num_features) array, ordered by FEATURE_NAMES."""
        rows = [self.extract(t) for t in texts]
        return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
