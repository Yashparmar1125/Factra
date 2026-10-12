from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _is_safe_url(url: str) -> bool:
    """Block private, loopback, and link-local IPs to prevent SSRF."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Resolve hostname to IP
        ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        # Block private, loopback, link-local, reserved
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
        return True
    except (ValueError, socket.gaierror):
        return False


def fetch_article_text(url: str, timeout_s: int = 12, max_chars: int = 8000) -> Optional[str]:
    """
    Scrape article text from <p> tags.
    Returns None if scraping fails or yields empty text.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    if not _is_safe_url(url):
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; +https://example.com/bot)",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch article from %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    ps = soup.find_all("p")
    parts = []
    for p in ps:
        t = p.get_text(" ", strip=True)
        if t:
            parts.append(t)

    text = "\n".join(parts).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text

