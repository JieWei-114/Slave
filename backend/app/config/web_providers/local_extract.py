from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from readability import Document

from app.config.settings import settings

logger = logging.getLogger(__name__)

MAX_CHARS = settings.LOCAL_EXTRACT_MAX_CHARS
MAX_BYTES = settings.LOCAL_EXTRACT_MAX_BYTES
REQUEST_TIMEOUT = settings.LOCAL_EXTRACT_TIMEOUT

USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/120.0 Safari/537.36'
)

ALLOWED_CONTENT_TYPES = ('text/html', 'application/xhtml+xml')

ALLOWED_SCHEMES = ('http', 'https')
MAX_REDIRECTS = 5


def _is_url_safe(url: str) -> bool:
    """
    SSRF guard: allow only http/https URLs whose hostname resolves exclusively
    to public IP addresses (no private/loopback/link-local/multicast/reserved).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        logger.warning('Local extract blocked (scheme not allowed): %s', url)
        return False

    host = parsed.hostname
    if not host:
        return False

    try:
        infos = socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        logger.warning('Local extract blocked (DNS resolution failed for %s): %s', host, e)
        return False

    if not infos:
        return False

    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            logger.warning('Local extract blocked (non-public IP %s for host %s)', ip_str, host)
            return False

    return True


async def _fetch_html(url: str) -> Optional[str]:
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Sec-Fetch-Site': 'none',
        # 'Referer': 'https://www.google.com/',
        # 'Accept-Encoding': 'identity',
    }

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    # Handle redirects manually so every hop is validated against SSRF
    async with httpx.AsyncClient(
        follow_redirects=False,
        headers=headers,
        timeout=timeout,
    ) as client:
        try:
            current_url = url
            resp = None
            for _ in range(MAX_REDIRECTS + 1):
                if not await asyncio.to_thread(_is_url_safe, current_url):
                    return None

                resp = await client.get(current_url)

                if resp.is_redirect:
                    location = resp.headers.get('location')
                    if not location:
                        return None
                    current_url = urljoin(current_url, location)
                    resp = None
                    continue
                break

            if resp is None:
                logger.warning('Local extract aborted (too many redirects): %s', url)
                return None

            resp.raise_for_status()

            if resp.content and len(resp.content) > MAX_BYTES:
                logger.warning(
                    'Local extract skipped (response too large): %s bytes', len(resp.content)
                )
                return None

            ctype = resp.headers.get('content-type', '').lower()
            if not any(t in ctype for t in ALLOWED_CONTENT_TYPES):
                return None

            if not resp.text.strip():
                logger.debug('Local extract: empty body returned')
                return None

            return resp.text
        except Exception as e:
            logger.warning('Local extract fetch error: %s', e)
            return None


def _extract_main_text(html: str) -> str:
    doc = Document(html)
    clean_html = doc.summary(html_partial=True)

    soup = BeautifulSoup(clean_html, 'lxml')
    text = soup.get_text('\n')

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines).strip()


async def extract_url_local(url: str) -> str:
    # Download and extract main text from a URL with safety limits.
    logger.info('Local extract called for %s', url)

    try:
        html = await _fetch_html(url)
        if not html:
            return ''

        text = await asyncio.to_thread(_extract_main_text, html)
    except Exception:
        return ''

    if not text:
        return ''

    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS].rstrip() + '…'

    return text
