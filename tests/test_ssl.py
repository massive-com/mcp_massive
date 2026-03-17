"""Verify that HTTPS connections work after os.environ.clear().

This reproduces the Windows SSL bug where clearing the environment (done
for security hardening in __init__.py) removes SSL_CERT_FILE and other
hints that Python's ssl module needs to locate the CA store on Windows.
"""

import os

import httpx


def test_ssl_handshake_after_env_clear():
    """An HTTPS handshake must succeed even after os.environ.clear().

    This is the exact scenario that caused [SSL] unknown error (_ssl.c:3134)
    on Windows.  We connect to a public HTTPS host and only check that the
    TLS handshake completes — the HTTP status code doesn't matter.
    """
    saved_env = os.environ.copy()
    try:
        os.environ.clear()

        # Without an explicit SSL context, this fails on Windows.
        resp = httpx.get(
            "https://httpbin.org/get",
            timeout=10.0,
        )
        # Any HTTP response means the SSL handshake succeeded.
        assert resp.status_code > 0
    finally:
        os.environ.update(saved_env)
