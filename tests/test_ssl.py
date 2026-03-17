"""Verify that SSL and DNS work after the security env stripping in __init__.py.

The server strips most environment variables at startup to prevent secret
exfiltration.  On Windows this previously broke DNS (SYSTEMROOT) and SSL
(certificate store lookup).  These tests verify the fix.
"""

import os
import ssl
import socket

import certifi

# Must match the keep-list in __init__.py.
_KEEP = {
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "PROGRAMDATA", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
    "PATH", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
}


def _simulate_env_strip():
    """Apply the same env stripping logic as __init__.py.

    Snapshots the keep-list values from the current env, clears everything,
    then restores only the kept values.
    """
    kept = {k: v for k, v in os.environ.items() if k in _KEEP}
    os.environ.clear()
    os.environ.update(kept)


def test_ssl_context_loads_after_env_strip():
    """An SSL context built from certifi must load CA certs after env stripping."""
    saved_env = os.environ.copy()
    try:
        _simulate_env_strip()

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        stats = ssl_ctx.cert_store_stats()
        assert stats["x509_ca"] > 0, "SSL context has no CA certificates after env strip"
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def test_dns_resolution_after_env_strip():
    """DNS resolution must work after env stripping (requires SYSTEMROOT on Windows)."""
    saved_env = os.environ.copy()
    try:
        _simulate_env_strip()

        # If DNS is broken (missing SYSTEMROOT on Windows), this raises
        # socket.gaierror: [Errno 11001] getaddrinfo failed
        result = socket.getaddrinfo("localhost", 443)
        assert len(result) > 0
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def test_certifi_ca_bundle_is_valid():
    """certifi.where() must point to a readable CA bundle."""
    ca_path = certifi.where()
    assert os.path.isfile(ca_path)

    ctx = ssl.create_default_context(cafile=ca_path)
    stats = ctx.cert_store_stats()
    assert stats["x509_ca"] > 0
