"""CGNAT + IPv6 SSRF regression tests for ``_reject_internal_url``.

The block-list this replaced was RFC1918-only: it missed CGNAT
``100.64.0.0/10`` (this estate's Tailscale tailnet — postgres 100.64.0.5,
registry, k3s master), ``0.0.0.0/8``, ``224.0.0.0/4``, ``240.0.0.0/4``, the
broadcast address, and had **no IPv6 entries at all**. A scoped IPv6 literal
also reached a bare ``ipaddress.ip_address()`` and produced an HTTP 500.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest
from fastapi import HTTPException

from src.api.routes.dependencies import _reject_internal_url, _validate_external_url


def _patch_dns(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, str]) -> None:
    """Point ``w2v_common``'s resolver at a fixed hostname -> IP mapping."""

    def fake_getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        ip = mapping.get(host)
        if ip is None:
            raise socket.gaierror(f"unmapped host {host}")
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr("w2v_common.net.safe_fetch.socket.getaddrinfo", fake_getaddrinfo)


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://100.64.0.1/x",
        "http://100.64.0.5/x",
        "http://[::ffff:100.64.0.5]/x",
    ],
)
def test_cgnat_literals_rejected_400(bad_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dns(
        monkeypatch,
        {
            "100.64.0.1": "100.64.0.1",
            "100.64.0.5": "100.64.0.5",
            "::ffff:100.64.0.5": "::ffff:100.64.0.5",
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        _reject_internal_url(bad_url, "image_url")
    assert exc_info.value.status_code == 400


def test_hostname_resolving_to_cgnat_rejected_400(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dns(monkeypatch, {"asset-host.example.com": "100.64.0.5"})
    with pytest.raises(HTTPException) as exc_info:
        _reject_internal_url("https://asset-host.example.com/img.png", "image_url")
    assert exc_info.value.status_code == 400
    assert "private/internal" in str(exc_info.value.detail)


def test_ipv6_loopback_and_ula_rejected_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old list had zero IPv6 entries, so ``::1`` / ``fd00::`` sailed through."""
    _patch_dns(monkeypatch, {"v6-loop.example.com": "::1", "v6-ula.example.com": "fd00::1"})
    for host in ("v6-loop.example.com", "v6-ula.example.com"):
        with pytest.raises(HTTPException) as exc_info:
            _reject_internal_url(f"http://{host}/x", "callback_url")
        assert exc_info.value.status_code == 400


def test_scoped_ipv6_is_4xx_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fe80::1%eth0`` used to raise an uncaught ValueError -> HTTP 500."""
    _patch_dns(monkeypatch, {"scoped.example.com": "fe80::1%eth0"})
    with pytest.raises(HTTPException) as exc_info:
        _reject_internal_url("http://scoped.example.com/x", "callback_url")
    assert 400 <= exc_info.value.status_code < 500


def test_public_host_still_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dns(monkeypatch, {"example.com": "93.184.216.34"})
    _reject_internal_url("https://example.com/v.mp4", "image_url")


@pytest.mark.parametrize("ip", ["100.63.255.255", "100.128.0.0"])
def test_cgnat_boundary_addresses_still_accepted(ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deny is the /10, not ``startswith("100.")`` — both neighbours pass."""
    _patch_dns(monkeypatch, {ip: ip})
    _reject_internal_url(f"http://{ip}/x", "image_url")


def test_call_site_still_yields_400_for_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_validate_external_url`` keeps 422 for a bad scheme and 400 for internal."""
    _patch_dns(monkeypatch, {"tailnet.example.com": "100.64.0.5"})
    with pytest.raises(HTTPException) as exc_info:
        _validate_external_url("ftp://example.com/img.png", "image_url")
    assert exc_info.value.status_code == 422

    with pytest.raises(HTTPException) as exc_info:
        _validate_external_url("https://tailnet.example.com/img.png", "image_url")
    assert exc_info.value.status_code == 400
