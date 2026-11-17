"""Unit tests for auth middleware and User model."""

from __future__ import annotations

import ipaddress
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from starlette.responses import JSONResponse
from w2v_common.tenant import MissingTenantError

from src.auth.middleware import (
    AuthMiddleware,
    _get_license_cache,
    _introspect_license,
    _is_trusted_network,
)
from src.auth.models import User
from tests.conftest import TENANT_A, TENANT_B

# ===========================================================================
# User model
# ===========================================================================


class TestUserModel:
    """Tests for the User pydantic model."""

    def test_create_with_all_fields(self):
        user = User(sub="sub-1", preferred_username="alice", tenant_id=uuid.UUID(TENANT_A))
        assert user.sub == "sub-1"
        assert user.preferred_username == "alice"
        assert user.tenant_id == uuid.UUID(TENANT_A)

    def test_default_preferred_username(self):
        user = User(sub="tenant-1")
        assert user.preferred_username == "anonymous"

    def test_default_tenant_id_is_none(self):
        user = User(sub="tenant-1")
        assert user.tenant_id is None

    def test_effective_tenant_id_uses_tenant_id(self):
        user = User(sub="sub-1", tenant_id=uuid.UUID(TENANT_A))
        assert user.effective_tenant_id == uuid.UUID(TENANT_A)

    def test_effective_tenant_id_refuses_to_fall_back_to_sub(self):
        """w2v_common 0.67.0 DELETED the ``tenant_id or sub`` fallback.

        That fallback minted a tenant equal to the subject: a tenant that was
        never registered, and (because ``sub`` is free-form) not even a uuid.
        Absence is now an error, so the refusal is what this must assert.
        """
        user = User(sub="sub-1", tenant_id=None)
        with pytest.raises(MissingTenantError):
            _ = user.effective_tenant_id

    def test_a_non_uuid_tenant_is_rejected_at_construction(self):
        with pytest.raises(ValidationError):
            # ty: a str is not `UUID | None` -- that is the point of the test:
            # a legacy slug must be rejected at the pydantic boundary too.
            User(sub="sub-1", tenant_id="t1")


# ===========================================================================
# _is_trusted_network
# ===========================================================================


# w2v_common 0.89.0: TRUSTED_CIDRS defaults to EMPTY -- no source IP is trusted
# unless the operator names the networks (cell plan B22). Production config for
# this service sets none. Tests that exercise a trust lane therefore configure
# one explicitly instead of leaning on a library default that no longer exists.
_EXPLICIT_TRUSTED_CIDRS = "127.0.0.1/32,::1/128,172.16.0.0/12,10.0.0.0/8"


@pytest.fixture
def explicit_trusted_cidrs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure TRUSTED_CIDRS explicitly (module-level, auto-restored)."""
    monkeypatch.setattr(
        "w2v_common.auth.middleware.TRUSTED_NETWORKS",
        [ipaddress.ip_network(c, strict=False) for c in _EXPLICIT_TRUSTED_CIDRS.split(",")],
    )


@pytest.fixture
def no_trusted_cidrs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped/production shape: no trust lane configured."""
    monkeypatch.setattr("w2v_common.auth.middleware.TRUSTED_NETWORKS", [])


@pytest.mark.usefixtures("explicit_trusted_cidrs")
class TestIsTrustedNetwork:
    """Tests for the _is_trusted_network helper with lanes configured explicitly."""

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "10.0.0.1", "172.16.0.1"],
    )
    def test_trusted_addresses(self, host: str):
        assert _is_trusted_network(host) is True

    @pytest.mark.parametrize(
        "host",
        # 192.168.0.0/16 is intentionally excluded from the default
        # TRUSTED_NETWORKS in w2v_common >=0.6.0 to close the SSR-proxy
        # source-IP bypass (see project_orchestrator_trusted_networks_bypass).
        ["8.8.8.8", "203.0.113.1", "192.168.1.1"],
    )
    def test_untrusted_addresses(self, host: str):
        assert _is_trusted_network(host) is False

    def test_none_returns_false(self):
        assert _is_trusted_network(None) is False

    def test_empty_string_returns_false(self):
        assert _is_trusted_network("") is False

    def test_invalid_input_returns_false(self):
        assert _is_trusted_network("not-an-ip") is False


@pytest.mark.usefixtures("no_trusted_cidrs")
class TestIsTrustedNetworkNoLanesConfigured:
    """With TRUSTED_CIDRS unset (production), not even loopback is trusted."""

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "::1", "10.0.0.1", "172.16.0.1", "192.168.1.1"],
    )
    def test_nothing_is_trusted(self, host: str):
        assert _is_trusted_network(host) is False


# ===========================================================================
# _extract_license_token
# ===========================================================================


class TestExtractLicenseToken:
    """Tests for AuthMiddleware._extract_license_token."""

    def _middleware(self) -> AuthMiddleware:
        return AuthMiddleware(app=MagicMock())

    def test_opaque_authorization_bearer_is_a_license_token(self):
        """w2v_common 0.74.0 contract change.

        Until 0.74.0 an ``Authorization: Bearer`` was reserved exclusively for
        upstream JWT auth and was NEVER read as a licence token (the 0.6.5 /
        FR-401 behaviour this test used to assert). 0.74.0 deliberately widened
        it: an OPAQUE bearer on ``Authorization`` is now accepted as a licence
        token, because that is the only header the ``license-authz`` ext-authz
        gate reads, and w2v has to be able to sit behind that gate.

        The JWT path is untouched — see the JWT-shaped case below.
        """
        mw = self._middleware()
        request = MagicMock()
        request.headers = {"Authorization": "Bearer my-secret-token"}
        assert mw._extract_license_token(request) == "my-secret-token"
        # Sibling assertion: the same token is still the JWT-bearer candidate.
        assert mw._extract_jwt_bearer(request) == "my-secret-token"

    def test_jwt_shaped_authorization_bearer_is_not_a_license_token(self):
        """The other half of the 0.74.0 contract: only an OPAQUE bearer is
        widened. A JWT-shaped bearer (three non-empty dot-separated segments)
        still goes down the JWT path and is NEVER submitted to licence
        introspection — the invariant behind the 2026-04-26 prod 401s."""
        mw = self._middleware()
        request = MagicMock()
        # Fabricated, unsigned, never valid anywhere — only its SHAPE matters
        # here (three non-empty dot-separated segments, per _looks_like_jwt).
        jwt_shaped = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhIn0.c2ln"  # pragma: allowlist secret
        request.headers = {"Authorization": f"Bearer {jwt_shaped}"}
        assert mw._extract_license_token(request) is None
        assert mw._extract_jwt_bearer(request) == jwt_shaped

    def test_bearer_in_x_tenant_id_header(self):
        mw = self._middleware()
        request = MagicMock()
        request.headers = {"X-Tenant-Id": "Bearer tenant-token-123"}
        assert mw._extract_license_token(request) == "tenant-token-123"

    def test_x_tenant_id_bearer_wins_over_authorization_bearer(self):
        """Precedence, per w2v_common 0.74.0.

        Both headers are now licence-token sources, so precedence is
        load-bearing rather than academic: ``X-Tenant-Id`` is checked FIRST.
        ``BaseServiceClient`` puts the CALLER's token on ``X-Tenant-Id`` while
        its constructor-baked SERVICE-ACCOUNT token sits on ``Authorization``,
        so preferring ``Authorization`` would silently re-attribute every
        delegated call from the customer tenant to the service account's.
        """
        mw = self._middleware()
        request = MagicMock()
        request.headers = {
            "X-Tenant-Id": "Bearer from-tenant-header",
            "Authorization": "Bearer from-auth-header",
        }
        assert mw._extract_license_token(request) == "from-tenant-header"
        # Authorization is still surfaced as a JWT-bearer candidate.
        assert mw._extract_jwt_bearer(request) == "from-auth-header"

    def test_no_token_present(self):
        mw = self._middleware()
        request = MagicMock()
        request.headers = {}
        assert mw._extract_license_token(request) is None
        assert mw._extract_jwt_bearer(request) is None

    def test_non_bearer_headers_ignored(self):
        mw = self._middleware()
        request = MagicMock()
        request.headers = {
            "Authorization": "Basic dXNlcjpwYXNz",
            "X-Tenant-Id": "plain-tenant-id",
        }
        assert mw._extract_license_token(request) is None
        # Basic-auth scheme is also not a JWT bearer.
        assert mw._extract_jwt_bearer(request) is None


# ===========================================================================
# dispatch -- helpers
# ===========================================================================


def _make_request(
    path: str = "/api/content/videos/generate",
    host: str = "8.8.8.8",
    headers: dict | None = None,
) -> MagicMock:
    """Build a mock Request for middleware dispatch tests."""
    request = MagicMock()
    request.url.path = path
    request.client.host = host
    request.headers = headers or {}
    request.state = MagicMock()
    return request


def _make_call_next() -> AsyncMock:
    """Return an AsyncMock call_next that produces a 200 response."""
    response = MagicMock(status_code=200)
    return AsyncMock(return_value=response)


# ===========================================================================
# dispatch -- public endpoints
# ===========================================================================


class TestDispatchPublicEndpoints:
    """Public endpoints must bypass auth entirely."""

    @pytest.mark.parametrize("path", ["/health", "/docs", "/openapi.json"])
    async def test_public_endpoints_skip_auth(self, path: str):
        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(path=path)
        call_next = _make_call_next()

        response = await mw.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "path",
        ["/health/live", "/docs/something", "/health/deep", "/api/health/ready"],
    )
    async def test_public_prefixes_skip_auth(self, path: str):
        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(path=path)
        call_next = _make_call_next()

        response = await mw.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert response.status_code == 200


# ===========================================================================
# dispatch -- valid / invalid tokens
# ===========================================================================


class TestDispatchTokenAuth:
    """Token-based authentication paths."""

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_valid_token_populates_request_state(self, mock_introspect):
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": TENANT_A,
            "subject_id": "user-1",
        }

        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(headers={"X-Tenant-Id": "Bearer valid-token"})
        call_next = _make_call_next()

        response = await mw.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200
        # Verify request.state was populated
        assert request.state.tenant_id == TENANT_A
        assert request.state.user.sub == TENANT_A
        assert request.state.user.preferred_username == "user-1"
        assert request.state.user.tenant_id == uuid.UUID(TENANT_A)
        assert request.state.license == {
            "active": True,
            "tenant_id": TENANT_A,
            "subject_id": "user-1",
        }

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_inactive_token_returns_401(self, mock_introspect):
        mock_introspect.return_value = {"active": False, "reason": "expired"}

        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(headers={"X-Tenant-Id": "Bearer expired-token"})
        call_next = _make_call_next()

        response = await mw.dispatch(request, call_next)

        call_next.assert_not_awaited()
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_introspect_returns_none_gives_401(self, mock_introspect):
        mock_introspect.return_value = None

        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(headers={"X-Tenant-Id": "Bearer bad-token"})
        call_next = _make_call_next()

        response = await mw.dispatch(request, call_next)

        call_next.assert_not_awaited()
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401


# ===========================================================================
# dispatch -- trusted network
# ===========================================================================


@pytest.mark.usefixtures("explicit_trusted_cidrs")
class TestDispatchTrustedNetwork:
    """Trusted-network (no token) authentication paths.

    The lane is configured explicitly so the 401 cases below are refused for
    the reason they name (no token / no tenant), not because 127.0.0.1 simply
    stopped being trusted when w2v_common 0.89.0 emptied the default.
    """

    async def test_trusted_network_with_x_tenant_id_opt_in(self):
        """Legacy fail-open path: AUTH_TRUST_INCLUSTER_IDENTITY=true (opt-in in 0.6.9)."""
        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(
            host="127.0.0.1",
            headers={"X-Tenant-Id": TENANT_B},
        )
        call_next = _make_call_next()

        with patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True):
            response = await mw.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200
        assert request.state.tenant_id == TENANT_B
        assert request.state.user.sub == TENANT_B
        assert request.state.user.preferred_username == "internal"
        assert request.state.license is None

    async def test_trusted_network_default_tenant_opt_in(self):
        """Even WITH the legacy opt-in trust, a tenant-less request is refused.

        REWRITTEN TWICE. It originally asserted
        ``request.state.tenant_id == DEFAULT_TENANT_ID`` (the nil-UUID
        sentinel); when that sentinel was deleted it asserted the request still
        passed the middleware and was refused later, at the route layer.
        w2v_common 0.67.0 moves the refusal INTO the middleware: with
        AUTH_TRUST_INCLUSTER_IDENTITY=true and no X-Tenant-Id there is nothing
        to trust, so the request never reaches the app at all. That is strictly
        stronger than the previous contract, and it is what must be asserted.
        (The sibling below covers the same shape WITHOUT the opt-in.)
        """
        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(host="127.0.0.1", headers={})
        call_next = _make_call_next()

        with patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True):
            response = await mw.dispatch(request, call_next)

        call_next.assert_not_awaited()
        assert response.status_code == 401

    async def test_trusted_network_no_token_fail_closed_default(self):
        """w2v_common 0.6.9: fail-closed by default — trusted-net unauth → 401."""
        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(host="127.0.0.1", headers={})
        call_next = _make_call_next()

        response = await mw.dispatch(request, call_next)

        call_next.assert_not_awaited()
        assert response.status_code == 401

    async def test_untrusted_no_token_returns_401(self):
        mw = AuthMiddleware(app=MagicMock())
        request = _make_request(host="8.8.8.8", headers={})
        call_next = _make_call_next()

        response = await mw.dispatch(request, call_next)

        call_next.assert_not_awaited()
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401
        raw_body = response.body
        assert raw_body is not None
        body_text = bytes(raw_body).decode()
        assert "Missing authentication token" in body_text


# ===========================================================================
# _introspect_license caching (TTLCache)
# ===========================================================================


class TestIntrospectLicenseCaching:
    """Verify that _introspect_license uses the TTLCache."""

    def setup_method(self) -> None:
        _get_license_cache().clear()

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_second_call_uses_cache(self, mock_async_client_cls):
        """First call hits the HTTP endpoint; second call returns cached data."""
        _get_license_cache().clear()

        introspect_result = {"active": True, "tenant_id": "cached-tenant"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = introspect_result

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance

        # First call -- should make HTTP request
        result1 = await _introspect_license("token-abc")
        assert result1 == introspect_result
        assert mock_client_instance.post.await_count == 1

        # Second call -- should use the cache, no new HTTP request
        result2 = await _introspect_license("token-abc")
        assert result2 == introspect_result
        assert mock_client_instance.post.await_count == 1  # still 1

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_inactive_token_is_not_cached(self, mock_async_client_cls):
        """Inactive tokens should NOT be placed in the cache."""
        _get_license_cache().clear()

        introspect_result = {"active": False, "reason": "revoked"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = introspect_result

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance

        result = await _introspect_license("bad-token")
        assert result == introspect_result
        # Cache should be empty because active=False
        assert len(_get_license_cache()) == 0

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_introspect_http_failure_returns_none(self, mock_async_client_cls):
        """When the HTTP call raises, return None."""
        import httpx

        _get_license_cache().clear()

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance

        result = await _introspect_license("fail-token")
        assert result is None
