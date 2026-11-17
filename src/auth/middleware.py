"""License token authentication middleware.

Thin re-export from w2v_common. All authentication logic lives in the shared
library; this module exists solely to preserve backward-compatible import paths
for ``src.auth.middleware``.
"""

# NOTE: ``DEFAULT_TENANT_ID`` is deliberately NOT re-exported here. Product-owner
# ruling: a ``tenant_id`` may never hold a sentinel, and an absent tenant is an
# ERROR, never a substituted default. Re-exporting it made a nil-UUID tenant
# reachable from ``src.auth.middleware`` -- billing buckets that as unattributed
# and never charges it. Use ``src.services.metering.require_tenant`` instead.
# (w2v_common 0.67.0 removes the name upstream as well.)
from w2v_common.auth.middleware import (
    PUBLIC_ENDPOINTS as PUBLIC_ENDPOINTS,
)
from w2v_common.auth.middleware import (
    PUBLIC_PREFIXES as PUBLIC_PREFIXES,
)
from w2v_common.auth.middleware import (
    TRUSTED_NETWORKS as TRUSTED_NETWORKS,
)
from w2v_common.auth.middleware import (
    AuthMiddleware as AuthMiddleware,
)
from w2v_common.auth.middleware import (
    _get_license_cache as _get_license_cache,
)
from w2v_common.auth.middleware import (
    _introspect_license as _introspect_license,
)
from w2v_common.auth.middleware import (
    _is_trusted_network as _is_trusted_network,
)

__all__ = [
    "PUBLIC_ENDPOINTS",
    "PUBLIC_PREFIXES",
    "TRUSTED_NETWORKS",
    "AuthMiddleware",
    "_get_license_cache",
    "_introspect_license",
    "_is_trusted_network",
]
