"""User model for authentication.

Thin re-export from w2v_common. The ``User`` pydantic model is defined in the
shared library; this module preserves the ``src.auth.models`` import path.
"""

from w2v_common.auth.models import User as User

__all__ = ["User"]
