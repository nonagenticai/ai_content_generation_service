"""Per-job tenant provider-slot secrets: AES-256-GCM sealed at rest, bound to the job.

Pattern copied from ``w2v_orchestrator`` ``src/services/providers/crypto.py`` (P3).
A tenant slot's secret (a BYO image/video API key) arrives on a generation request
and must survive a pod crash so the reaper can resume the job on another replica.
It is persisted on the job row (``provider_secret_sealed``) as one opaque envelope::

    b"v1" | b"|" | kid | b"|" | nonce (12 bytes) | ciphertext + GCM tag (16 bytes)

* **kid** names the keyring entry that sealed it.
* **AAD** = the job id. The ciphertext is bound to the exact job it was written
  for: copying it onto another job row (another tenant's, say) fails the tag check.
* **Keyring** (``PROVIDER_SECRET_KEYS="kid:b64,..."``, ``src/config.py``) is ACG's  # pragma: allowlist secret
  OWN, per-service: the FIRST key encrypts, every key decrypts.

Fail closed, always: an empty keyring refuses to encrypt
(:class:`KeyringNotConfiguredError` -> the API answers 503); an unknown kid, a wrong
AAD, a tampered byte or a malformed envelope raise :class:`SecretDecryptError`.
No message in this module ever contains plaintext or key material.

The envelope is purged from the row when the job reaches a terminal state (a DB
trigger, ``purge_provider_secret_on_terminal`` -- see ``rls_resolvers``), so a
finished job never keeps a tenant credential at rest.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.config import PROVIDER_KEY_ID_PATTERN, get_settings, parse_provider_secret_keys

ENVELOPE_VERSION = b"v1"
_SEP = b"|"
NONCE_BYTES = 12
TAG_BYTES = 16


class ProviderSecretError(RuntimeError):
    """Base: a slot secret cannot be sealed or opened. Never carries the secret."""


class KeyringNotConfiguredError(ProviderSecretError):
    """``PROVIDER_SECRET_KEYS`` is empty -- slot requests must be refused (503)."""


class SecretDecryptError(ProviderSecretError):
    """The envelope cannot be opened: unknown kid, wrong AAD, tampered, malformed."""


def aad(job_id: str) -> bytes:
    """The associated data binding a ciphertext to its exact job row."""
    return f"acg-job|{job_id}".encode()


def _split(ciphertext: bytes) -> tuple[str, bytes, bytes]:
    head, sep1, rest = bytes(ciphertext).partition(_SEP)
    kid_raw, sep2, body = rest.partition(_SEP)
    if head != ENVELOPE_VERSION or not sep1 or not sep2:
        raise SecretDecryptError("provider slot secret envelope is malformed")
    try:
        kid = kid_raw.decode("ascii")
    except UnicodeDecodeError:
        raise SecretDecryptError("provider slot secret envelope is malformed") from None
    if not PROVIDER_KEY_ID_PATTERN.fullmatch(kid) or len(body) < NONCE_BYTES + TAG_BYTES:
        raise SecretDecryptError("provider slot secret envelope is malformed")
    return kid, body[:NONCE_BYTES], body[NONCE_BYTES:]


class Keyring:
    """An ordered set of AES-256 keys: ``keys[0]`` encrypts, all decrypt."""

    def __init__(self, entries: Sequence[tuple[str, bytes]]) -> None:
        self._order: tuple[str, ...] = tuple(kid for kid, _ in entries)
        if len(set(self._order)) != len(self._order):
            raise ValueError("keyring repeats a key id")
        self._keys: dict[str, AESGCM] = {kid: AESGCM(key) for kid, key in entries}

    @classmethod
    def from_env_value(cls, raw: str) -> Keyring:
        return cls(parse_provider_secret_keys(raw))

    def __repr__(self) -> str:  # never render key material
        return f"Keyring(kids={list(self._order)!r})"

    @property
    def configured(self) -> bool:
        return bool(self._order)

    @property
    def primary_key_id(self) -> str:
        if not self._order:
            raise KeyringNotConfiguredError("PROVIDER_SECRET_KEYS is not configured")
        return self._order[0]

    def encrypt(self, secret: str, *, job_id: str) -> bytes:
        """Seal ``secret`` for ``job_id`` under the primary key; returns the envelope."""
        kid = self.primary_key_id
        nonce = os.urandom(NONCE_BYTES)
        sealed = self._keys[kid].encrypt(nonce, secret.encode(), aad(job_id))
        return _SEP.join((ENVELOPE_VERSION, kid.encode("ascii"), nonce + sealed))

    def decrypt(self, ciphertext: bytes, *, job_id: str) -> str:
        """Open an envelope sealed for ``job_id``.

        An EMPTY keyring is a deployment fault, not a bad row: it raises
        :class:`KeyringNotConfiguredError`, never :class:`SecretDecryptError`.
        """
        if not self._order:
            raise KeyringNotConfiguredError("PROVIDER_SECRET_KEYS is not configured")
        kid, nonce, sealed = _split(ciphertext)
        cipher = self._keys.get(kid)
        if cipher is None:
            raise SecretDecryptError(f"provider slot secret sealed with unknown key id {kid!r}")
        try:
            plain = cipher.decrypt(nonce, sealed, aad(job_id))
        except InvalidTag:
            raise SecretDecryptError(
                "provider slot secret failed authentication (wrong key, job binding or tampered)"
            ) from None
        return plain.decode()


def get_keyring() -> Keyring:
    """The process keyring from settings (parsed per call; settings are cached)."""
    return Keyring(get_settings().provider_secrets.keyring_entries())
