"""File storage service for generated AI assets.

Object I/O is delegated to the shared ``w2v_common.storage.StorageService``
facade (the fleet-wide consolidation of the per-service MinIO clients). This
thin wrapper keeps the ai-content-generation domain concerns the facade does
not know about: the ``content_generation/{job_id}/{filename}`` key layout, the
local job-scratch directory, the traversal guard on stale/tampered local
paths, and content-type defaults. Both the local and MinIO backends stay live
(dual-backend) — routing is owned by the facade's configured backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

from w2v_common.storage import StorageService as CommonStorage

from src.config import get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Manages file storage for generated video and image assets over the facade."""

    def __init__(self) -> None:
        settings = get_settings()
        self._backend = settings.storage.backend
        self._local_path = Path(settings.storage.local_path)
        self._storage = CommonStorage(settings.storage)

    async def initialize(self) -> None:
        """Initialize the storage backend (create directories or buckets).

        The MinIO probe is best-effort: if the upstream is unreachable at boot
        we log and continue so the service still starts and serves non-storage
        routes; the bucket is created on the first write. Crashing here turned a
        transient minio blip into a fleet-wide w2v CrashLoop, so a boot-time
        failure must never propagate.
        """
        if self._backend == "minio":
            try:
                await self._storage.initialize()
                logger.info("MinIO storage initialized: bucket=%s", self._storage.bucket)
            except Exception as exc:  # noqa: BLE001 — startup must tolerate any minio failure
                logger.warning(
                    "MinIO bucket probe failed at startup (%s: %s) — proceeding "
                    "without verification; bucket will be created on first write.",
                    type(exc).__name__,
                    exc,
                )
        else:
            await self._storage.initialize()
            logger.info("Local storage initialized at %s", self._local_path)

    def get_job_dir(self, job_id: str) -> Path:
        """Get or create a local scratch directory for a specific job.

        Args:
            job_id: Job identifier used as directory name.

        Returns:
            Path to the job-specific directory.
        """
        job_dir = self._local_path / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    async def store_video(self, job_id: str, data: bytes, filename: str = "clip.mp4") -> str:
        """Store video data. Returns local path (local backend) or MinIO key."""
        key = f"content_generation/{job_id}/{filename}"
        return await self._storage.put_bytes(key, data, content_type="video/mp4")

    async def store_image(
        self,
        job_id: str,
        data: bytes,
        filename: str = "image.png",
        *,
        content_type: str = "image/png",
    ) -> str:
        """Store image data. Returns local path (local backend) or MinIO key."""
        key = f"content_generation/{job_id}/{filename}"
        return await self._storage.put_bytes(key, data, content_type=content_type)

    async def upload_to_minio(
        self,
        job_id: str,
        data: bytes,
        filename: str,
        *,
        content_type: str = "video/mp4",
    ) -> str | None:
        """Upload to MinIO (if configured). Returns MinIO key or None."""
        if self._backend != "minio":
            return None
        key = f"content_generation/{job_id}/{filename}"
        return await self._storage.put_bytes(key, data, content_type=content_type)

    async def retrieve(self, path: str) -> bytes | None:
        """Retrieve asset data from a local path or a MinIO key."""
        if self._backend == "minio" and not Path(path).is_absolute():
            return await self._storage.get_bytes(path)
        file_path = self._safe_local_path(path)
        if file_path is None:
            return None
        return await self._storage.get_bytes(str(file_path))

    def _safe_local_path(self, raw: str) -> Path | None:
        """Resolve ``raw`` and reject anything outside ``self._local_path``.

        Defense in depth: ``store_*`` only emits paths under the configured
        root, so the DB row should always round-trip cleanly. A rejection here
        means either a stale row from a different configuration or a tampered DB
        entry — return ``None`` (treated as "not found" by callers) rather than
        honouring the traversal.
        """
        root = self._local_path.resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._local_path / raw
        try:
            resolved = candidate.resolve()
        except OSError, RuntimeError:
            logger.warning("Could not resolve storage path: %s", raw)
            return None
        if not resolved.is_relative_to(root):
            logger.warning("Rejected path outside storage root: %s", raw)
            return None
        return resolved

    async def presign_url(self, key: str, expires_s: int = 3600) -> str | None:
        """Generate a presigned URL for a MinIO object (``None`` otherwise).

        Returns ``None`` for the local backend. When
        ``STORAGE_MINIO_EXTERNAL_ENDPOINT`` is configured the facade rewrites the
        internal host portion of the presigned URL so that clients outside
        Docker/k8s can reach MinIO directly.
        """
        return await self._storage.presign_get(key, expires=expires_s)

    async def delete_local(self, path: str) -> bool:
        """Delete a local file, pruning an emptied parent dir. Returns True if deleted."""
        file_path = self._safe_local_path(path)
        if file_path is None or not file_path.exists():
            return False
        await self._storage.delete_object(str(file_path))
        # Clean up empty parent directory (domain concern the facade does not own).
        parent = file_path.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError as e:
            logger.error("Failed to prune empty parent dir %s: %s", parent, e)
        return True

    async def delete_from_minio(self, key: str) -> bool:
        """Delete a MinIO object. Returns True if deleted."""
        return await self._storage.delete_object(key)

    async def health_check(self) -> bool:
        """Check storage backend health."""
        return await self._storage.check_health()
