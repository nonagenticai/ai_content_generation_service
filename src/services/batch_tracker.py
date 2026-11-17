"""Batch state management with optional Redis persistence."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 3600.0  # 1 hour
_REDIS_TTL = 7200  # 2 hours


@dataclass
class BatchState:
    """State of a batch generation job."""

    batch_id: str
    job_ids: list[str] = field(default_factory=list)
    status: str = "pending"
    # Owner of the batch. Optional only so a snapshot written by an older
    # release still deserialises; a scoped read treats ``None`` as NOT owned
    # (fail-closed) — such entries age out with the 2h Redis TTL.
    tenant_id: str | None = None
    clips_submitted: int = 0
    total_cost: float = 0.0
    budget_limit: float = 5.0
    clips_completed: int = 0
    clips_failed: int = 0
    clips_skipped: int = 0
    # A child whose task was CANCELLED. Terminal like the other three: the clip
    # will never be produced, so it must be counted or the batch never reaches a
    # terminal ``computed_status``. Defaulted so a snapshot written by an older
    # release still deserialises.
    clips_cancelled: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def computed_status(self) -> str:
        total = len(self.job_ids)
        done = self.clips_completed + self.clips_failed + self.clips_skipped + self.clips_cancelled
        if done == 0:
            return "pending"
        if done < total:
            return "processing"
        if self.clips_cancelled == total:
            return "cancelled"
        if self.clips_failed == total:
            return "failed"
        if self.clips_completed == total:
            return "completed"
        return "partial"

    def to_dict(self) -> dict:
        """Serialize to dict for Redis storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BatchState:
        """Deserialize from dict."""
        return cls(**data)


class BatchTracker:
    """Tracker for batch generation state with optional Redis persistence."""

    def __init__(self, ttl: float = _DEFAULT_TTL, redis_client: RedisClient | None = None) -> None:
        self._batches: dict[str, BatchState] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl
        self._redis: RedisClient | None = redis_client

    def _redis_key(self, batch_id: str) -> str:
        return f"content_generation:batch:{batch_id}"

    async def _persist(self, state: BatchState) -> None:
        """Persist batch state to Redis."""
        if self._redis is None:
            return
        try:
            await self._redis.set(self._redis_key(state.batch_id), state.to_dict(), ttl=_REDIS_TTL)
        except OSError:
            logger.debug("Failed to persist batch %s to Redis", state.batch_id)

    async def _load_from_redis(self, batch_id: str) -> BatchState | None:
        """Load batch state from Redis."""
        if self._redis is None:
            return None
        try:
            data = await self._redis.get(self._redis_key(batch_id))
            if data and isinstance(data, dict):
                state = BatchState.from_dict(data)
                self._batches[batch_id] = state
                return state
        except OSError:
            logger.debug("Failed to load batch %s from Redis", batch_id)
        return None

    # ------------------------------------------------------------------
    # Cross-pod atomic counters
    # ------------------------------------------------------------------
    #
    # The batch snapshot (``content_generation:batch:{id}``) is a JSON blob — a
    # non-atomic read-modify-write across pods would lose counts (last-writer-
    # wins). The AUTHORITATIVE counters live in a separate Redis HASH mutated with
    # server-side atomic HINCRBY/HINCRBYFLOAT, so a clip finishing on ANY pod is
    # summed without a lost update. The snapshot is then refreshed from the hash
    # for the status endpoints (eventually-consistent mirror of the hash truth).

    _COUNTER_FIELDS = ("clips_completed", "clips_failed", "clips_skipped", "clips_cancelled")

    def _counter_key(self, batch_id: str) -> str:
        return f"content_generation:batch:{batch_id}:counters"

    def _raw_redis(self):  # noqa: ANN202 — redis.Redis, kept dynamic to avoid a hard import
        """Reach the underlying redis client for atomic hash ops.

        w2v_common.cache.RedisClient exposes no HINCRBY wrapper; reaching the raw
        client for server-side atomics is an accepted, documented pattern in this
        fleet (cf. the run_lease raw-redis reach). Returns None if unavailable.
        """
        if self._redis is None:
            return None
        try:
            return self._redis._ensure_connected()  # reach raw client — see docstring
        except RuntimeError:
            return None

    async def _incr_counters(self, batch_id: str, field: str, cost: float) -> dict | None:
        """Atomically bump one counter (+cost) and return the authoritative counts.

        Returns None when Redis is unavailable so the caller falls back to the
        pod-local in-memory increment (single-replica correctness preserved).
        """
        raw = self._raw_redis()
        if raw is None:
            return None
        key = self._counter_key(batch_id)
        try:
            await raw.hincrby(key, field, 1)
            if cost:
                await raw.hincrbyfloat(key, "clips_total_cost", cost)
            await raw.expire(key, _REDIS_TTL)
            return await raw.hgetall(key)
        except Exception:  # degrade to in-memory on any redis error
            logger.debug("Atomic counter update failed for batch %s", batch_id, exc_info=True)
            return None

    @staticmethod
    def _apply_counters(state: BatchState, counts: dict) -> None:
        """Overlay authoritative hash counts onto the in-memory/snapshot state."""

        def _as_int(v: object) -> int:
            try:
                return int(str(v))
            except TypeError, ValueError:
                return 0

        def _as_float(v: object) -> float:
            try:
                return float(str(v))
            except TypeError, ValueError:
                return 0.0

        state.clips_completed = _as_int(counts.get("clips_completed", 0))
        state.clips_failed = _as_int(counts.get("clips_failed", 0))
        state.clips_skipped = _as_int(counts.get("clips_skipped", 0))
        state.clips_cancelled = _as_int(counts.get("clips_cancelled", 0))
        state.total_cost = _as_float(counts.get("clips_total_cost", 0.0))
        state.status = state.computed_status

    async def _record_clip(self, batch_id: str, field: str, cost: float = 0.0) -> None:
        """Shared record path: atomic cross-pod increment with in-memory fallback.

        Loads the batch from Redis if this pod never created it (the missing
        ``_load_from_redis`` fallback that silently no-op'd cross-pod completions).
        """
        async with self._lock:
            state = self._batches.get(batch_id) or await self._load_from_redis(batch_id)
            counts = await self._incr_counters(batch_id, field, cost)
            if state is None:
                # Counters still advanced (no lost count); we just cannot recompute
                # terminal status without the snapshot. A clip for an unknown batch
                # is unexpected — log and move on.
                logger.warning("record %s for unknown batch %s (snapshot missing)", field, batch_id)
                return
            if counts is not None:
                self._apply_counters(state, counts)
            else:
                # No Redis — pod-local increment (single-replica correctness).
                setattr(state, field, getattr(state, field) + 1)
                state.total_cost += cost
                state.status = state.computed_status
            await self._persist(state)

    async def create_batch(
        self,
        batch_id: str,
        job_ids: list[str],
        budget_limit: float = 5.0,
        tenant_id: str | None = None,
    ) -> BatchState:
        """Create a new batch with the given job IDs.

        Args:
            batch_id: Unique batch identifier.
            job_ids: List of job IDs belonging to this batch.
            budget_limit: Maximum total spend in USD.
            tenant_id: Owner of the batch. Stamped here so
                :meth:`get_batch` can enforce ownership — this store is Redis
                (and process memory), which has no RLS backstop.

        Returns:
            The newly created BatchState.
        """
        async with self._lock:
            state = BatchState(
                batch_id=batch_id,
                job_ids=job_ids,
                clips_submitted=len(job_ids),
                budget_limit=budget_limit,
                tenant_id=tenant_id,
            )
            self._batches[batch_id] = state
            await self._persist(state)
            return state

    async def get_batch(self, batch_id: str, tenant_id: str | None = None) -> BatchState | None:
        """Retrieve batch state by ID, checking Redis if not in memory.

        Args:
            batch_id: Batch identifier.
            tenant_id: When given, the batch must ALSO be owned by this tenant
                or it reads as absent. Unlike the postgres-backed lookups this
                predicate has NO RLS backstop underneath it — the state lives
                in Redis and process memory — so it is the only thing standing
                between a guessed batch id and another tenant's clip list.
                Background workers pass nothing and still see every batch.

        Returns:
            BatchState if found (and owned, when scoped), None otherwise.
        """
        async with self._lock:
            state = self._batches.get(batch_id)
            if state is None:
                state = await self._load_from_redis(batch_id)
            if state is None:
                return None
            if tenant_id is not None and state.tenant_id != tenant_id:
                # Fail closed, including for a legacy snapshot with no owner.
                #
                # ROLLOUT HAZARD, accepted deliberately: a batch written by the
                # PREVIOUS release has ``tenant_id is None`` and therefore
                # matches nobody, so its RIGHTFUL owner gets 404 from
                # /batches/{id}/status and /result while the workers keep
                # generating and charging for its clips. This is bounded by the
                # 2h Redis TTL (``_REDIS_TTL``) — after that the snapshots are
                # gone and every batch in the store carries an owner.
                #
                # The alternative — treating an ownerless batch as belonging to
                # whoever asks — would make a guessed batch id from before the
                # deploy readable by ANY tenant, i.e. it reopens exactly the leak
                # this predicate closes, and "temporary" leniency of that shape
                # does not get removed on schedule. A missing owner is therefore
                # a permanent, tested refusal (see
                # ``test_legacy_ownerless_batch_is_refused_even_for_its_owner``),
                # not a transition window. Operator note: drain in-flight video
                # batches before deploying, or expect owner-visible 404s for up
                # to one TTL.
                if state.tenant_id is None:
                    logger.warning(
                        "Refusing ownerless (pre-upgrade) batch snapshot %s "
                        "requested by tenant %s; it will age out with the TTL",
                        batch_id,
                        tenant_id,
                    )
                return None
            # Overlay the authoritative hash counters so cross-pod completions are
            # reflected even for a pod that only read (never wrote) this batch.
            counts = None
            raw = self._raw_redis()
            if raw is not None:
                try:
                    counts = await raw.hgetall(self._counter_key(batch_id))
                except Exception:  # noqa: BLE001 — snapshot is an acceptable fallback
                    counts = None
            if counts:
                self._apply_counters(state, counts)
            return state

    async def record_clip_completed(self, batch_id: str, cost: float = 0.0) -> None:
        """Record a completed clip in the batch (atomic, cross-pod)."""
        await self._record_clip(batch_id, "clips_completed", cost=cost)

    async def record_clip_failed(self, batch_id: str) -> None:
        """Record a failed clip in the batch (atomic, cross-pod)."""
        await self._record_clip(batch_id, "clips_failed")

    async def record_clip_skipped(self, batch_id: str) -> None:
        """Record a budget-skipped clip in the batch (atomic, cross-pod)."""
        await self._record_clip(batch_id, "clips_skipped")

    async def record_clip_cancelled(self, batch_id: str) -> None:
        """Record a cancelled clip in the batch (atomic, cross-pod).

        Without this the ``asyncio.CancelledError`` arm of the batch worker bumped
        NO counter, so ``done < total`` held forever: ``computed_status`` stayed
        ``processing`` and ``GET /batches/{id}/result`` 409'd until the 2h Redis
        TTL and then 404'd. The clip was never accounted for anywhere.
        """
        await self._record_clip(batch_id, "clips_cancelled")

    async def cleanup_expired(self) -> int:
        """Remove batch states older than TTL. Returns count of removed entries."""
        now = time.time()
        async with self._lock:
            expired = [
                bid for bid, state in self._batches.items() if now - state.created_at > self._ttl
            ]
            for bid in expired:
                del self._batches[bid]
            return len(expired)
