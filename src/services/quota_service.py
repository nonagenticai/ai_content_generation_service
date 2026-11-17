"""Per-tenant usage quota enforcement."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from redis.exceptions import RedisError

from src.observability.metrics import record_quota_degraded

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

# TTLs are part of the stored contract, not an implementation detail: the
# daily key must outlive the day it counts (48h covers any UTC/local skew and
# late completions), and the concurrent gauge must self-heal within an hour if
# a pod dies between start and completion. An atomic rewrite that dropped
# these would trade a lost-update bug for an unbounded key leak.
_DAILY_TTL_SECONDS = 172800  # 48h
_CONCURRENT_TTL_SECONDS = 3600  # 1h

# The concurrent-gauge KEY outlives its longest-lived member so a live holder
# can never have the key expire out from under it. Reclaim is per-MEMBER (see
# below), so this is only garbage collection for a tenant that went idle.
_CONCURRENT_KEY_TTL_SECONDS = _CONCURRENT_TTL_SECONDS + 60

# ---------------------------------------------------------------------------
# Atomic counter scripts
#
# The service runs at 2 replicas, so GET -> mutate in Python -> SET is a
# lost-update generator: both replicas read the same value and the second
# write silently discards the first. Every discarded write is a generation
# the tenant is not charged or throttled for.
#
# These scripts run server-side in a single round trip, so the read and the
# write cannot be interleaved. Same approach as the token bucket in
# ``src/middleware/rate_limit.py``; ``src/services/batch_tracker.py`` reaches
# the raw client for HINCRBY for the same reason.
#
# The daily value keeps the exact ``{"generations": int, "spend": float}``
# JSON object the previous implementation wrote, so a rolling upgrade reads
# and writes the same keys as the pods it is replacing — no migration, no
# reset of a live tenant's counters mid-day.
#
# ---------------------------------------------------------------------------
# ⛔ WHY THE CONCURRENT GAUGE IS A ZSET OF job_ids AND NOT AN INCR/DECR COUNTER
#
# It used to be ``INCR``/``DECR`` on a plain key carrying a 1h TTL, and three
# call-site comments in this repo justified their error handling with "the slot
# leaks only until its 1h TTL". **That TTL could never fire.** Both scripts
# re-issued ``EXPIRE`` on the counter key on EVERY start and EVERY completion,
# so for any tenant generating at least once an hour the key's TTL was pushed
# forward forever and a unit leaked by a SIGKILLed pod was never reclaimed. It
# is the same shape that wedged video_production for 9h: a lease whose TTL is
# defeated by something that renews it unconditionally. Symptom: a busy tenant
# drifts to a permanent 429 "Concurrent job limit exceeded" with nothing
# running and ZERO error lines anywhere.
#
# A counter is also not idempotent, and this service releases more times than
# it acquires. ``ReaperService._dispatch_video`` / ``_dispatch_image``
# re-dispatch a reclaimed SINGLE job straight into ``_process_job_bg`` /
# ``_process_concept_image_job_bg``, whose ``finally`` releases unconditionally
# because "the route already recorded the start" — but on the reclaim path
# there IS no route. Every reclaim therefore spent one DECR against zero INCRs,
# and on a rollout that fires for every in-flight job. That is exactly the
# defect ``_handle_batch_job_exception`` documents as fixed for batch children:
# it "zeroed the tenant's concurrency gauge (cancelling slots held by its other
# in-flight generations)", i.e. it silently disabled the cap on paid GPU work.
#
# Holding one ZSET MEMBER per job_id, scored with its expiry DEADLINE, fixes
# both at once and is the shape already proven in
# ``bdd_playwright_service.services.background``:
#
# * the score is a deadline, and every script prunes ``ZREMRANGEBYSCORE`` up to
#   ``now`` before counting — so a holder that is cancelled, evicted or
#   SIGKILLed before its release runs frees its slot at its own deadline. The
#   key's TTL is no longer load-bearing for reclaim, so refreshing it is safe.
# * membership is keyed by job_id, so release is IDEMPOTENT (``ZREM`` of an
#   absent member is a no-op) and a double release can never free a DIFFERENT
#   job's slot, and a re-acquire by the reaper re-scores that job's own member
#   instead of inventing a second one.
#
# ⚠️ A re-acquire DOES push one job's deadline out, so do NOT call
# ``record_generation_start`` from a periodic loop — that would reintroduce the
# unconditional-renewal bug per-member. It is called once per dispatch, and a
# re-dispatch is bounded by the reaper's own max_attempts / max_lifetime.
# ---------------------------------------------------------------------------

_READ_DAILY_LUA = """
local generations = 0
local spend = 0
local raw = redis.call('GET', KEYS[1])
if raw then
    local ok, decoded = pcall(cjson.decode, raw)
    if ok and type(decoded) == 'table' then
        generations = tonumber(decoded['generations']) or 0
        spend = tonumber(decoded['spend']) or 0
    end
end
"""

# Prune every holder whose DEADLINE has passed, then count what is left. This
# is what makes a leaked slot reclaimable at all — it runs before every read
# and every write of the gauge.
_PRUNE_CONCURRENT_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', ARGV[4])
"""

# KEYS[1]=daily key, KEYS[2]=concurrent zset
# ARGV[1]=daily ttl, ARGV[2]=concurrent key ttl, ARGV[3]=job_id,
# ARGV[4]=now (epoch s), ARGV[5]=this holder's deadline (epoch s)
_RECORD_START_LUA = (
    _READ_DAILY_LUA
    + _PRUNE_CONCURRENT_LUA
    + """
generations = generations + 1
redis.call('SETEX', KEYS[1], tonumber(ARGV[1]),
           cjson.encode({generations = generations, spend = spend}))

redis.call('ZADD', KEYS[2], ARGV[5], ARGV[3])
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[2]))
local concurrent = redis.call('ZCARD', KEYS[2])

return {generations, tostring(spend), concurrent}
"""
)

# KEYS[1]=daily key, KEYS[2]=concurrent zset
# ARGV[1]=daily ttl, ARGV[2]=concurrent key ttl, ARGV[3]=job_id,
# ARGV[4]=now (epoch s), ARGV[6]=cost
_RECORD_COMPLETE_LUA = (
    _READ_DAILY_LUA
    + _PRUNE_CONCURRENT_LUA
    + """
local cost = tonumber(ARGV[6]) or 0

-- ZREM of an absent member is a no-op, so a second release for the same job
-- (the reaper re-dispatch path, a retried teardown) cannot free some OTHER
-- job's slot the way a bare DECR did.
redis.call('ZREM', KEYS[2], ARGV[3])
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[2]))
local concurrent = redis.call('ZCARD', KEYS[2])

if cost > 0 then
    spend = spend + cost
    redis.call('SETEX', KEYS[1], tonumber(ARGV[1]),
               cjson.encode({generations = generations, spend = spend}))
end

return {generations, tostring(spend), concurrent}
"""
)

# KEYS[1]=daily key (unused, kept so every script shares one KEYS layout)
# KEYS[2]=concurrent zset, ARGV[4]=now (epoch s)
_READ_CONCURRENT_LUA = (
    _PRUNE_CONCURRENT_LUA
    + """
return redis.call('ZCARD', KEYS[2])
"""
)

# RedisError is NOT an OSError subclass, so it has to be named explicitly or an
# outage would escape as an unhandled 500. RuntimeError covers eval_script on a
# client that was never connected.
_REDIS_FAILURE = (RedisError, OSError, ValueError, RuntimeError)

# ---------------------------------------------------------------------------
# Degraded mode: what happens when the quota store is unreachable
#
# The old posture was unconditional fail-open — for the whole duration of a
# Redis outage EVERY paid GPU generation was admitted, silently, with the
# recording paths dropping the spend as well. The gap was unbounded and, worse,
# unmeasurable: nothing errored and nothing counted it.
#
# Hard fail-closed is not the answer either. Redis here is a counter cache, not
# the system of record (Postgres holds the jobs and their costs), and a blip
# would take all paid generation down with it — an availability outage caused
# by a component that is not on the critical path of generating anything.
#
# The posture is therefore BOUNDED fail-open, and it is deliberately three
# things at once:
#
#   1. A short retry. Most Redis failures here are a dropped connection that
#      the next call reconnects. Retrying once absorbs the blip without any
#      policy consequence at all.
#   2. A per-pod spend ceiling for the outage window. Each admission taken
#      without a readable cap charges its OWN estimated cost against
#      ``degraded_spend_budget``; once that is exhausted admission is REFUSED.
#      This is the "fail-closed above a configurable threshold" option,
#      generalised: it bounds the total unmetered spend an outage can cause per
#      pod instead of only the size of one job. It resets when Redis recovers.
#   3. ERROR-level logs plus ``ai_content_gen_quota_degraded_total`` on every
#      single degraded decision. A silent fail-open window cannot be alerted
#      on, cannot be sized afterwards, and cannot be reconciled against the
#      bill — which is how this class of defect stays invisible for months.
#
# A caller that passes no estimated_cost (or a provider we cannot price) still
# consumes ``_DEGRADED_MIN_CHARGE``, so an unpriced flood is bounded too rather
# than being free forever.
# ---------------------------------------------------------------------------
_DEGRADED_RETRY_DELAY_S = 0.2
_DEGRADED_MIN_CHARGE = 0.10
_DEFAULT_DEGRADED_SPEND_BUDGET = 25.0


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""

    allowed: bool
    reason: str | None = None
    remaining_generations: int | None = None
    remaining_spend: float | None = None
    remaining_concurrent: int | None = None


class QuotaService:
    """Enforces per-tenant quotas using Redis counters.

    Tracks daily generation counts, daily spend, and concurrent
    job counts per tenant. All state is stored in Redis with TTLs.

    Attributes:
        _redis: Redis client for quota state.
        _max_daily_generations: Maximum generations per day.
        _max_daily_spend: Maximum spend in USD per day.
        _max_concurrent_jobs: Maximum concurrent jobs per tenant.
    """

    def __init__(
        self,
        redis_client: RedisClient | None = None,
        max_daily_generations: int = 100,
        max_daily_spend: float = 50.0,
        max_concurrent_jobs: int = 10,
        degraded_spend_budget: float = _DEFAULT_DEGRADED_SPEND_BUDGET,
    ) -> None:
        self._redis = redis_client
        self._max_daily_generations = max_daily_generations
        self._max_daily_spend = max_daily_spend
        self._max_concurrent_jobs = max_concurrent_jobs
        # Degraded-window state. Pod-local by design: it bounds what THIS
        # replica can admit while blind, and there is nowhere shared to put it
        # precisely because the shared store is the thing that is down.
        self._degraded_spend_budget = degraded_spend_budget
        self._degraded_since: float | None = None
        self._degraded_spend = 0.0

    def _daily_key(self, tenant_id: str) -> str:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return f"content_generation:quota:daily:{tenant_id}:{today}"

    def _concurrent_key(self, tenant_id: str) -> str:
        """Key of the per-tenant live-holder set.

        ⚠️ The namespace is DELIBERATELY not the old ``…:concurrent:{tenant}``.
        That key is a Redis STRING (the old INCR/DECR counter) and this code
        runs ZSET commands, so during a rolling upgrade an old-format key still
        in Redis would answer every new pod with ``WRONGTYPE`` — i.e. every
        admission check failing into degraded mode across the whole fleet. A
        fresh namespace makes the two formats disjoint; the abandoned string key
        carries the old 1h TTL and collects itself once the last old pod stops
        touching it.
        """
        return f"content_generation:quota:concurrent:jobs:{tenant_id}"

    # ------------------------------------------------------------------
    # Degraded mode
    # ------------------------------------------------------------------

    def _enter_degraded(self, phase: str, exc: BaseException) -> None:
        """Mark the quota store unreachable and make the fact observable."""
        if self._degraded_since is None:
            self._degraded_since = time.monotonic()
            self._degraded_spend = 0.0
        logger.error(
            "Quota store unreachable during %s; quota enforcement is DEGRADED "
            "(paid generation is no longer capped by the daily counters): %s",
            phase,
            exc,
            exc_info=True,
        )

    def _exit_degraded(self) -> None:
        """Called on every successful read; closes an open degraded window."""
        if self._degraded_since is None:
            return
        logger.error(
            "Quota store recovered after %.1fs degraded; $%.2f of generation was "
            "admitted on this pod without a readable daily cap",
            time.monotonic() - self._degraded_since,
            self._degraded_spend,
        )
        self._degraded_since = None
        self._degraded_spend = 0.0

    async def _read_concurrent(self, tenant_id: str) -> int:
        """Prune expired holders, then count the live ones. Raises on Redis faults.

        The prune is a WRITE, so this is not a pure read — deliberately. A gauge
        whose stale entries are only ever removed by the releasing caller cannot
        recover from a caller that never gets to release, which is the whole
        defect this replaced. Reading is the one operation guaranteed to happen
        while a tenant is being throttled by a leak, so it is the right place to
        collect.
        """
        redis = self._redis
        if redis is None:  # pragma: no cover - callers gate on this
            raise RuntimeError("quota store is not configured")
        now = time.time()
        raw = await redis.eval_script(
            _READ_CONCURRENT_LUA,
            keys=[self._daily_key(tenant_id), self._concurrent_key(tenant_id)],
            args=["", _CONCURRENT_KEY_TTL_SECONDS, "", now],
        )
        return int(raw or 0)

    async def _read_counters(self, tenant_id: str) -> tuple[int, float, int]:
        """Read (daily_generations, daily_spend, concurrent). Raises on Redis faults."""
        redis = self._redis
        if redis is None:  # pragma: no cover - callers gate on this
            raise RuntimeError("quota store is not configured")
        daily_data = await redis.get(self._daily_key(tenant_id))
        daily_gens = 0
        daily_spend = 0.0
        if daily_data and isinstance(daily_data, dict):
            daily_gens = daily_data.get("generations", 0)
            daily_spend = daily_data.get("spend", 0.0)
        concurrent = await self._read_concurrent(tenant_id)
        return daily_gens, daily_spend, concurrent

    def _degraded_decision(self, tenant_id: str, estimated_cost: float) -> QuotaCheckResult:
        """Admit or refuse while blind, charging the pod's degraded budget."""
        charge = max(estimated_cost, _DEGRADED_MIN_CHARGE)
        if self._degraded_spend + charge > self._degraded_spend_budget:
            record_quota_degraded(phase="check", outcome="rejected")
            logger.error(
                "Quota store unreachable and this pod's degraded spend budget "
                "($%.2f) is exhausted ($%.2f used); REFUSING paid generation "
                "for tenant %s (estimated $%.2f)",
                self._degraded_spend_budget,
                self._degraded_spend,
                tenant_id,
                charge,
            )
            return QuotaCheckResult(
                allowed=False,
                reason=(
                    "Quota store unavailable; the degraded spend budget "
                    f"(${self._degraded_spend_budget:.2f}) is exhausted"
                ),
                remaining_spend=0.0,
            )
        self._degraded_spend += charge
        record_quota_degraded(phase="check", outcome="admitted")
        logger.error(
            "Quota store unreachable; ADMITTING generation for tenant %s UNMETERED "
            "against the degraded budget ($%.2f of $%.2f used, this job $%.2f)",
            tenant_id,
            self._degraded_spend,
            self._degraded_spend_budget,
            charge,
        )
        return QuotaCheckResult(
            allowed=True,
            remaining_spend=max(0.0, self._degraded_spend_budget - self._degraded_spend),
        )

    async def check_quota(
        self, tenant_id: str, estimated_cost: float = 0.0, *, tenant_funded: bool = False
    ) -> QuotaCheckResult:
        """Check if a tenant is within quotas.

        ``tenant_funded`` marks a job that runs on the TENANT's own provider key
        (a provider slot): it costs the platform nothing, so the platform daily
        SPEND cap (and the degraded spend budget) does not apply to it. The
        generation-count and concurrency limits still do -- those protect this
        service's capacity, not the platform's wallet. ``estimated_cost`` is
        ignored for such a job.

        This path takes no counter decision of its own — it just reports what
        the atomic writers stored (its one write is the expired-holder prune in
        ``_read_concurrent``, which removes only entries whose deadline has
        already passed). It remains advisory: a caller does check-then-record,
        so concurrent callers can both pass a check taken at the limit. With
        the counters now atomic that overshoot is bounded by in-flight
        concurrency instead of being unbounded, but closing it entirely needs
        an atomic check-and-reserve, which changes the caller contract and is
        deliberately out of scope here.

        When the quota store is unreachable this does NOT unconditionally
        admit: see the degraded-mode block at the top of this module.
        """
        if self._redis is None:
            return QuotaCheckResult(allowed=True)

        try:
            daily_gens, daily_spend, concurrent = await self._read_counters(tenant_id)
        except _REDIS_FAILURE as exc:
            # A dropped connection is the common case; one retry absorbs it
            # without any policy consequence.
            await asyncio.sleep(_DEGRADED_RETRY_DELAY_S)
            try:
                daily_gens, daily_spend, concurrent = await self._read_counters(tenant_id)
            except _REDIS_FAILURE as retry_exc:
                del exc
                self._enter_degraded("check_quota", retry_exc)
                if tenant_funded:
                    # No platform money at risk: never charge (or be refused by)
                    # the degraded SPEND budget for a tenant-key job.
                    record_quota_degraded(phase="check", outcome="admitted")
                    return QuotaCheckResult(allowed=True)
                return self._degraded_decision(tenant_id, estimated_cost)

        self._exit_degraded()

        remaining_gens = self._max_daily_generations - daily_gens
        remaining_spend = self._max_daily_spend - daily_spend
        remaining_concurrent = self._max_concurrent_jobs - concurrent

        if daily_gens >= self._max_daily_generations:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Daily generation limit exceeded ({self._max_daily_generations})",
                remaining_generations=0,
                remaining_spend=remaining_spend,
                remaining_concurrent=remaining_concurrent,
            )

        if not tenant_funded and daily_spend + estimated_cost > self._max_daily_spend:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Daily spend limit exceeded (${self._max_daily_spend:.2f})",
                remaining_generations=remaining_gens,
                remaining_spend=0.0,
                remaining_concurrent=remaining_concurrent,
            )

        if concurrent >= self._max_concurrent_jobs:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Concurrent job limit exceeded ({self._max_concurrent_jobs})",
                remaining_generations=remaining_gens,
                remaining_spend=remaining_spend,
                remaining_concurrent=0,
            )

        return QuotaCheckResult(
            allowed=True,
            remaining_generations=remaining_gens,
            remaining_spend=remaining_spend,
            remaining_concurrent=remaining_concurrent,
        )

    async def record_generation_start(self, tenant_id: str, job_id: str) -> None:
        """Atomically increment the daily generation count and take a concurrency slot.

        Both counters move inside one server-side script, so concurrent callers
        across both replicas each get their own increment instead of racing to
        overwrite one another.

        ``job_id`` NAMES the slot. It is required, not optional: an anonymous
        slot cannot be released idempotently, and every caller already has the
        id in scope. The slot carries its own expiry deadline, so a caller that
        dies before ``record_generation_complete`` releases it at that deadline
        rather than holding it forever — see the module comment above the Lua.
        """
        if self._redis is None:
            return
        now = time.time()
        try:
            result = await self._redis.eval_script(
                _RECORD_START_LUA,
                keys=[self._daily_key(tenant_id), self._concurrent_key(tenant_id)],
                args=[
                    _DAILY_TTL_SECONDS,
                    _CONCURRENT_KEY_TTL_SECONDS,
                    job_id,
                    now,
                    now + _CONCURRENT_TTL_SECONDS,
                ],
            )
            logger.debug("Quota start recorded for %s: %s", tenant_id, result)
            self._exit_degraded()
        except _REDIS_FAILURE as exc:
            # Swallowed on purpose — a counter write must not fail a generation
            # that is already going ahead. But the DROP is the whole problem:
            # this generation is now uncounted against the tenant's daily cap,
            # so it is recorded at ERROR + counter rather than left silent, and
            # it opens the degraded window so ``check_quota`` starts bounding
            # admissions instead of waiting to discover Redis is down itself.
            self._enter_degraded("record_generation_start", exc)
            record_quota_degraded(phase="start", outcome="dropped")

    async def record_generation_complete(
        self, tenant_id: str, job_id: str, cost: float = 0.0
    ) -> None:
        """Atomically release ``job_id``'s concurrency slot and accumulate spend.

        The spend accumulation is the money-critical half: a lost update here
        is spend the tenant incurred but is never charged or capped for.

        Unlike the previous implementation this also records spend when the
        daily key is absent (it used to silently drop the cost if the key had
        expired or the start was never recorded).

        Releasing is IDEMPOTENT in the slot: the release names ``job_id``, so a
        second call for the same job (the reaper re-dispatch path, a retried
        teardown) frees nothing extra. The spend half is NOT idempotent and is
        not meant to be — a re-run that really spent again really is more spend.
        """
        if self._redis is None:
            return
        now = time.time()
        try:
            result = await self._redis.eval_script(
                _RECORD_COMPLETE_LUA,
                keys=[self._daily_key(tenant_id), self._concurrent_key(tenant_id)],
                args=[
                    _DAILY_TTL_SECONDS,
                    _CONCURRENT_KEY_TTL_SECONDS,
                    job_id,
                    now,
                    "",
                    cost,
                ],
            )
            logger.debug("Quota completion recorded for %s: %s", tenant_id, result)
            self._exit_degraded()
        except _REDIS_FAILURE as exc:
            # The money-critical drop: ``cost`` dollars were genuinely spent and
            # will never be counted against this tenant's daily cap, so the
            # counter under-reports for the rest of the day. ERROR + a counter
            # labelled with the phase makes the shortfall reconcilable against
            # the cost bus afterwards; ``_enter_degraded`` makes the next
            # admission bounded rather than blind.
            self._enter_degraded("record_generation_complete", exc)
            record_quota_degraded(phase="complete", outcome="dropped")
            logger.error(
                "Quota completion DROPPED for tenant %s: $%.4f of real spend is "
                "not counted against the daily cap",
                tenant_id,
                cost,
            )

    async def get_quota_status(self, tenant_id: str) -> dict:
        """Get current quota usage for a tenant."""
        daily_key = self._daily_key(tenant_id)

        daily_gens = 0
        daily_spend = 0.0
        concurrent = 0

        if self._redis is not None:
            try:
                daily_data = await self._redis.get(daily_key)
                if daily_data and isinstance(daily_data, dict):
                    daily_gens = daily_data.get("generations", 0)
                    daily_spend = daily_data.get("spend", 0.0)
                concurrent = await self._read_concurrent(tenant_id)
            except _REDIS_FAILURE:
                logger.debug("Failed to read quota counters from Redis")

        return {
            "tenant_id": tenant_id,
            "daily_generations_used": daily_gens,
            "daily_generations_limit": self._max_daily_generations,
            "daily_spend_used": daily_spend,
            "daily_spend_limit": self._max_daily_spend,
            "concurrent_jobs": concurrent,
            "concurrent_jobs_limit": self._max_concurrent_jobs,
        }
