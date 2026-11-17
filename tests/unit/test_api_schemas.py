"""Unit tests for API request/response schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    AsyncJobResponse,
    BatchJobResponse,
    BatchResultResponse,
    BatchStatusResponse,
    ClipCostItem,
    ClipStatusItem,
    CostReport,
    GenerateBatchRequest,
    GenerateRequest,
    GenerationResultResponse,
    JobStatusResponse,
    ProviderInfoResponse,
    ProvidersListResponse,
)


def test_generate_request_minimal():
    req = GenerateRequest(prompt="a cat video")
    assert req.prompt == "a cat video"
    assert req.provider is None
    assert req.budget_limit == 5.0


def test_generate_request_full():
    req = GenerateRequest(
        prompt="test",
        provider="runway",
        negative_prompt="bad quality",
        image_url="https://example.com/img.png",
        duration_s=10.0,
        width=1280,
        height=720,
        style="vivid",
        seed=42,
        extra_params={"foo": "bar"},
        budget_limit=2.5,
    )
    assert req.provider == "runway"
    assert req.budget_limit == 2.5
    assert req.seed == 42


def test_generate_batch_request():
    req = GenerateBatchRequest(
        items=[GenerateRequest(prompt="clip 1"), GenerateRequest(prompt="clip 2")],
        provider="kling",
        budget_limit=3.0,
    )
    assert len(req.items) == 2
    assert req.budget_limit == 3.0
    assert req.provider == "kling"


class TestBudgetLimitBounds:
    """``budget_limit`` is the only per-request spend cap — it must be finite and positive.

    Pydantic v2 defaults to ``allow_inf_nan=True`` for ``float``, so an
    unconstrained field parses ``1e400`` to ``inf``. ``check_budget`` then does
    ``inf - spent == inf`` and every clip is affordable: the cap is disabled.
    """

    @pytest.mark.parametrize("model", [GenerateRequest, GenerateBatchRequest])
    @pytest.mark.parametrize("bad", [1e400, -1e400, float("nan"), 0, 0.0, -1.0, -0.01])
    def test_non_finite_or_non_positive_budget_is_rejected(self, model, bad):
        kwargs = {"prompt": "x"} if model is GenerateRequest else {"items": [{"prompt": "x"}]}
        with pytest.raises(ValidationError):
            model(budget_limit=bad, **kwargs)

    @pytest.mark.parametrize("model", [GenerateRequest, GenerateBatchRequest])
    def test_absurd_budget_above_the_reachable_worst_case_is_rejected(self, model):
        kwargs = {"prompt": "x"} if model is GenerateRequest else {"items": [{"prompt": "x"}]}
        with pytest.raises(ValidationError):
            model(budget_limit=1_000_000.0, **kwargs)

    @pytest.mark.parametrize("model", [GenerateRequest, GenerateBatchRequest])
    def test_realistic_budgets_still_accepted(self, model):
        kwargs = {"prompt": "x"} if model is GenerateRequest else {"items": [{"prompt": "x"}]}
        # $60.00 = the priciest reachable batch (10 sora clips x 300s @ $0.02/s).
        assert model(budget_limit=60.0, **kwargs).budget_limit == 60.0
        assert model(budget_limit=0.01, **kwargs).budget_limit == 0.01


class TestBatchItemBounds:
    """One batch item = one DB row + one leased asyncio task + one PAID generation."""

    def test_batch_over_the_cap_is_rejected(self):
        with pytest.raises(ValidationError):
            GenerateBatchRequest(items=[GenerateRequest(prompt="x") for _ in range(11)])

    def test_empty_batch_is_rejected(self):
        with pytest.raises(ValidationError):
            GenerateBatchRequest(items=[])

    def test_batch_at_the_cap_is_accepted(self):
        req = GenerateBatchRequest(items=[GenerateRequest(prompt="x") for _ in range(10)])
        assert len(req.items) == 10


def test_async_job_response():
    resp = AsyncJobResponse(job_id="j1", status="pending", poll_url="/api/content/videos/jobs/j1")
    assert resp.job_id == "j1"
    assert resp.status == "pending"


def test_batch_job_response():
    resp = BatchJobResponse(batch_id="b1", job_ids=["j1", "j2"])
    assert resp.batch_id == "b1"
    assert resp.status == "pending"
    assert resp.cost_report is None


def test_job_status_response():
    resp = JobStatusResponse(job_id="j1", status="completed", provider="runway")
    assert resp.provider == "runway"


def test_generation_result_response():
    resp = GenerationResultResponse(
        job_id="j1",
        status="completed",
        provider="runway",
        cost=0.05,
        generation_time_seconds=12.3,
    )
    assert resp.cost == 0.05
    assert resp.generation_time_seconds == 12.3


def test_cost_report():
    report = CostReport(
        total_cost=0.15,
        budget_limit=5.0,
        clips_generated=3,
        clips_skipped=1,
        clips_failed=0,
        per_clip_costs=[
            ClipCostItem(
                job_id="j1", provider="runway", duration_s=5.0, cost=0.05, status="completed"
            ),
        ],
    )
    assert report.total_cost == 0.15
    assert len(report.per_clip_costs) == 1


def test_cost_report_defaults():
    report = CostReport()
    assert report.total_cost == 0.0
    assert report.clips_generated == 0
    assert report.per_clip_costs == []


def test_provider_info_response():
    resp = ProviderInfoResponse(
        name="runway",
        available=True,
        models=["gen-3"],
        max_duration_s=10.0,
        supports_image_to_video=True,
    )
    assert resp.supports_image_to_video is True


def test_providers_list_response():
    resp = ProvidersListResponse(
        providers=[ProviderInfoResponse(name="mock", available=True)],
        default_provider="mock",
    )
    assert len(resp.providers) == 1


def test_clip_status_item():
    item = ClipStatusItem(job_id="j1", status="completed", provider="kling", cost=0.03)
    assert item.cost == 0.03


def test_batch_status_response():
    resp = BatchStatusResponse(
        batch_id="b1",
        status="processing",
        clips_submitted=3,
        clips_completed=1,
        clips_failed=0,
        clips_skipped=0,
        total_cost=0.05,
        budget_limit=5.0,
    )
    assert resp.status == "processing"


def test_batch_result_response():
    resp = BatchResultResponse(
        batch_id="b1",
        status="completed",
        job_ids=["j1", "j2"],
        clips_completed=2,
        clips_failed=0,
        total_cost=0.10,
    )
    assert resp.clips_completed == 2


def test_generate_request_seed_valid():
    req = GenerateRequest(prompt="test", seed=0)
    assert req.seed == 0

    req = GenerateRequest(prompt="test", seed=4294967295)
    assert req.seed == 4294967295


def test_generate_request_seed_too_large():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", seed=4294967296)


def test_generate_request_seed_negative():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", seed=-1)


def test_generate_request_extra_params_valid():
    req = GenerateRequest(prompt="test", extra_params={"key": "value"})
    assert req.extra_params == {"key": "value"}


def test_generate_request_extra_params_too_many_keys():
    params = {f"key_{i}": f"value_{i}" for i in range(21)}
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", extra_params=params)


def test_generate_request_extra_params_too_large():
    params = {"key": "x" * 11000}
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", extra_params=params)
