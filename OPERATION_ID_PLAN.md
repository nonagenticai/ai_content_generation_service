# Operation ID Plan: ai_content_generation_service

Canonical registry of `operation_id` values for every route in
`ai_content_generation_service`. These IDs feed the org-wide n8n custom-node
generator and the `ultrareview` cross-service catalogue, so they must be:

- **stable** -- never rename without coordinating downstream,
- **`verb_noun` snake_case** -- matches the org standard (nlp_service,
  worker_management, kag, analysis_service, video_compiler_service,
  video_analyzer_service),
- **globally unique** -- disambiguated against the rest of the fleet's verbs.

This service hosts two domains:

- `content-videos` -- multi-provider AI video generation (Runway, Kling,
  Veo, Sora, Seedance, Wan2.2, stub).
- `concept-images` -- AI-generated concept images.

Plus a `health` surface.

Generic verbs (`list_*`, `cancel_*`, `download_*`, `get_*_status`,
`get_*_result`) are prefixed with the domain (`video_` / `concept_image_`)
to keep them out of the fleet-wide collision pool.

## Files to edit

- `src/asgi.py` (root endpoint, prometheus metrics endpoint)
- `src/api/health.py`
- `src/api/routes/generation.py`
- `src/api/routes/jobs.py`
- `src/api/routes/batches.py`
- `src/api/routes/preview.py`
- `src/api/routes/providers.py`
- `src/api/routes/usage.py`
- `src/api/content_routes.py`

## Mapping Tables

### `src/asgi.py` (tag: `health`)

| Method | Function           | operation_id          | Rationale                                                |
|--------|--------------------|-----------------------|----------------------------------------------------------|
| GET    | `root`             | `service_root`        | Service banner; matches the fleet standard.              |
| GET    | `metrics_endpoint` | `prometheus_metrics`  | Public metrics scrape; verb_noun matches Prometheus role. |

### `src/api/health.py` (tag: `health`)

| Method | Function       | operation_id   | Rationale                                                  |
|--------|----------------|----------------|------------------------------------------------------------|
| GET    | `health_check` | `health_check` | Primary health probe; matches the fleet standard verb.     |
| GET    | `health_live`  | `health_live`  | Liveness sub-probe; one-to-one with path.                  |
| GET    | `health_ready` | `health_ready` | Readiness sub-probe; one-to-one with path.                 |

### `src/api/routes/generation.py` (tag: `content-videos`, prefix `/api/content/videos`)

| Method | Function          | operation_id                    | Rationale                                                                 |
|--------|-------------------|---------------------------------|---------------------------------------------------------------------------|
| POST   | `generate_video`  | `start_video_generation`        | `generate_video` is too generic; fleet uses `start_<domain>_<verb>`.       |
| POST   | `generate_batch`  | `start_video_generation_batch`  | Batch sibling of the above; domain-prefixed to avoid colliding with image. |

### `src/api/routes/jobs.py` (tag: `content-videos`, prefix `/api/content/videos`)

| Method | Function          | operation_id              | Rationale                                                                                          |
|--------|-------------------|---------------------------|----------------------------------------------------------------------------------------------------|
| POST   | `cancel_job`      | `cancel_video_job`        | Generic `cancel_job` collides with other services; prefix with the domain (`video_`).               |
| GET    | `list_jobs`       | `list_video_jobs`         | Same -- domain-prefix.                                                                              |
| GET    | `get_job_status`  | `get_video_job_status`    | Same -- domain-prefix.                                                                              |
| GET    | `get_job_result`  | `get_video_job_result`    | Same -- domain-prefix.                                                                              |
| GET    | `get_job_costs`   | `get_video_job_costs`     | Same -- domain-prefix.                                                                              |
| GET    | `get_job_video`   | `download_video_job_video`| Verb is `download` (binary or 307 to MinIO presigned URL); domain-prefix the noun for disambiguation. |

### `src/api/routes/batches.py` (tag: `content-videos`, prefix `/api/content/videos`)

| Method | Function            | operation_id              | Rationale                                                          |
|--------|---------------------|---------------------------|--------------------------------------------------------------------|
| GET    | `get_batch_status`  | `get_video_batch_status`  | Domain-prefix to disambiguate from concept-image batch.            |
| GET    | `get_batch_result`  | `get_video_batch_result`  | Same.                                                              |

### `src/api/routes/preview.py` (tag: `content-videos`, prefix `/api/content/videos`)

| Method | Function         | operation_id                | Rationale                                                            |
|--------|------------------|-----------------------------|----------------------------------------------------------------------|
| POST   | `preview_video`  | `preview_video_generation`  | `preview_video` already in verb_noun shape; align with `start_*_generation`. |

### `src/api/routes/providers.py` (tag: `content-videos`, prefix `/api/content/videos`)

| Method | Function                | operation_id                  | Rationale                                                                      |
|--------|-------------------------|-------------------------------|--------------------------------------------------------------------------------|
| GET    | `list_providers`        | `list_video_providers`        | Generic `list_providers` would collide across the fleet; domain-prefix.        |
| GET    | `get_provider_health`   | `get_video_provider_health`   | Same -- domain-prefix.                                                         |

### `src/api/routes/usage.py` (tag: `content-videos`, prefix `/api/content/videos`)

| Method | Function             | operation_id             | Rationale                                                            |
|--------|----------------------|--------------------------|----------------------------------------------------------------------|
| GET    | `get_usage`          | `get_video_usage`        | Domain-prefix.                                                       |
| GET    | `get_quota_status`   | `get_video_quota_status` | Domain-prefix; quota is per-tenant in the video-generation domain.   |

### `src/api/content_routes.py` (tag: `concept-images`, prefix `/api/concept-images`)

| Method | Function                            | operation_id                       | Rationale                                                                                |
|--------|-------------------------------------|------------------------------------|------------------------------------------------------------------------------------------|
| POST   | `create_concept_image`              | `create_concept_image`             | Already canonical verb_noun.                                                             |
| POST   | `create_concept_image_batch`        | `create_concept_image_batch`       | Already canonical verb_noun.                                                             |
| POST   | `cancel_concept_image_job`          | `cancel_concept_image_job`         | Domain-prefixed generic verb, mirrors `cancel_video_job`.                                |
| GET    | `get_concept_image_job`             | `get_concept_image_job`            | Already canonical verb_noun (domain-prefix already present in function name).             |
| GET    | `get_concept_image_job_result`      | `get_concept_image_job_result`     | Same.                                                                                    |
| GET    | `get_concept_image_batch`           | `get_concept_image_batch`          | Same.                                                                                    |
| GET    | `get_concept_image_batch_result`    | `get_concept_image_batch_result`   | Same.                                                                                    |
| GET    | `get_concept_image_asset`           | `get_concept_image_asset`          | Same.                                                                                    |
| GET    | `download_concept_image_asset`      | `download_concept_image_asset`     | Same; `download_*` would otherwise collide -- domain-prefix already in the function name. |

## Verification

```bash
cd ai_content_generation_service
uv run pytest tests/test_openapi.py -x
```

`tests/test_openapi.py` loads `app.openapi()` and asserts that the set of
`operationId` values equals the union of every right-hand column above.
Drift (rename, missed decorator, fallback) fails the test loudly.

## Disambiguation rationale

Concrete renames where the function name was generic enough to collide
across the fleet:

- `generate_video` -> `start_video_generation` -- aligns with the
  `start_<domain>_<verb>` shape (e.g. analysis_service's
  `start_video_analysis`).
- `generate_batch` -> `start_video_generation_batch` -- domain-prefix to
  keep it distinct from image-batch and other services' batch routes.
- `cancel_job`, `list_jobs`, `get_job_*`, `get_batch_*`, `list_providers`,
  `get_provider_health`, `get_usage`, `get_quota_status` -- all
  domain-prefixed with `video_` to keep them out of the generic `*_job` /
  `*_batch` namespaces shared with other services.
- `get_job_video` -> `download_video_job_video` -- verb is the operation
  (download), object is the video bytes; the route 307-redirects to a
  MinIO presigned URL when available.

`concept_image_*` operations already carry the `concept_image_` namespace
in their function names, so the operation_ids are 1:1 with the function
names.

## Future routes

When adding a route in this service, **always** set `operation_id="..."`
explicitly. The `generate_unique_id_function` configured in `asgi.py` is a
safety net for forgotten kwargs, never the primary identifier. New IDs go
in the table above as part of the same PR (no out-of-band updates).
