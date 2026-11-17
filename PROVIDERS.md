# AI Content Generation — Provider catalogue

Every image / video provider currently wired into `ai_content_generation_service`,
the env vars that drive each, and the single knob to flip the default once an
external account is provisioned. **Drop a key in, flip the default — no code
change.** This file is the contract the cluster ConfigMap + the SOPS-encrypted
Secret point at.

## Defaults today

The cluster runs with both stacks pinned to deterministic stubs (see
`gitops-production/clusters/strato/services/w2v/configmaps.yaml`):

```
STUB_IMAGE_PROVIDER=stub
STUB_VIDEO_PROVIDER=stub
```

When `STUB_*_PROVIDER=stub` is set, the service registers a deterministic
PNG/MP4 stub and uses it as the default for that stack, regardless of what
`GENERATION_DEFAULT_PROVIDER` / `IMAGE_GENERATION_DEFAULT_PROVIDER` say.
Unset the stub to let the per-stack default take over.

## Cutting over to an external provider

Two steps, both in gitops:

1. **Drop the key** — `sops edit secrets.yaml` (in
   `clusters/strato/services/w2v/`), add the `<PROVIDER>_API_KEY=<real-key>`
   entry under `ai-content-generation-service-secrets`. SOPS re-encrypts in
   place.
2. **Flip the default** — in `configmaps.yaml` under
   `ai-content-generation-service-config`:
   - For video: unset `STUB_VIDEO_PROVIDER` and set
     `GENERATION_DEFAULT_PROVIDER=<name>`.
   - For image: unset `STUB_IMAGE_PROVIDER` and set
     `IMAGE_GENERATION_DEFAULT_PROVIDER=<name>`.
3. Flux reconciles; the next pod restart picks up the new env. No image
   rebuild needed.

Optional: set `GENERATION_FALLBACK_CHAIN=<a>,<b>,<stub>` to keep `stub`
as a safety net while the new provider stabilises. `GENERATION_FALLBACK_ENABLED=true`
is on by default.

---

## Video providers

Defined in `src/services/providers/`. Each provider class loads its own
`<Name>Settings` from `src/config.py`; a provider is "enabled" when it has
a non-empty API key (or, for local providers, an explicit `<PREFIX>_ENABLED=true`).

| `provider=` | Env prefix | Required env (key) | Optional env | Provider class |
|---|---|---|---|---|
| `runway` | `RUNWAY_` | `RUNWAY_API_KEY` | `RUNWAY_MODEL` (def `gen-3`), `RUNWAY_TIMEOUT` | Runway ML |
| `kling` | `KLING_` | `KLING_API_KEY` | `KLING_MODEL` (def `kling-v1`), `KLING_TIMEOUT` | Kling AI |
| `veo` | `VEO_` | `VEO_API_KEY` | `VEO_MODEL` (def `veo-2`), `VEO_TIMEOUT` | Google Veo |
| `sora` | `SORA_` | `SORA_API_KEY` | `SORA_MODEL` (def `sora-1`), `SORA_TIMEOUT` | OpenAI Sora |
| `seedance` | `SEEDANCE_` | `SEEDANCE_API_KEY` | `SEEDANCE_MODEL` (def `seedance-1`), `SEEDANCE_TIMEOUT` | Seedance |
| `wan` | `WAN_` | none (local model) | `WAN_ENABLED=true` to activate, `WAN_MODEL_NAME` (def `Wan-AI/Wan2.2-TI2V-5B`), `WAN_DEVICE` (def `auto`) | Wan2.2 local (GPU host) |
| `mock` | `MOCK_PROVIDER_` | none | `MOCK_PROVIDER_ENABLED=true`, `MOCK_PROVIDER_DELAY_S` | Test fixture (sleep + return) |
| `stub` | (top-level) | none | `STUB_VIDEO_PROVIDER=stub` to activate | Deterministic ffprobe-valid MP4 |

### Video defaults

```
GENERATION_DEFAULT_PROVIDER       # current default video provider (def: runway)
GENERATION_DEFAULT_DURATION_S=5
GENERATION_DEFAULT_WIDTH=1920
GENERATION_DEFAULT_HEIGHT=1080
GENERATION_PARALLEL_REQUESTS=3
GENERATION_FALLBACK_ENABLED=true
GENERATION_FALLBACK_CHAIN=        # comma-separated, e.g. "runway,kling,stub"
GENERATION_POLL_INTERVAL_S=5
GENERATION_MAX_POLL_TIME_S=600
```

---

## Image providers

Defined in `src/services/image_providers/`.

| `provider=` | Env prefix | Required env | Optional env | Provider class |
|---|---|---|---|---|
| `openai_images` / `dalle` | `DALLE_` | `DALLE_API_KEY`, `DALLE_ENABLED=true` | `DALLE_MODEL` (def `gpt-image-1`), `DALLE_SIZE` (def `1792x1024`), `DALLE_QUALITY`, `DALLE_BACKGROUND`, `DALLE_OUTPUT_FORMAT`, `DALLE_TIMEOUT`, `DALLE_BASE_URL` | OpenAI image gen |
| `local_image` | `LOCAL_IMAGE_` | `LOCAL_IMAGE_ENABLED=true` | `LOCAL_IMAGE_BACKEND_URL` (def `http://image-gen.test:9204`), `LOCAL_IMAGE_TIMEOUT` | mflux/FLUX backend (launchd unit on m1/m4) |
| `stub` | (top-level) | none | `STUB_IMAGE_PROVIDER=stub` to activate | Deterministic PNG |

### Image defaults

```
IMAGE_GENERATION_DEFAULT_PROVIDER  # def: openai_images
IMAGE_GENERATION_PARALLEL_REQUESTS=4
IMAGE_GENERATION_POLL_INTERVAL_S=2
IMAGE_GENERATION_MAX_POLL_TIME_S=180
```

---

## Operational knobs (provider-stack-wide)

```
HEALTH_MONITOR_ENABLED=true            # disable a provider after N consecutive failures
HEALTH_MONITOR_MAX_CONSECUTIVE_FAILURES=5
HEALTH_MONITOR_COOLDOWN_SECONDS=300

QUOTA_ENABLED=true                     # per-tenant generation + spend caps
QUOTA_DEFAULT_MAX_DAILY_GENERATIONS=100
QUOTA_DEFAULT_MAX_DAILY_SPEND=50.0
QUOTA_DEFAULT_MAX_CONCURRENT_JOBS=10

RATE_LIMIT_ENABLED=false
NORMALIZATION_ENABLED=false            # FFmpeg post-process output to a target resolution/codec

METERING_REDIS_URL=                    # cost-event stream → metering_service (empty = silent)
METERING_STREAM=metering:costs
```

---

## Smoke check after a cutover

```
# 1. New deployment came up and the picked provider initialised.
kubectl -n w2v logs deploy/ai-content-generation-service | grep -iE "provider|init|registered"

# 2. Generate a single concept video / image via the API.
curl -sS http://ai-content-generation-service.w2v:6310/health
# expects {"status":"healthy", ...}

# 3. Drive a generic blueprint with ai_image_gen / ai_video_gen enabled
# from w2v_orchestrator and confirm the run reaches completed.
```

If a provider startup throws, the service still boots — failing providers are
logged at WARNING and excluded from the active registry. Confirm via
`kubectl logs … | grep -iE "warning|error"`.
