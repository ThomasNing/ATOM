# Serving ATOM behind llm-d

[llm-d](https://github.com/llm-d/llm-d) is a Kubernetes-native orchestration
layer for LLM inference. It does not run models — it decides *which* replica
each request should go to, using queue depth, KV-cache occupancy, and prefix
reuse as the signals. ATOM plugs into it as a model server.

This guide covers the **non-Kubernetes** path: three processes on one host, no
cluster required. Everything below was validated on 8×MI355X with two
GLM-5.2-MXFP4 TP4 instances.

## Nothing needs to be installed into the ATOM image

The llm-d components run as **separate containers**, not inside the model
server:

| Component | Where it runs | Image |
| --- | --- | --- |
| Envoy | own container | `docker.io/envoyproxy/envoy:distroless-v1.33.2` |
| EPP (Endpoint Picker) | own container | `ghcr.io/llm-d/llm-d-router-endpoint-picker:main` |
| ATOM | own container | `rocm/atom-dev` — **unmodified** |

ATOM's side of the contract is one HTTP endpoint, `/metrics`, which is built in
and pulls in no new dependency: the Prometheus exposition format is emitted
directly, so `prometheus_client` is not required and `pyproject.toml` and the
Dockerfile are untouched.

The stock endpoint-picker image is used as published — no patched build.

## Architecture

```
                     ┌──────────────────────────┐
   client ──────────►│  Envoy            :8081  │
                     └────────────┬─────────────┘
                                  │ ext-proc gRPC
                     ┌────────────▼─────────────┐
                     │  EPP              :9002  │  scrapes /metrics every 50ms
                     └────────────┬─────────────┘
                                  │ "send it to 127.0.0.1:8001"
             ┌────────────────────┴────────────────────┐
             ▼                                         ▼
   ┌───────────────────┐                     ┌───────────────────┐
   │ ATOM  GPU 0-3     │                     │ ATOM  GPU 4-7     │
   │ TP4        :8000  │                     │ TP4        :8001  │
   └───────────────────┘                     └───────────────────┘
```

EPP never carries traffic. It reads `/metrics` in the background, and when a
request arrives it returns a target address to Envoy, which does the
forwarding.

## Metrics ATOM exposes

`GET /metrics` returns the four signals llm-d's scorers consume. Names carry an
`atom:` prefix, the same convention SGLang (`sglang:`) and TensorRT-LLM
(`trtllm_`) use.

| Metric | Type | Consumed by |
| --- | --- | --- |
| `atom:num_requests_waiting` | gauge | `queue-scorer`, `load-aware-scorer` |
| `atom:num_requests_running` | gauge | `running-requests-size-scorer` |
| `atom:kv_cache_usage_perc` | gauge (0–1) | `kv-cache-utilization-scorer` |
| `atom:cache_config_info{block_size,num_gpu_blocks}` | gauge | `prefix-cache-scorer` (auto-tune) |

```console
$ curl -s http://127.0.0.1:8000/metrics
# HELP atom:num_requests_running Number of requests currently running on GPU.
# TYPE atom:num_requests_running gauge
atom:num_requests_running{model_name="GLM-5.2-MXFP4"} 6.0
...
atom:cache_config_info{block_size="16",num_gpu_blocks="138302",model_name="GLM-5.2-MXFP4"} 1.0
```

Scrapes cost nothing on the engine: the engine cores publish these counters
onto the output channel they already use whenever the counters change, and
`/metrics` serves the last published values out of memory.

## 1. Start the ATOM instances

Two TP4 servers, one per half of the box. `--served-model-name` is what clients
must send in the request body.

```bash
MODEL=/mnt/models/GLM-5.2-MXFP4
QUANT='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}'

start_atom() {  # $1=gpus  $2=port  $3=log
  AITER_LOG_LEVEL=WARNING \
  AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  AITER_USE_FLYDSL_MOE_SORTING=1 \
  HIP_VISIBLE_DEVICES="$1" \
  nohup python -m atom.entrypoints.openai_server \
    --model "$MODEL" --served-model-name GLM-5.2-MXFP4 \
    --server-port "$2" -tp 4 \
    --kv_cache_dtype fp8 --gpu-memory-utilization 0.85 \
    --enable_prefix_caching \
    --online_quant_config "$QUANT" > "$3" 2>&1 &
}

start_atom 0,1,2,3 8000 /tmp/atom_a.log
start_atom 4,5,6,7 8001 /tmp/atom_b.log

for p in 8000 8001; do
  until curl -sf "http://127.0.0.1:$p/health" >/dev/null; do sleep 5; done
done
```

Confirm the GPUs are actually loaded, not just that HTTP answers:

```bash
rocm-smi --showmemuse | grep VRAM     # expect ~85% on all eight
```

## 2. Configure the EPP

```bash
mkdir -p /etc/epp
```

`/etc/epp/endpoints.yaml` — the pool. Addresses must be literal IPv4; the
file-discovery plugin does not resolve hostnames. With `watchFile: true` the
EPP reloads this on every atomic rewrite, so replicas can come and go without a
restart.

```yaml
endpoints:
  - name: atom-0
    address: 127.0.0.1
    port: "8000"
    labels:
      model: GLM-5.2-MXFP4
  - name: atom-1
    address: 127.0.0.1
    port: "8001"
    labels:
      model: GLM-5.2-MXFP4
```

`/etc/epp/config.yaml` — plugins and scoring. The `engineConfigs` block is what
teaches a stock EPP to read ATOM's metric names; user-defined engines are
merged ahead of the built-in list, so no patched image is needed.

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig

plugins:
  - name: file-discovery
    type: file-discovery
    parameters:
      path: /etc/epp/endpoints.yaml
      watchFile: true

  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: prefix-cache-scorer
  - type: no-hit-lru-scorer

  - name: metrics-source
    type: metrics-data-source
  - name: metrics-extractor
    type: core-metrics-extractor
    parameters:
      defaultEngine: atom
      engineConfigs:
        - name: atom
          queuedRequestsSpec: "atom:num_requests_waiting"
          runningRequestsSpec: "atom:num_requests_running"
          kvUsageSpec: "atom:kv_cache_usage_perc"
          cacheInfoSpec: "atom:cache_config_info"

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
        weight: 2
      - pluginRef: kv-cache-utilization-scorer
        weight: 2
      - pluginRef: prefix-cache-scorer
        weight: 3
      - pluginRef: no-hit-lru-scorer
        weight: 2

dataLayer:
  injectDefaults: false
  discovery:
    pluginRef: file-discovery
  sources:
    - pluginRef: metrics-source
      extractors:
        - pluginRef: metrics-extractor
```

Start it:

```bash
docker run -d --name llmd-epp --network host \
  -v /etc/epp:/etc/epp:ro \
  ghcr.io/llm-d/llm-d-router-endpoint-picker:main \
  --config-file=/etc/epp/config.yaml \
  --pool-name=file-discovery --pool-namespace=default \
  --grpc-port=9002 --grpc-health-port=9003 --metrics-port=9090 \
  --secure-serving=false --v=2
```

`atom` should appear alongside the built-in engines:

```bash
docker logs llmd-epp 2>&1 | grep -o '"engine":"[a-z-]*"' | sort -u
# "engine":"atom"  "engine":"sglang"  "engine":"vllm"  ...
```

## 3. Start Envoy

Take `envoy.yaml` from the llm-d repo
([`guides/no-kubernetes-deployment/router/envoy/envoy.yaml`](https://github.com/llm-d/llm-d/blob/main/guides/no-kubernetes-deployment/router/envoy/envoy.yaml))
unmodified — it listens on `:8081`, calls ext-proc on `127.0.0.1:9002`, and
routes by the `x-gateway-destination-endpoint` header EPP sets.

```bash
mkdir -p /etc/envoy && cp envoy.yaml /etc/envoy/

docker run -d --name llmd-envoy --network host \
  -v /etc/envoy/envoy.yaml:/etc/envoy/envoy.yaml:ro \
  docker.io/envoyproxy/envoy:distroless-v1.33.2 \
  --service-node envoy-proxy --log-level warn \
  --concurrency 8 --drain-strategy immediate --drain-time-s 60 \
  -c /etc/envoy/envoy.yaml
```

## 4. Verify

```bash
curl -s http://127.0.0.1:19000/ready                 # Envoy: LIVE
curl -s http://127.0.0.1:9090/metrics | head         # EPP's own metrics
curl -s http://127.0.0.1:8000/metrics | grep '^atom:'

curl http://127.0.0.1:8081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2-MXFP4","prompt":"What is tensor parallelism?","max_tokens":60}'
```

Allow a generous client timeout on the first request — it pays for CUDA-graph
warmup (TTFT ~8 s cold, milliseconds after).

## Accuracy and routing

GSM8K driven through Envoy, so every request traverses the full path. Note
`tokenizer=` is passed separately: `model=` goes into the request body, where
the server matches it against `--served-model-name`, while lm_eval would
otherwise try to resolve that name as a HuggingFace repo.

```bash
lm_eval --model local-completions \
  --model_args "model=GLM-5.2-MXFP4,tokenizer=/mnt/models/GLM-5.2-MXFP4,\
base_url=http://127.0.0.1:8081/v1/completions,num_concurrent=32,\
max_retries=3,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5
```

Measured on 8×MI355X, two GLM-5.2-MXFP4 TP4 instances, full 1319 samples:

| Filter | n-shot | exact_match | Stderr |
| --- | ---: | ---: | ---: |
| flexible-extract | 5 | 0.9287 | ±0.0071 |
| strict-match | 5 | 0.9280 | ±0.0071 |

Matching the direct-serving baseline in [GLM-5.md](GLM-5.md), so routing
through llm-d costs no accuracy. The 1319 requests split 651 / 668 across the
two instances, with no failed requests.

## Notes

- **Scoring is tunable without code changes.** Weights live in
  `schedulingProfiles`; the defaults above favour cache affinity
  (`prefix-cache-scorer` at 3) over pure load spreading.
- **`max-score-picker` is deterministic**, so within one 50 ms metrics refresh
  every arriving request scores the same endpoint the same way. Under high QPS
  `weighted-random-picker` spreads those bursts instead.
- **The atomesh entrypoint has no `/metrics`.** With
  `USE_ATOMESH_ENTRYPOINTS`, HTTP is served by the Rust layer and a scrape
  returns 404. Use the Python OpenAI server for llm-d deployments.
- **P/D disaggregation** needs more than this guide: a `pd-sidecar` container
  in front of each decode server and `llm-d.ai/role: prefill|decode` labels on
  the endpoints. See [pd_disaggregation_guide.md](pd_disaggregation_guide.md)
  for the ATOM side.
- **On Kubernetes**, only endpoint discovery changes — an `InferencePool` CRD
  replaces `endpoints.yaml`. Scoring, metrics, and the Envoy path are identical.
