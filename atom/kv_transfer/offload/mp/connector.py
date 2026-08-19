# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.2 adapter for LMCache's standalone multiprocess server.

ATOM exposes GLM-5.2's paged cache as one MLA latent and one DSA index cache
per layer. LMCache MP can attach several physical kernel groups to the same
engine block-id space, so this adapter registers the latent and index tensors
as two groups while reusing ATOM's existing dense offload scheduler policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from atom.kv_transfer.disaggregation.base import KVConnectorBase
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadCompletionId,
    LoadOperationId,
    SaveCompletionId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta

logger = logging.getLogger("atom")

_MP_LAYOUT_VERSION = 1
_GLM52_LATENT_WIDTH = 576
_GLM52_INDEX_WIDTH = 144


def _extra_config(config: Any) -> dict[str, Any]:
    kvc = getattr(config, "kv_transfer_config", {}) or {}
    extra = kvc.get("kv_connector_extra_config", {}) or {}
    if not isinstance(extra, dict):
        raise TypeError("kv_connector_extra_config must be a dictionary")
    return extra


def _validate_glm52_config(config: Any) -> tuple[int, int]:
    hf_config = getattr(config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    if getattr(text_config, "model_type", None) != "glm_moe_dsa":
        raise NotImplementedError(
            "lmcache_mp currently supports only GLM-5.2 (model_type " "'glm_moe_dsa')"
        )

    tp_size = offcfg._strict_integer(
        "tensor_parallel_size",
        getattr(config, "tensor_parallel_size", 1) or 1,
        minimum=1,
    )
    pp_size = offcfg._strict_integer(
        "pipeline_parallel_size",
        getattr(config, "pipeline_parallel_size", 1) or 1,
        minimum=1,
    )
    dcp_size = offcfg._strict_integer(
        "decode_context_parallel_size",
        getattr(config, "decode_context_parallel_size", 1) or 1,
        minimum=1,
    )
    pcp_size = offcfg._strict_integer(
        "prefill_context_parallel_size",
        getattr(config, "prefill_context_parallel_size", 1) or 1,
        minimum=1,
    )
    parallel_config = getattr(config, "parallel_config", None)
    dp_size = offcfg._strict_integer(
        "data_parallel_size",
        getattr(
            parallel_config,
            "data_parallel_size",
            getattr(config, "data_parallel_size", 1),
        )
        or 1,
        minimum=1,
    )
    if pp_size != 1:
        raise NotImplementedError("GLM-5.2 lmcache_mp does not support PP yet")
    if dcp_size != 1:
        raise NotImplementedError("GLM-5.2 lmcache_mp does not support DCP yet")
    if pcp_size != 1:
        raise NotImplementedError("GLM-5.2 lmcache_mp does not support PCP yet")
    if dp_size != 1 or bool(getattr(config, "enable_dp_attention", False)):
        raise NotImplementedError(
            "GLM-5.2 lmcache_mp currently supports TP-only deployments"
        )

    extra = _extra_config(config)
    configured_mode = extra.get("lmcache.mp.mp_transfer_mode")
    if configured_mode is None:
        configured_mode = os.environ.get("LMCACHE_MP_TRANSFER_MODE", "auto")
    transfer_mode = str(configured_mode).strip().lower()
    if transfer_mode not in ("auto", "lmcache_driven", "engine_driven"):
        raise ValueError(
            "LMCache MP transfer mode must be 'auto', 'lmcache_driven', or "
            f"'engine_driven', got {configured_mode!r}"
        )
    if transfer_mode == "engine_driven":
        raise NotImplementedError(
            "GLM-5.2 lmcache_mp requires LMCache's lmcache_driven transfer "
            "path because engine_driven does not support multiple physical "
            "cache groups"
        )
    return tp_size, pp_size


def _server_urls(config: Any) -> list[str]:
    extra = _extra_config(config)
    configured = extra.get("lmcache.mp.server_urls")
    if configured is not None:
        if isinstance(configured, (list, tuple)):
            urls = [str(value).strip() for value in configured if str(value).strip()]
        else:
            urls = [
                value.strip() for value in str(configured).split(",") if value.strip()
            ]
    else:
        host = str(extra.get("lmcache.mp.host", "tcp://localhost")).strip()
        if not host:
            raise ValueError("lmcache.mp.host must be non-empty")
        port = offcfg._strict_integer(
            "lmcache.mp.port",
            extra.get("lmcache.mp.port", 5555),
            minimum=1,
        )
        if not 1 <= port <= 65535:
            raise ValueError("lmcache.mp.port must be in [1, 65535]")
        urls = [f"{host}:{port}"]
    urls = [url if "://" in url else f"tcp://{url}" for url in urls]
    if len(urls) != 1:
        raise NotImplementedError(
            "GLM-5.2 lmcache_mp currently supports exactly one LMCache server"
        )
    return urls


def _model_namespace(config: Any) -> str:
    """Build a stable namespace that cannot alias vLLM or legacy ATOM bytes."""

    hf_config = getattr(config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    model_name = str(
        getattr(config, "model_tag", None) or getattr(config, "model", "atom-model")
    )
    fields = (
        "model_type",
        "num_hidden_layers",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "index_head_dim",
        "index_topk",
        "index_topk_freq",
        "indexer_types",
    )
    document = {
        "schema": "atom-glm52-lmcache-mp",
        "version": _MP_LAYOUT_VERSION,
        "model": model_name,
        "block_size": offcfg._strict_integer(
            "GLM-5.2 LMCache MP block size",
            config.kv_cache_block_size,
            minimum=1,
        ),
        "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "auto")),
        "index_cache_dtype": str(
            getattr(
                config,
                "index_cache_dtype",
                getattr(config, "kv_cache_dtype", "auto"),
            )
        ),
        "tp_size": offcfg._strict_integer(
            "GLM-5.2 LMCache MP TP size",
            getattr(config, "tensor_parallel_size", 1) or 1,
            minimum=1,
        ),
        "hf": offcfg._stable_config_value(
            {name: getattr(text_config, name, None) for name in fields}
        ),
        "speculative_config": offcfg._stable_config_value(
            getattr(config, "speculative_config", None)
        ),
    }
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.blake2b(
        encoded,
        digest_size=16,
        person=b"ATOM-GLM52-MP1",
    ).hexdigest()
    return f"{model_name}::atom-glm52-mp-v{_MP_LAYOUT_VERSION}-{digest}"


def _parallel_strategy(config: Any, worker_id: int) -> Any:
    from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy

    tp_size, pp_size = _validate_glm52_config(config)
    if worker_id < 0 or worker_id >= tp_size:
        raise ValueError(
            f"GLM-5.2 LMCache MP worker rank {worker_id} is outside " f"[0, {tp_size})"
        )
    return ParallelStrategy(
        # LMCache's pure-MLA optimization collapses the KV world to one rank:
        # only TP rank 0 stores, while lookup acquires a single set of read
        # locks.  GLM-5.2 is not pure MLA because every TP rank also owns the
        # IndexShare/DSA cache registered below.  Collapsing the world lets all
        # TP workers retrieve the same rank-0 object with only one lock; one
        # worker can consequently miss while the scheduler reports a hit.
        # Keep every TP slice independently keyed and transferred instead.
        mla_only=False,
        vllm_world_size=tp_size * pp_size,
        vllm_worker_id=worker_id,
        tp_size=tp_size,
        pp_size=pp_size,
        n_servers=1,
    )


def _make_scheduler_adapter(config: Any) -> Any:
    import zmq
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

    return LMCacheMPSchedulerAdapter(
        server_urls=_server_urls(config),
        context=zmq.Context.instance(),
        model_name=_model_namespace(config),
        vllm_block_size=int(config.kv_cache_block_size),
        parallel_strategy=_parallel_strategy(config, 0),
        extra_config=_extra_config(config),
    )


def _make_worker_adapter(config: Any, rank: int) -> Any:
    import zmq
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPWorkerAdapter,
    )

    return LMCacheMPWorkerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(config),
        vllm_block_size=int(config.kv_cache_block_size),
        parallel_strategy=_parallel_strategy(config, rank),
        extra_config=_extra_config(config),
    )


@dataclass(frozen=True)
class _GLM52CacheViews:
    tensors: dict[str, torch.Tensor]
    latent_indices: tuple[int, ...]
    index_indices: tuple[int, ...]
    bytes_per_block: int


def _reshape_block_cache(
    tensor: torch.Tensor,
    *,
    name: str,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    if tensor.numel() == 0:
        raise ValueError(f"GLM-5.2 lmcache_mp cache {name} is empty")
    if not tensor.is_contiguous():
        raise ValueError(f"GLM-5.2 lmcache_mp cache {name} must be contiguous")
    rows = num_blocks * block_size
    if tensor.numel() % rows:
        raise ValueError(
            f"GLM-5.2 lmcache_mp cache {name} has {tensor.numel()} elements, "
            f"not divisible by num_blocks*block_size={rows}"
        )
    return tensor.reshape(num_blocks, block_size, tensor.numel() // rows)


def _build_glm52_cache_views(
    kv_caches: dict[str, Any],
    *,
    num_blocks: int,
    block_size: int,
    latent_width: int = _GLM52_LATENT_WIDTH,
    index_width: int = _GLM52_INDEX_WIDTH,
) -> _GLM52CacheViews:
    """Flatten ATOM ``KVCacheTensor`` objects into LMCache MP tensor groups."""

    if num_blocks <= 0 or block_size <= 0:
        raise ValueError("num_blocks and block_size must be positive")

    latent: list[tuple[str, torch.Tensor]] = []
    indexes: list[tuple[str, torch.Tensor]] = []
    latent_layers: set[str] = set()
    index_layers: set[str] = set()
    for layer_name, cache in kv_caches.items():
        latent_tensor = getattr(cache, "k_cache", None)
        if isinstance(latent_tensor, torch.Tensor) and latent_tensor.numel() > 0:
            latent_view = _reshape_block_cache(
                latent_tensor,
                name=f"{layer_name}.latent",
                num_blocks=num_blocks,
                block_size=block_size,
            )
            if latent_view.shape[2] != latent_width:
                raise ValueError(
                    f"GLM-5.2 lmcache_mp cache {layer_name}.latent has width "
                    f"{latent_view.shape[2]}, expected {latent_width}"
                )
            latent.append((f"{layer_name}.latent", latent_view))
            latent_layers.add(layer_name)
        index_tensor = getattr(cache, "index_cache", None)
        if isinstance(index_tensor, torch.Tensor) and index_tensor.numel() > 0:
            index_view = _reshape_block_cache(
                index_tensor,
                name=f"{layer_name}.index",
                num_blocks=num_blocks,
                block_size=block_size,
            )
            if index_view.shape[2] != index_width:
                raise ValueError(
                    f"GLM-5.2 lmcache_mp cache {layer_name}.index has width "
                    f"{index_view.shape[2]}, expected {index_width}"
                )
            indexes.append((f"{layer_name}.index", index_view))
            index_layers.add(layer_name)

    if not latent:
        raise ValueError("GLM-5.2 lmcache_mp found no MLA latent cache tensors")
    if not indexes:
        raise ValueError("GLM-5.2 lmcache_mp found no DSA index cache tensors")
    if latent_layers != index_layers:
        missing_index = sorted(latent_layers - index_layers)
        missing_latent = sorted(index_layers - latent_layers)
        raise ValueError(
            "GLM-5.2 lmcache_mp requires one latent and index cache per layer: "
            f"missing_index={missing_index}, missing_latent={missing_latent}"
        )

    ordered = latent + indexes
    devices = {tensor.device for _, tensor in ordered}
    if len(devices) != 1:
        raise ValueError("GLM-5.2 lmcache_mp cache tensors must share one device")
    tensors = dict(ordered)
    latent_indices = tuple(range(len(latent)))
    index_indices = tuple(range(len(latent), len(ordered)))
    total_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    return _GLM52CacheViews(
        tensors=tensors,
        latent_indices=latent_indices,
        index_indices=index_indices,
        bytes_per_block=total_bytes // num_blocks,
    )


@dataclass
class _LookupState:
    token_ids: list[int]
    hit: int | None = None
    retrieve_start: int | None = None


@dataclass
class _PendingLoad:
    completion: LoadCompletionId
    blocks: set[int]
    future: Any | None
    event: Any
    adapter_reported: bool = False
    failed_hint: bool = False


@dataclass
class _PendingSave:
    completion: SaveCompletionId
    future: Any | None
    event: Any
    adapter_reported: bool = False


def _adapter_transfer_future(
    adapter: Any, request_id: str, *, store: bool
) -> Any | None:
    """Retain LMCache v0.5.3's device future after submission.

    The pinned adapter removes futures and CUDA events as soon as
    ``get_finished`` reports an ID, including its unhealthy fast path. ATOM
    retains both until the device future is truly terminal so blocks cannot be
    recomputed or released while a server-side copy may still touch them.
    """

    attribute = "store_futures" if store else "retrieve_futures"
    tracked = getattr(adapter, attribute, None)
    if not isinstance(tracked, dict):
        raise TypeError(
            f"LMCache MP adapter does not expose required {attribute} tracking"
        )
    value = tracked.get(request_id)
    if value is None:
        # The v0.5.3 adapter deliberately creates no future when it drops an
        # operation before submission because the server is already unhealthy.
        return None
    future = value if store else value[0]
    if not callable(getattr(future, "query", None)) or not callable(
        getattr(future, "result", None)
    ):
        raise TypeError(
            f"LMCache MP {attribute}[{request_id!r}] has no future interface"
        )
    return future


def _terminal_future_result(future: Any | None) -> tuple[bool, Any]:
    """Return ``(terminal, result)`` without blocking on a device future."""

    if future is None:
        return True, None
    try:
        if not future.query():
            return False, None
        return True, future.result(timeout=0)
    except Exception:
        # A query/result exception is not proof that the remote GPU stream has
        # quiesced. Keep the operation pending rather than risk block reuse.
        logger.warning(
            "LMCache MP transfer future could not prove terminal state",
            exc_info=True,
        )
        return False, None


class _MPLookupClient:
    """Synchronous lookup facade used by ATOM's existing scheduler policy."""

    token_database = None

    def __init__(self, adapter: Any, *, timeout: float, poll_interval: float) -> None:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("LMCache MP lookup timeout and poll interval must be > 0")
        self._adapter = adapter
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._lookups: dict[str, _LookupState] = {}

    def lookup(self, token_ids: list[int], lookup_id: str) -> int:
        state = _LookupState(token_ids=list(token_ids))
        self._lookups[lookup_id] = state
        self._adapter.maybe_submit_lookup_request(lookup_id, token_ids)
        deadline = time.monotonic() + self._timeout
        while True:
            result = self._adapter.check_lookup_result(lookup_id)
            if result is not None:
                hit = int(result)
                state.hit = hit
                return hit
            if time.monotonic() >= deadline:
                logger.warning(
                    "LMCache MP lookup timed out after %.1fs for request %s",
                    self._timeout,
                    lookup_id,
                )
                # The MP API has no cancel-prefetch call. Keep the adapter job
                # intact so request_finished() can release locks if the result
                # becomes available; eagerly cleaning it here would orphan the
                # server-side lookup and its locks.
                return 0
            time.sleep(self._poll_interval)

    def prepare_retrieve(self, lookup_id: str, start: int) -> None:
        state = self._lookups.get(lookup_id)
        if state is not None and state.hit is not None and start > 0:
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=0,
                end=min(start, state.hit),
                request_id=lookup_id,
            )
        if state is not None:
            state.retrieve_start = start
        self._adapter.cleanup_lookup_result(lookup_id)

    def complete_retrieve(self, lookup_id: str, *, succeeded: bool) -> None:
        # Once a retrieve is submitted, LMCache owns the remaining lookup
        # locks. Its lmcache-driven transfer releases them on both success and
        # failure (including partial failures). Releasing the range again from
        # the scheduler would decrement every TP rank's read locks twice. The
        # scheduler cannot distinguish a pre-submit failure; that rarer case is
        # left to request end_session/server TTL cleanup instead.
        self._lookups.pop(lookup_id, None)
        self._adapter.cleanup_lookup_result(lookup_id)

    def hit_tokens(self, lookup_id: str) -> int | None:
        state = self._lookups.get(lookup_id)
        return None if state is None else state.hit

    def clear_lookup_status(self, lookup_id: str) -> None:
        state = self._lookups.pop(lookup_id, None)
        if state is not None and state.hit is None:
            result = self._adapter.check_lookup_result(lookup_id)
            if result is None:
                logger.warning(
                    "LMCache MP lookup for request %s is still pending during "
                    "cleanup; dropping local state while server TTL/session "
                    "cleanup releases any eventual locks",
                    lookup_id,
                )
                self._adapter.cleanup_lookup_result(lookup_id)
                return
            state.hit = int(result)
        # Once retrieve has started, the transfer owns the remaining read
        # locks. Failed terminal loads call complete_retrieve(False) first.
        if (
            state is not None
            and state.retrieve_start is None
            and state.hit
            and state.hit > 0
        ):
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=0,
                end=state.hit,
                request_id=lookup_id,
            )
        self._adapter.cleanup_lookup_result(lookup_id)


class GLM52LMCacheMPConnector(KVConnectorBase):
    """Worker-side GLM-5.2 LMCache MP connector."""

    is_producer = False

    def __init__(self, config: Any) -> None:
        _validate_glm52_config(config)
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self.block_size = offcfg._strict_integer(
            "GLM-5.2 LMCache MP block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        self.chunk_size: int | None = None
        self._adapter: Any = None
        self._pending_saves: dict[str, _PendingSave] = {}
        self._pending_loads: dict[str, _PendingLoad] = {}
        self._immediate_saves: set[SaveCompletionId] = set()
        self._immediate_load_failures: set[LoadCompletionId] = set()
        self._lock = threading.Lock()

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        if num_blocks is None:
            raise ValueError("GLM-5.2 lmcache_mp requires the scheduler block count")
        normalized_num_blocks = offcfg._strict_integer(
            "GLM-5.2 LMCache MP block count",
            num_blocks,
            minimum=1,
        )

        from aiter.dist.parallel_state import get_tp_group
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo

        tp = get_tp_group()
        rank = int(tp.rank_in_group)
        hf_config = getattr(self._config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", hf_config)
        index_head_dim = offcfg._strict_integer(
            "GLM-5.2 index head dimension",
            getattr(text_config, "index_head_dim", 128),
            minimum=1,
        )
        index_width = ((index_head_dim + 4 + 15) // 16) * 16
        views = _build_glm52_cache_views(
            kv_caches,
            num_blocks=normalized_num_blocks,
            block_size=self.block_size,
            index_width=index_width,
        )
        block_regions = getattr(transfer_tensors, "block_regions", None) or []
        if block_regions:
            expected = sum(int(region.unit_bytes) for region in block_regions)
            if expected != views.bytes_per_block:
                raise ValueError(
                    "GLM-5.2 lmcache_mp block geometry mismatch: "
                    f"views={views.bytes_per_block} transfer_regions={expected}"
                )

        adapter = _make_worker_adapter(self._config, rank)
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=views.latent_indices,
                tokens_per_block=self.block_size,
            ),
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=views.index_indices,
                tokens_per_block=self.block_size,
            ),
        ]
        try:
            for attribute in ("store_futures", "retrieve_futures"):
                if not isinstance(getattr(adapter, attribute, None), dict):
                    raise TypeError(
                        "GLM-5.2 lmcache_mp requires LMCache v0.5.3 device "
                        f"future tracking ({attribute})"
                    )
            adapter.register_kv_caches(views.tensors, engine_group_infos=groups)
            chunk_size = offcfg._strict_integer(
                "LMCache MP chunk size",
                adapter.lmcache_tokens_per_chunk,
                minimum=1,
            )
            if chunk_size % self.block_size:
                raise ValueError(
                    f"LMCache MP chunk size {chunk_size} must be divisible by "
                    f"ATOM block size {self.block_size}"
                )
        except Exception:
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise
        self._adapter = adapter
        self.chunk_size = chunk_size
        logger.info(
            "GLM-5.2 LMCache MP registered rank=%d tensors=%d latent_layers=%d "
            "index_layers=%d bytes_per_block=%d chunk=%d save=%s load=%s",
            rank,
            len(views.tensors),
            len(views.latent_indices),
            len(views.index_indices),
            views.bytes_per_block,
            self.chunk_size,
            self._do_save,
            self._do_load,
        )

    def start_load_kv(self, metadata: Any) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        if self._adapter is None or self.chunk_size is None:
            raise RuntimeError("GLM-5.2 lmcache_mp KV caches are not registered")

        requests = [
            req
            for req in metadata.requests
            if (req.load_spec is not None and self._do_load)
            or (req.save_spec is not None and self._do_save)
        ]
        if not requests:
            return
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        for req in requests:
            if req.load_spec is not None and self._do_load:
                self._submit_load(req, event)
            if req.save_spec is not None and self._do_save:
                self._submit_save(req, event)

    def _block_slice(
        self, req: LMCacheReqMeta, start: int, end: int
    ) -> tuple[list[int], set[int]]:
        if start < 0 or end < start:
            raise ValueError(f"invalid LMCache MP token range [{start}, {end})")
        if start % self.block_size or end % self.block_size:
            raise ValueError(
                f"LMCache MP token range [{start}, {end}) must align to "
                f"block size {self.block_size}"
            )
        block_ids = list(
            req.block_ids[start // self.block_size : end // self.block_size]
        )
        expected = (end - start) // self.block_size
        if len(block_ids) != expected:
            raise ValueError(
                f"LMCache MP request {req.req_id} needs {expected} blocks for "
                f"[{start}, {end}), got {len(block_ids)}"
            )
        return block_ids, set(block_ids)

    def _submit_load(self, req: LMCacheReqMeta, event: Any) -> None:
        from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

        assert req.load_spec is not None
        completion = req.load_operation or req.req_id
        request_id = str(req.req_id)
        start = int(req.load_spec.hbm_cached_tokens)
        end = (
            int(req.load_spec.lmcache_cached_tokens)
            if req.load_spec.transfer_end_tokens is None
            else int(req.load_spec.transfer_end_tokens)
        )
        try:
            if start % self.chunk_size or end % self.chunk_size:
                raise ValueError(
                    f"load range [{start}, {end}) is not LMCache chunk aligned "
                    f"({self.chunk_size})"
                )
            block_ids, block_set = self._block_slice(req, start, end)
            op = LoadStoreOp(
                token_ids=list(req.token_ids),
                block_ids=[block_ids],
                start=start,
                end=end,
            )
            self._adapter.submit_retrieve_request(request_id, op, event)
        except Exception:
            logger.exception(
                "GLM-5.2 LMCache MP load submission failed for %s",
                req.req_id,
            )
            with self._lock:
                self._immediate_load_failures.add(completion)
            return
        # Failure to observe a submitted future is an adapter incompatibility,
        # not a safe request-level miss: propagate and fail-stop the worker.
        future = _adapter_transfer_future(
            self._adapter,
            request_id,
            store=False,
        )
        with self._lock:
            self._pending_loads[request_id] = _PendingLoad(
                completion=completion,
                blocks=block_set,
                future=future,
                event=event,
                failed_hint=future is None,
            )

    def _submit_save(self, req: LMCacheReqMeta, event: Any) -> None:
        from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

        assert req.save_spec is not None
        completion = req.save_operation or req.req_id
        request_id = str(req.req_id)
        end = (len(req.token_ids) // self.chunk_size) * self.chunk_size
        start = (
            int(req.save_spec.skip_leading_tokens) // self.chunk_size
        ) * self.chunk_size
        if start >= end:
            with self._lock:
                self._immediate_saves.add(completion)
            return
        try:
            _prepare_store_generation(self._adapter, request_id)
            block_ids, _ = self._block_slice(req, start, end)
            op = LoadStoreOp(
                token_ids=list(req.token_ids),
                block_ids=[block_ids],
                start=start,
                end=end,
            )
            self._adapter.submit_store_request(request_id, op, event)
        except Exception:
            logger.exception(
                "GLM-5.2 LMCache MP save submission failed for %s",
                req.req_id,
            )
            with self._lock:
                self._immediate_saves.add(completion)
            return
        future = _adapter_transfer_future(
            self._adapter,
            request_id,
            store=True,
        )
        with self._lock:
            self._pending_saves[request_id] = _PendingSave(
                completion=completion,
                future=future,
                event=event,
            )

    def get_finished(self) -> KVConnectorOutput:
        if self._adapter is None:
            return KVConnectorOutput()
        with self._lock:
            engine_finished = set(self._pending_saves)
        try:
            finished_saves, finished_loads = self._adapter.get_finished(engine_finished)
            error_blocks = set(self._adapter.get_block_ids_with_load_errors())
        except Exception:
            logger.exception("GLM-5.2 LMCache MP completion polling failed")
            with self._lock:
                # A polling exception does not prove that remote device work
                # has quiesced. Preserve every submitted operation and drain
                # only requests that failed or completed before submission.
                failed = set(self._immediate_load_failures)
                saved = set(self._immediate_saves)
                self._immediate_load_failures.clear()
                self._immediate_saves.clear()
            return KVConnectorOutput(failed_loading=failed, finished_saving=saved)

        done_load: set[LoadCompletionId] = set()
        failed_load: set[LoadCompletionId] = set()
        done_save: set[SaveCompletionId] = set()
        with self._lock:
            for request_id in finished_saves or set():
                pending = self._pending_saves.get(request_id)
                if pending is not None:
                    pending.adapter_reported = True
            for request_id in finished_loads or set():
                pending = self._pending_loads.get(request_id)
                if pending is not None:
                    pending.adapter_reported = True
            for pending in self._pending_loads.values():
                if pending.blocks & error_blocks:
                    pending.failed_hint = True

            for request_id, pending in list(self._pending_saves.items()):
                if not pending.adapter_reported:
                    continue
                terminal, _result = _terminal_future_result(pending.future)
                if terminal:
                    self._pending_saves.pop(request_id, None)
                    done_save.add(pending.completion)

            for request_id, pending in list(self._pending_loads.items()):
                if not pending.adapter_reported:
                    continue
                terminal, result = _terminal_future_result(pending.future)
                if not terminal:
                    continue
                self._pending_loads.pop(request_id, None)
                if pending.failed_hint or result is not True:
                    failed_load.add(pending.completion)
                else:
                    done_load.add(pending.completion)
            done_save.update(self._immediate_saves)
            failed_load.update(self._immediate_load_failures)
            self._immediate_saves.clear()
            self._immediate_load_failures.clear()
        return KVConnectorOutput(
            finished_loading=done_load,
            failed_loading=failed_load,
            finished_saving=done_save,
        )


def _prepare_store_generation(adapter: Any, request_id: str) -> None:
    """Allow ATOM's incremental stores to reuse one engine request ID.

    LMCache v0.5.3 permanently deduplicates completed store request IDs because
    vLLM normally stores once at request completion. ATOM stores each newly
    computed prompt frontier. Reset only the already-terminal bookkeeping before
    the next generation; the connector guarantees at most one in-flight store
    per engine request.
    """

    public_method = getattr(adapter, "prepare_store_generation", None)
    if callable(public_method):
        public_method(request_id)
        return
    for attribute in (
        "_returned_finished",
        "finished_stores",
        "previously_finished",
    ):
        values = getattr(adapter, attribute, None)
        if values is not None:
            values.discard(request_id)


class GLM52LMCacheMPConnectorScheduler(DenseOffloadScheduler):
    """Scheduler-side GLM-5.2 LMCache MP connector."""

    def __init__(self, config: Any) -> None:
        _validate_glm52_config(config)
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        validated_kv_role(kvc)
        offcfg._strict_integer(
            "GLM-5.2 LMCache MP block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        adapter = _make_scheduler_adapter(config)
        try:
            extra = _extra_config(config)
            timeout = float(extra.get("lmcache.mp.lookup_timeout", 30.0))
            poll_interval = float(extra.get("lmcache.mp.lookup_poll_interval", 0.01))
            lookup_client = _MPLookupClient(
                adapter,
                timeout=timeout,
                poll_interval=poll_interval,
            )
            self._mp_adapter = adapter
            self._init_dense_scheduler(
                config,
                chunk_size=int(adapter.lmcache_tokens_per_chunk),
                lookup_client=lookup_client,
            )
        except Exception:
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        matched = super().get_num_new_matched_tokens(seq)
        sid = str(seq.id)
        hit = self._lookup_client.hit_tokens(sid)
        num_prompt = int(seq.num_prompt_tokens)
        if hit != num_prompt or num_prompt % self.chunk_size:
            return matched

        load_spec = self._load_specs.get(sid)
        if load_spec is None:
            return matched
        if self.block_size == 1:
            # Allocating prompt-1 one-token blocks cannot provide a destination
            # for the complete final LMCache chunk. Recompute instead.
            self._clear_pending_load(sid)
            return 0, False

        load_spec.transfer_end_tokens = num_prompt
        self._hit_save_floors[sid] = num_prompt
        return matched

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        metadata = super().build_connector_meta()
        for req in metadata.requests:
            if req.load_spec is not None:
                self._lookup_client.prepare_retrieve(
                    str(req.req_id),
                    int(req.load_spec.hbm_cached_tokens),
                )
        return metadata

    def load_finished(self, req_id: Any) -> bool:
        finished = super().load_finished(req_id)
        if finished:
            raw_id = req_id.req_id if hasattr(req_id, "req_id") else req_id
            try:
                self._lookup_client.complete_retrieve(
                    str(raw_id),
                    succeeded=True,
                )
            except Exception:
                logger.warning(
                    "LMCache MP successful-load cleanup failed for request %s",
                    raw_id,
                    exc_info=True,
                )
        return finished

    def load_failed(self, req_id: Any) -> bool:
        raw_id = req_id.req_id if hasattr(req_id, "req_id") else req_id
        active = self._active_load_operations.get(str(raw_id))
        if isinstance(req_id, LoadOperationId):
            is_current = active is not None and active[1] == req_id
        else:
            is_current = active is None
        if is_current:
            try:
                self._lookup_client.complete_retrieve(
                    str(raw_id),
                    succeeded=False,
                )
            except Exception:
                logger.warning(
                    "LMCache MP failed-load cleanup failed for request %s",
                    raw_id,
                    exc_info=True,
                )
        return super().load_failed(req_id)

    def request_finished(self, seq: Any) -> None:
        super().request_finished(seq)
        try:
            self._mp_adapter.end_session(str(seq.id))
        except Exception:
            logger.warning(
                "GLM-5.2 LMCache MP end_session failed for request %s",
                seq.id,
                exc_info=True,
            )


__all__ = [
    "GLM52LMCacheMPConnector",
    "GLM52LMCacheMPConnectorScheduler",
]
