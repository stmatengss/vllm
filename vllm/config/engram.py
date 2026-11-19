# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator
from typing_extensions import Self

from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Requires sufficient
    /dev/shm capacity and a shared IPC namespace. Defaults to enabled whenever
    the other settings allow it, falling back to per-replica tables when DP
    replicas are not co-located on one node or /dev/shm cannot hold them."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory."""

    storage_backend: Literal["native", "mooncake"] = "native"
    """Where CPU-offloaded Engram rows live. ``native`` keeps today's pinned
    or ``/dev/shm`` tables. ``mooncake`` publishes each table into a local
    Mooncake EngramStore so processes that load the same bytes share one
    snapshot. Remote Mooncake lookup is not implemented."""

    mooncake_local_dir: str = "/dev/shm/vllm-engram"
    """Directory for local Mooncake Engram snapshots. Co-located processes
    must see the same directory. Ignored when ``storage_backend`` is native."""

    mooncake_namespace: str = "vllm-v1"
    """Prefix of the content-addressed Mooncake namespace. A hash of the
    published bytes is appended, so a new checkpoint does not reuse old rows."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.storage_backend == "mooncake":
            if not self.cpu_offload:
                raise ValueError("storage_backend='mooncake' requires cpu_offload=True")
            if self.dp_shared_memory:
                raise ValueError(
                    "storage_backend='mooncake' cannot be combined with "
                    "dp_shared_memory"
                )
            if self.use_thp:
                raise ValueError(
                    "storage_backend='mooncake' cannot be combined with use_thp"
                )
            if not self.mooncake_local_dir:
                raise ValueError("mooncake_local_dir must be a nonempty string")
            if not self.mooncake_namespace:
                raise ValueError("mooncake_namespace must be a nonempty string")
            return self
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )
        return self

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda_alike()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings, non-empty n-gram layer ids, and a CUDA-alike "
                "device (CUDA or ROCm)."
            )
        if self.storage_backend == "mooncake" and not current_platform.is_cuda():
            raise ValueError(
                "Mooncake Engram storage requires NVIDIA CUDA "
                "(mooncake.engram.EngramStore)."
            )

    def resolve_dp_shared_memory(self, parallel_config: "ParallelConfig") -> None:
        """Share host tables by default wherever the configuration permits."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.storage_backend == "native"
                and self.cpu_offload
                and not self.use_thp
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
