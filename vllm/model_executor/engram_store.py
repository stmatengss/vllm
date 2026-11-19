# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-row storage for immutable Engram tables.

Models own row ids, dtypes, scales and collectives. A backend owns the table
and reads rows. The Mooncake backend publishes a local snapshot through
``mooncake.engram.EngramStore`` and reopens that snapshot when another process
has already published the same bytes.
"""

import contextlib
import hashlib
from abc import ABC, abstractmethod

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_FINGERPRINT_CHUNK = 8 * 1024 * 1024


class EngramStorage(ABC):
    """One immutable table of opaque byte rows."""

    @abstractmethod
    def initialize(self, rows: torch.Tensor) -> None:
        """Copy a CPU ``uint8`` matrix of shape ``[num_rows, row_bytes]``."""

    @abstractmethod
    def lookup(
        self,
        ids: torch.Tensor,
        output: torch.Tensor | None = None,
        *,
        stream: int | None = None,
    ) -> torch.Tensor:
        """Return ``uint8 [*ids.shape, row_bytes]`` on ``ids.device``.

        Ids are int32 or int64. ``-1`` leaves the corresponding output bytes
        untouched, so callers zero ``output`` when those rows must be zero.
        CPU lookup is synchronous. CUDA lookup is ordered on ``stream``, or on
        the current stream when ``stream`` is omitted.
        """

    @abstractmethod
    def close(self) -> None:
        """Release mappings after lookups and captured graphs have finished."""


def fingerprint_byte_rows(rows: torch.Tensor) -> str:
    """Hash row shape and bytes so identical tables share one snapshot."""
    array = rows.detach().contiguous().view(torch.uint8).numpy()
    hasher = hashlib.sha256()
    hasher.update(f"{array.shape[0]}x{array.shape[1]}".encode())
    view = memoryview(array).cast("B")
    for start in range(0, view.nbytes, _FINGERPRINT_CHUNK):
        hasher.update(view[start : start + _FINGERPRINT_CHUNK])
    return hasher.hexdigest()


def create_engram_storage(local_dir: str, namespace: str) -> EngramStorage:
    """Build the local Mooncake backend. Remote lookup is not implemented."""
    return MooncakeEngramStorage(local_dir=local_dir, namespace=namespace)


class MooncakeEngramStorage(EngramStorage):
    """Local Mooncake EngramStore snapshot, shared by content hash."""

    def __init__(self, *, local_dir: str, namespace: str) -> None:
        if not isinstance(local_dir, str) or not local_dir:
            raise ValueError("mooncake_local_dir must be a nonempty string")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("mooncake_namespace must be a nonempty string")
        self._local_dir = local_dir
        self._namespace = namespace
        self._store = None
        self._row_bytes: int | None = None

    def initialize(self, rows: torch.Tensor) -> None:
        from mooncake.engram import EngramStore

        if self._store is not None:
            raise RuntimeError("Engram tables are immutable; already initialized")
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up Engram tables before CUDA graph capture")
        rows = rows.detach().contiguous()
        if rows.device.type != "cpu" or rows.dtype != torch.uint8 or rows.ndim != 2:
            raise ValueError("Engram initialization requires a CPU byte matrix")
        digest = fingerprint_byte_rows(rows)
        store = EngramStore(
            tables={"rows": {"num_rows": rows.shape[0], "row_bytes": rows.shape[1]}},
            namespace=f"{self._namespace}/{digest}",
            local_dir=self._local_dir,
            persistent=True,
        )
        try:
            try:
                store.populate("rows", rows.numpy())
                action = "published"
            except FileExistsError:
                action = "attached to"
        except BaseException:
            store.close()
            raise
        self._row_bytes = rows.shape[1]
        self._store = store
        logger.info(
            "Mooncake Engram: %s %d rows x %d bytes (%.2f GiB) under %s",
            action,
            rows.shape[0],
            rows.shape[1],
            rows.numel() / 1024**3,
            self._local_dir,
        )

    def lookup(
        self,
        ids: torch.Tensor,
        output: torch.Tensor | None = None,
        *,
        stream: int | None = None,
    ) -> torch.Tensor:
        if self._store is None or self._row_bytes is None:
            raise RuntimeError("Initialize Engram storage before lookup")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Engram row ids must be int32 or int64")
        flat_ids = ids.reshape(-1, 1).contiguous()
        if output is None:
            output = torch.zeros(
                (*ids.shape, self._row_bytes),
                dtype=torch.uint8,
                device=ids.device,
            )
        elif (
            output.shape != (*ids.shape, self._row_bytes)
            or output.dtype != torch.uint8
            or output.device != ids.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "Engram lookup output must be a contiguous uint8 tensor of "
                "shape [*ids.shape, row_bytes] on the id device"
            )
        flat_out = output.view(flat_ids.shape[0], 1, self._row_bytes)
        if flat_ids.numel():
            if ids.is_cuda:
                if stream is None:
                    stream = torch.cuda.current_stream(ids.device).cuda_stream
                self._store.lookup(["rows"], flat_ids, flat_out, stream=stream)
            else:
                if stream is not None:
                    raise ValueError("CPU Engram lookup does not take a CUDA stream")
                self._store.lookup(["rows"], flat_ids.numpy(), flat_out.numpy())
        return output

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
