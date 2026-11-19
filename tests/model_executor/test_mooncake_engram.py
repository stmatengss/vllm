# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for Mooncake Engram snapshots. No Mooncake install required."""

import sys
import unittest
from types import ModuleType
from unittest import mock

import numpy as np
import torch

from vllm.model_executor.engram_store import (
    MooncakeEngramStorage,
    fingerprint_byte_rows,
)
from vllm.models.deepseek_v41.nvidia.engram import (
    dequantize_packed_engram,
    lookup_mooncake_engram_rows,
    owned_engram_row_ids,
    publish_mooncake_engrams,
)


class _SharedTables:
    rows: dict[str, np.ndarray] = {}


class _FakeEngramStore:
    def __init__(self, *, tables, namespace, local_dir, persistent):
        spec = tables["rows"]
        self.namespace = namespace
        self.num_rows = spec["num_rows"]
        self.row_bytes = spec["row_bytes"]
        self.local_dir = local_dir
        self.persistent = persistent
        self.close_count = 0

    def populate(self, table_id, rows):
        if table_id != "rows" or not self.persistent:
            raise ValueError("expected a persistent rows table")
        if rows.dtype != np.uint8 or rows.shape != (self.num_rows, self.row_bytes):
            raise ValueError("table metadata does not match the published rows")
        if not rows.flags.c_contiguous:
            raise ValueError("populate requires contiguous rows")
        if self.namespace in _SharedTables.rows:
            raise FileExistsError(self.namespace)
        _SharedTables.rows[self.namespace] = rows.copy()

    def lookup(self, table_ids, indices, output, stream=None):
        del stream
        if table_ids != ["rows"] or indices.shape[1:] != (1,):
            raise ValueError("expected one table column")
        rows = _SharedTables.rows[self.namespace]
        valid = indices[:, 0] >= 0
        output[valid, 0] = rows[indices[valid, 0]]

    def close(self):
        self.close_count += 1


class MooncakeEngramStorageTest(unittest.TestCase):
    def setUp(self):
        _SharedTables.rows = {}
        module = ModuleType("mooncake.engram")
        module.EngramStore = _FakeEngramStore
        patch = mock.patch.dict(
            sys.modules,
            {"mooncake": ModuleType("mooncake"), "mooncake.engram": module},
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(_SharedTables.rows.clear)

    def test_publish_then_attach_shares_one_snapshot(self):
        rows = torch.arange(24, dtype=torch.uint8).reshape(6, 4)
        rows = rows[:, ::2]
        expected = rows.contiguous().clone()
        first = MooncakeEngramStorage(
            local_dir="/dev/shm/vllm-engram", namespace="vllm-v1"
        )
        second = MooncakeEngramStorage(
            local_dir="/dev/shm/vllm-engram", namespace="vllm-v1"
        )
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        first.initialize(rows)
        rows.zero_()
        second.initialize(expected)
        self.assertEqual(len(_SharedTables.rows), 1)
        ids = torch.tensor([[5, -1], [0, 2]], dtype=torch.int32)
        actual = second.lookup(ids)
        wanted = torch.zeros((*ids.shape, 2), dtype=torch.uint8)
        wanted[ids >= 0] = expected.reshape(-1, 2)[ids[ids >= 0].long()]
        torch.testing.assert_close(actual, wanted)

    def test_distinct_bytes_do_not_alias(self):
        left = MooncakeEngramStorage(local_dir="/tmp/engram", namespace="vllm-v1")
        right = MooncakeEngramStorage(local_dir="/tmp/engram", namespace="vllm-v1")
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        left.initialize(torch.arange(8, dtype=torch.uint8).reshape(2, 4))
        right.initialize(torch.arange(8, dtype=torch.uint8).reshape(2, 4) + 3)
        self.assertEqual(len(_SharedTables.rows), 2)
        torch.testing.assert_close(
            left.lookup(torch.tensor([1])),
            torch.tensor([[4, 5, 6, 7]], dtype=torch.uint8),
        )

    def test_rejected_rows_can_be_retried(self):
        storage = MooncakeEngramStorage(local_dir="/tmp/engram", namespace="ns")
        self.addCleanup(storage.close)
        for rows in (
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones(8, dtype=torch.uint8),
            torch.empty((2, 4), dtype=torch.uint8, device="meta"),
        ):
            with self.assertRaisesRegex(ValueError, "CPU byte matrix"):
                storage.initialize(rows)
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        storage.initialize(rows)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            storage.initialize(rows)
        self.assertEqual(
            fingerprint_byte_rows(rows),
            fingerprint_byte_rows(rows.clone()),
        )

    def test_publication_failure_leaves_storage_closed(self):
        storage = MooncakeEngramStorage(local_dir="/tmp/engram", namespace="ns")
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        with (
            mock.patch.object(
                _FakeEngramStore, "populate", side_effect=OSError("disk")
            ),
            self.assertRaisesRegex(OSError, "disk"),
        ):
            storage.initialize(rows)
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            storage.lookup(torch.tensor([0]))
        storage.initialize(rows)
        torch.testing.assert_close(
            storage.lookup(torch.tensor([0], dtype=torch.int64)), rows[:1]
        )

    def test_empty_lookup_skips_the_backend(self):
        storage = MooncakeEngramStorage(local_dir="/tmp/engram", namespace="ns")
        self.addCleanup(storage.close)
        storage.initialize(torch.ones((2, 4), dtype=torch.uint8))
        with mock.patch.object(_FakeEngramStore, "lookup", side_effect=AssertionError):
            actual = storage.lookup(torch.empty((0, 3), dtype=torch.int64))
        self.assertEqual(actual.shape, (0, 3, 4))

    def test_composed_lookup_dequantizes_packed_rows(self):
        dim, block_size = 4, 2
        weight = torch.tensor(
            [[1.0, 2.0, 4.0, 8.0], [2.0, 2.0, 2.0, 2.0]],
            dtype=torch.float8_e4m3fn,
        )
        scales = torch.tensor([[127, 128], [127, 127]], dtype=torch.uint8)
        storage = MooncakeEngramStorage(local_dir="/tmp/engram", namespace="ns")
        self.addCleanup(storage.close)
        storage.initialize(torch.cat((weight.view(torch.uint8), scales), dim=1))
        ids = torch.tensor([[1, -1], [0, 1]], dtype=torch.int32)
        out = torch.empty(2, 2, dim, dtype=torch.bfloat16)
        lookup_mooncake_engram_rows(
            storage, ids, dim=dim, block_size=block_size, out=out
        )
        expected = torch.tensor(
            [
                [[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
                [[1.0, 2.0, 8.0, 16.0], [2.0, 2.0, 2.0, 2.0]],
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(out, expected)

    def test_graph_capture_rejects_publication(self):
        storage = MooncakeEngramStorage(local_dir="/tmp/engram", namespace="ns")
        with (
            mock.patch("torch.cuda.is_initialized", return_value=True),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"),
        ):
            storage.initialize(torch.ones((2, 4), dtype=torch.uint8))
        self.assertEqual(_SharedTables.rows, {})


class EngramRowMathTest(unittest.TestCase):
    def test_owned_ids_drop_other_heads_and_other_shards(self):
        indices = torch.tensor([[0, 5, 10, -1], [4, 11, 3, 9]], dtype=torch.int32)
        local = owned_engram_row_ids(
            indices,
            head_start=1,
            local_heads=3,
            total_heads=4,
            vocab_start=4,
            vocab_end=12,
        )
        torch.testing.assert_close(
            local,
            torch.tensor([[1, 6, -1], [7, -1, 5]], dtype=torch.int32),
        )

    def test_packed_fp8_rows_match_ue8m0_scales(self):
        weight = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float8_e4m3fn)
        scales = torch.tensor([[127, 128]], dtype=torch.uint8)
        packed = torch.cat((weight.view(torch.uint8), scales), dim=1)
        out = torch.empty(1, 4, dtype=torch.bfloat16)
        dequantize_packed_engram(packed, dim=4, block_size=2, out=out)
        expected = torch.tensor([[1.0, 2.0, 8.0, 16.0]], dtype=torch.bfloat16)
        torch.testing.assert_close(out, expected)

    def test_publish_walker_only_calls_engram_modules(self):
        class Table(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def publish_mooncake(self):
                self.calls += 1

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = Table()
                self.other = torch.nn.Linear(2, 2)

        root = Root()
        publish_mooncake_engrams(root)
        publish_mooncake_engrams(root)
        self.assertEqual(root.table.calls, 2)


class PleSelectorTest(unittest.TestCase):
    def test_backend_selects_the_embedding_class(self):
        from vllm.config import EngramConfig
        from vllm.models.qwen4_exp.common.ngram_embedding import (
            Qwen4ExpPLEDeviceEmbedding,
            Qwen4ExpPLEMooncakeEmbedding,
            Qwen4ExpPLEPinnedHostEmbedding,
            select_ple_embedding_class,
        )

        mooncake = select_ple_embedding_class(EngramConfig(storage_backend="mooncake"))
        pinned = select_ple_embedding_class(EngramConfig(cpu_offload=True))
        device = select_ple_embedding_class(EngramConfig(cpu_offload=False))
        self.assertIs(mooncake, Qwen4ExpPLEMooncakeEmbedding)
        self.assertIs(pinned, Qwen4ExpPLEPinnedHostEmbedding)
        self.assertIs(device, Qwen4ExpPLEDeviceEmbedding)
        self.assertIs(select_ple_embedding_class(None), Qwen4ExpPLEDeviceEmbedding)


if __name__ == "__main__":
    unittest.main()
