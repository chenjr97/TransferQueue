# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU staging contracts, using a synchronous Store double with real buffer copies."""

import argparse
import ctypes
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from scripts.benchmark_mooncake_local_buffer import run
from transfer_queue.storage.clients import mooncake_client as mc
from transfer_queue.utils.mooncake_utils import _aligned_offsets


class BufferHandle(bytearray):
    pass


class BufferStore:
    def __init__(self):
        self.data = {}
        self.regions = {}
        self.register_calls = []
        self.staged_calls = []
        self.direct_calls = []
        self.staged_get_calls = []
        self.direct_get_calls = []
        self.handles = []
        self.closed = False

    def setup(self, *args):
        return 0

    def register_buffer(self, ptr, size):
        assert size > 0
        self.regions[ptr] = size
        self.register_calls.append((ptr, size))
        return 0

    def unregister_buffer(self, ptr):
        del self.regions[ptr]
        return 0

    def upsert_batch(self, keys, values, config):
        assert all(v.nbytes > 0 for v in values)
        self.staged_calls.append((list(keys), [v.nbytes for v in values]))
        for key, value in zip(keys, values, strict=True):
            assert value.itemsize == 1 and value.contiguous
            self.data[key] = bytes(value)
        return 0

    def batch_upsert_from(self, keys, ptrs, sizes, config):
        self.direct_calls.append(list(keys))
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            if size:
                self.data[key] = ctypes.string_at(ptr, size)
        return [0 if size else -600 for size in sizes]

    def batch_get_into(self, keys, ptrs, sizes):
        self.direct_get_calls.append(list(keys))
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            assert len(self.data[key]) == size
            ctypes.memmove(ptr, self.data[key], size)
        return sizes

    def batch_get_buffer(self, keys):
        self.staged_get_calls.append(list(keys))
        handles = [BufferHandle(self.data[key]) if key in self.data else None for key in keys]
        self.handles.extend(weakref.ref(handle) for handle in handles if handle is not None)
        return handles

    def batch_remove(self, keys, force):
        for key in keys:
            self.data.pop(key, None)
        return [0] * len(keys)

    def close(self):
        assert not self.regions
        self.closed = True


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setattr(mc, "MOONCAKE_STORE_IMPORTED", True)
    monkeypatch.setattr(mc, "ReplicateConfig", type("ReplicateConfig", (), {}), raising=False)
    monkeypatch.setattr(mc, "MooncakeDistributedStore", BufferStore, raising=False)
    monkeypatch.setattr(mc, "RETRY_DELAY_SECONDS", 0)
    clients = []

    def make(enabled=True, **staging):
        config = {
            "local_hostname": "127.0.0.1",
            "metadata_server": "P2PHANDSHAKE",
            "master_server_address": "127.0.0.1:50051",
            "local_buffer_size": 4096,
            "local_buffer_staging": {
                "enabled": enabled,
                "max_bytes": 2048,
                "acquire_timeout_s": 0.1,
                **staging,
            },
        }
        client = mc.MooncakeStoreClient(config)
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


def roundtrip(client, values):
    keys = [f"k{i}" for i in range(len(values))]
    meta = client.put(keys, values)
    result = client.get(
        keys,
        shapes=[v.shape if isinstance(v, torch.Tensor) else None for v in values],
        dtypes=[v.dtype if isinstance(v, torch.Tensor) else None for v in values],
        custom_backend_meta=meta,
    )
    return keys, meta, result


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.bool, torch.complex64]
)
def test_tensor_wire_format_and_roundtrip(make_client, dtype):
    client = make_client()
    values = [torch.arange(6).to(dtype).reshape(2, 3).t(), torch.tensor(1, dtype=dtype)]
    keys = ["matrix", "scalar"]
    assert client.put(keys, values) == [None, None]
    assert not client._store.register_calls
    assert not client._store.direct_calls
    for key, value in zip(keys, values, strict=True):
        raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        assert client._store.data[key] == raw
    result = client.get(keys, [v.shape for v in values], [dtype] * 2)
    assert client._store.staged_get_calls == [keys]
    assert not client._store.register_calls
    assert all(handle() is None for handle in client._store.handles)
    client.close()
    for value, actual in zip(values, result, strict=True):
        torch.testing.assert_close(actual, value)


def test_bytes_metadata_and_independent_decoded_storage(make_client):
    client = make_client()
    values = ["small", {"tensor": torch.arange(12).reshape(3, 4).t(), "array": np.arange(7)}, 42]
    keys, meta, result = roundtrip(client, values)
    assert meta == [{"packed_size": len(client._store.data[k])} for k in keys]
    assert client._store.staged_calls
    assert not client._store.direct_calls
    assert client._store.staged_get_calls
    assert not client._store.direct_get_calls
    assert all(handle() is None for handle in client._store.handles)
    client.put(keys, ["replacement"] * len(keys))
    client.close()
    assert result[0] == "small" and result[2] == 42
    torch.testing.assert_close(result[1]["tensor"], values[1]["tensor"])
    np.testing.assert_array_equal(result[1]["array"], values[1]["array"])


def test_mixed_output_order_and_default_off(make_client):
    values = [torch.ones(3), "abc", torch.tensor(7), {"key": "value"}]
    staged, direct = make_client(), make_client(enabled=False)
    _, staged_meta, result = roundtrip(staged, values)
    _, direct_meta, _ = roundtrip(direct, values)
    assert staged_meta == direct_meta
    assert staged._store.data == direct._store.data
    assert not direct._store.staged_calls
    assert not direct._store.staged_get_calls
    assert direct._store.direct_get_calls
    torch.testing.assert_close(result[0], values[0])
    torch.testing.assert_close(result[2], values[2])
    assert result[1] == values[1] and result[3] == values[3]


def test_byte_and_count_limits_alignment_and_oversized(make_client, monkeypatch):
    monkeypatch.setattr(mc, "BATCH_SIZE_LIMIT", 3)
    client = make_client(max_bytes=700)
    values = [torch.zeros(n, dtype=torch.uint8) for n in [1, 257, 700, 10, 1, 1, 1]]
    client.put([str(i) for i in range(len(values))], values)
    for keys, sizes in client._store.staged_calls:
        assert len(keys) <= 3
        assert _aligned_offsets(sizes)[1] <= 700
    assert sorted(client._store.direct_calls) == [["0", "1", "2"], ["3", "4", "5"]]
    assert client._store.staged_calls == [(["6"], [1])]
    assert set(client._store.data) == set(map(str, range(len(values))))


@pytest.mark.parametrize("count", [2, 3])
def test_whole_batch_capacity_boundary_preserves_direct_registration(make_client, count):
    client = make_client(max_bytes=512)
    source = torch.arange(count * 256, dtype=torch.int32).to(torch.uint8)
    values = list(source.split(256))
    keys = [str(i) for i in range(count)]
    client.put(keys, values)
    if count == 2:
        assert client._store.staged_calls == [(keys, [256] * count)]
        assert not client._store.register_calls
    else:
        assert not client._store.staged_calls
        assert client._store.direct_calls == [keys]
        assert client._store.register_calls == [(source.data_ptr(), source.nbytes)]
    actual = client.get(keys, [v.shape for v in values], [v.dtype for v in values])
    for expected, result in zip(values, actual, strict=True):
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("enabled", [False, True])
def test_zero_tensors_preserve_native_rejection_without_registering(make_client, enabled):
    client = make_client(enabled=enabled)
    with pytest.raises(RuntimeError, match="error codes.*-600"):
        client.put(["empty"], [torch.empty((0, 2), dtype=torch.int64)])
    assert not client._store.register_calls
    assert not client._store.staged_calls


def test_full_batch_fallback_then_per_key_retry(make_client, monkeypatch):
    client = make_client()
    monkeypatch.setattr(client._store, "upsert_batch", Mock(return_value=-500))
    original = client._store.batch_upsert_from
    calls = []

    def upsert(keys, ptrs, sizes, config):
        calls.append(list(keys))
        if len(calls) == 1:
            return [0, -500]
        return original(keys, ptrs, sizes, config)

    monkeypatch.setattr(client._store, "batch_upsert_from", upsert)
    client.put(["a", "b"], [torch.ones(1), torch.ones(1)])
    assert calls == [["a", "b"], ["b"]]
    assert not client._store.regions


def test_fallback_releases_budget_before_the_direct_put(make_client, monkeypatch):
    client = make_client(max_bytes=256)
    monkeypatch.setattr(client._store, "upsert_batch", Mock(return_value=-500))
    original = client._store.batch_upsert_from

    def upsert(keys, ptrs, sizes, config):
        # A concurrent lane must be able to take the whole budget while we fall back.
        with client._local_buffer_staging.acquire(256):
            return original(keys, ptrs, sizes, config)

    monkeypatch.setattr(client._store, "batch_upsert_from", upsert)
    client.put(["a"], [torch.ones(1)])
    assert client._store.data["a"] == bytes(torch.ones(1).numpy().data)


@pytest.mark.parametrize("failure", ["staged", "direct", "partial_registration"])
def test_failure_releases_budget_and_registered_regions(make_client, monkeypatch, failure):
    client = make_client()
    if failure == "staged":
        monkeypatch.setattr(client._store, "upsert_batch", Mock(side_effect=RuntimeError("test error")))
    else:
        monkeypatch.setattr(client._store, "upsert_batch", Mock(return_value=-500))
        if failure == "direct":
            monkeypatch.setattr(client._store, "batch_upsert_from", Mock(return_value=[-500, -500]))
        else:
            original = client._store.register_buffer
            calls = 0

            def register(ptr, size):
                nonlocal calls
                calls += 1
                return -1 if calls == 2 else original(ptr, size)

            monkeypatch.setattr(client._store, "register_buffer", register)
    # Distinct slices with a gap force two registration regions.
    source = torch.ones(1000)
    with pytest.raises(RuntimeError):
        client.put(["a", "b"], [source[:2], source[500:502]])
    assert not client._store.regions


@pytest.mark.parametrize("operation", ["put", "get"])
def test_budget_shared_across_public_calls(make_client, monkeypatch, operation):
    client = make_client(max_bytes=256, acquire_timeout_s=2)
    client.put(["b"], [torch.ones(1)])
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    store = client._store
    original = store.upsert_batch
    condition = client._local_buffer_staging._condition
    original_wait = condition.wait
    calls = 0

    def upsert(keys, values, config):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(3)
        return original(keys, values, config)

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(store, "upsert_batch", upsert)
    monkeypatch.setattr(condition, "wait", wait)
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            first = executor.submit(client.put, ["a"], [torch.ones(1)])
            assert entered.wait(2)
            if operation == "put":
                second = executor.submit(client.put, ["b"], [torch.ones(1)])
            else:
                second = executor.submit(client.get, ["b"], [(1,)], [torch.float32])
            assert waiting.wait(2)
            assert calls == 1
        finally:
            release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert calls == (2 if operation == "put" else 1)


def test_budget_timeout_does_not_submit_or_leak(make_client):
    client = make_client(max_bytes=256, acquire_timeout_s=0.01)
    with client._local_buffer_staging.acquire(256):
        with pytest.raises(TimeoutError, match="staging bytes"):
            client.put(["a"], [torch.ones(1)])
    assert not client._store.staged_calls
    client.put(["a"], [torch.ones(1)])


@pytest.mark.parametrize(
    "options",
    [
        {"max_bytes": 0},
        {"max_bytes": 8192},
        {"acquire_timeout_s": 0},
        {"acquire_timeout_s": float("nan")},
        {"acquire_timeout_s": float("inf")},
    ],
)
def test_configuration_validation(make_client, options):
    with pytest.raises(ValueError):
        make_client(**options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_input_conversion(make_client):
    client = make_client()
    value = torch.arange(12, device="cuda").reshape(3, 4).t()
    _, _, result = roundtrip(client, [value])
    torch.testing.assert_close(result[0], value.cpu())


def test_gdr_tensor_path_keeps_priority_while_bytes_use_cpu_staging(make_client, monkeypatch):
    client = make_client()
    client.use_gdr = True
    client._gdr_staging = Mock()
    gdr = Mock(return_value=[{"n_chunks": 2}])
    monkeypatch.setattr(client, "_put_tensors_gdr", gdr)
    meta = client.put(["tensor", "object"], [torch.ones(1), "abc"])
    gdr.assert_called_once()
    assert meta[0] == {"n_chunks": 2}
    assert meta[1] == {"packed_size": len(client._store.data["object"])}
    assert all(keys == ["object"] for keys, _ in client._store.staged_calls)
    gdr_get = Mock(return_value=([torch.ones(1)], [0]))
    monkeypatch.setattr(client, "_get_tensors_gdr", gdr_get)
    result = client.get(["tensor", "object"], [(1,), None], [torch.float32, None], meta)
    gdr_get.assert_called_once()
    assert client._store.staged_get_calls == [["object"]]
    torch.testing.assert_close(result[0], torch.ones(1))
    assert result[1] == "abc"


def test_mixed_empty_and_nonempty_matches_direct_path_outcome(make_client):
    staged, direct = make_client(), make_client(enabled=False)
    for client in (staged, direct):
        with pytest.raises(RuntimeError, match="error codes.*-600"):
            client.put(["empty", "full"], [torch.empty(0), torch.ones(2)])
    # The storable key lands on both paths; only the zero-byte key is rejected.
    assert staged._store.data == direct._store.data == {"full": direct._store.data["full"]}
    assert not staged._store.staged_calls
    assert staged._store.direct_calls == direct._store.direct_calls
    # Register only the nonempty tensor, preserving the original per-key retry path.
    assert [size for _, size in staged._store.register_calls] == [8]


def test_oversized_serialized_batch_preserves_one_registration_and_metadata(make_client):
    client = make_client(max_bytes=256)
    values = [{"payload": torch.arange(1000)}, {"payload": torch.arange(2000)}, "small"]
    keys = ["a", "b", "c"]
    meta = client.put(keys, values)
    assert not client._store.staged_calls
    assert client._store.direct_calls == [keys]
    assert len(client._store.register_calls) == 1
    assert client._store.register_calls[0][1] == sum(m["packed_size"] for m in meta)
    assert meta == [{"packed_size": len(client._store.data[key])} for key in keys]
    result = client.get(keys, [None] * len(keys), [None] * len(keys), meta)
    for expected, actual in zip(values[:2], result[:2], strict=True):
        torch.testing.assert_close(actual["payload"], expected["payload"])
    assert result[2] == "small"


@pytest.mark.parametrize("kind", ["tensor", "bytes"])
def test_benchmark_reports_distinct_put_and_get_measurements(make_client, kind):
    config = make_client().config
    args = argparse.Namespace(objects=3, bytes_per_object=100, concurrency=2, kind=kind, warmup=1, iterations=2)
    for enabled in (False, True):
        result = run(config, args, enabled)
        put, get = result["put"]["store"], result["get"]["store"]
        assert result["config"]["local_buffer_staging"]["enabled"] == enabled
        assert (put.get("register_buffer_calls", 0) == 0) == enabled
        assert (get.get("register_buffer_calls", 0) == 0) == enabled
        assert get["batch_get_buffer_calls" if enabled else "batch_get_into_calls"] == 4


@pytest.mark.parametrize("count", [2, 3])
def test_get_batch_limit_preserves_coalesced_direct_fallback(make_client, monkeypatch, count):
    monkeypatch.setattr(mc, "BATCH_SIZE_LIMIT", count)
    client = make_client(max_bytes=512)
    keys = [str(i) for i in range(count + 1)]
    values = [torch.arange(256, dtype=torch.uint8) for _ in keys]
    client.put(keys, values)
    client._store.register_calls.clear()
    result = client.get(keys, [v.shape for v in values], [v.dtype for v in values])
    if count == 2:
        assert sorted(client._store.staged_get_calls) == [keys[:count], keys[count:]]
        assert not client._store.register_calls
    else:
        assert client._store.staged_get_calls == [keys[count:]]
        assert client._store.direct_get_calls == [keys[:count]]
        assert [size for _, size in client._store.register_calls] == [count * 256]
    for expected, actual in zip(values, result, strict=True):
        torch.testing.assert_close(actual, expected)


def test_get_partial_failure_releases_handles_and_budget_before_retry(make_client, monkeypatch):
    client = make_client(max_bytes=512)
    keys = ["a", "b"]
    client.put(keys, [torch.ones(1), torch.ones(1)])
    staged_get = client._store.batch_get_buffer
    direct_get = client._store.batch_get_into
    calls = []

    def partial_get(keys):
        handles = staged_get(keys)
        handles[1] = None
        return handles

    def retry_get(keys, ptrs, sizes):
        assert all(handle() is None for handle in client._store.handles)
        with client._local_buffer_staging.acquire(512):
            calls.append(list(keys))
            if len(calls) == 1:
                direct_get(keys[:1], ptrs[:1], sizes[:1])
                return [sizes[0], -500]
            return direct_get(keys, ptrs, sizes)

    monkeypatch.setattr(client._store, "batch_get_buffer", partial_get)
    monkeypatch.setattr(client._store, "batch_get_into", retry_get)
    result = client.get(keys, [(1,)] * 2, [torch.float32] * 2)
    assert calls == [keys, ["b"]]
    assert not client._store.regions
    for value in result:
        torch.testing.assert_close(value, torch.ones(1))


@pytest.mark.parametrize("failure", ["native", "count", "size", "copy", "direct"])
def test_get_failure_releases_resources(make_client, monkeypatch, failure):
    client = make_client()
    client.put(["a"], [torch.ones(1)])
    store = client._store
    if failure == "native":
        monkeypatch.setattr(store, "batch_get_buffer", Mock(side_effect=RuntimeError("native failure")))
    elif failure == "count":
        original = store.batch_get_buffer
        monkeypatch.setattr(store, "batch_get_buffer", lambda keys: original(keys * 2))
    elif failure == "size":
        store.data["a"] = b"bad size"
    elif failure == "copy":
        monkeypatch.setattr(torch.Tensor, "copy_", Mock(side_effect=RuntimeError("copy failure")))
    else:
        monkeypatch.setattr(store, "batch_get_buffer", Mock(return_value=[None]))
        monkeypatch.setattr(store, "batch_get_into", Mock(return_value=[-500]))
    with pytest.raises(RuntimeError):
        client.get(["a"], [(1,)], [torch.float32])
    assert all(handle() is None for handle in store.handles)
    assert client._local_buffer_staging._in_use_bytes == 0
    assert not store.regions


def test_get_timeout_does_not_submit(make_client):
    client = make_client(max_bytes=256, acquire_timeout_s=0.01)
    client.put(["a"], [torch.ones(1)])
    with client._local_buffer_staging.acquire(256):
        with pytest.raises(TimeoutError, match="staging bytes"):
            client.get(["a"], [(1,)], [torch.float32])
    assert not client._store.staged_get_calls
    assert not client._store.direct_get_calls
    torch.testing.assert_close(client.get(["a"], [(1,)], [torch.float32])[0], torch.ones(1))


def test_get_zero_bytes_bypasses_native_allocator(make_client, monkeypatch):
    client = make_client()
    direct = Mock(return_value=[0])
    monkeypatch.setattr(client._store, "batch_get_into", direct)
    result = client.get(["empty"], [(0, 2)], [torch.int64])
    assert result[0].shape == (0, 2)
    assert not client._store.staged_get_calls
    assert not client._store.register_calls
    direct.assert_called_once()
