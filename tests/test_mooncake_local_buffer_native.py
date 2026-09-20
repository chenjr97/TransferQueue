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

"""Opt-in native Store checks; TQ_MOONCAKE_TEST_CONFIG points to client-config JSON."""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.benchmark_mooncake_local_buffer import TimedStore
from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

_CONFIG_PATH = os.environ.get("TQ_MOONCAKE_TEST_CONFIG")
if not _CONFIG_PATH:
    pytest.skip("set TQ_MOONCAKE_TEST_CONFIG to run against a native Store", allow_module_level=True)


@pytest.fixture
def native_clients():
    config = json.loads(Path(_CONFIG_PATH).read_text())
    prefix = "tq-native-staging-" + uuid.uuid4().hex
    clients, keys = [], []

    def make(enabled=True):
        client = MooncakeStoreClient(
            {
                **config,
                "use_gdr": False,
                "global_segment_size": 32 * 1024 * 1024,
                "local_buffer_size": 64 * 1024,
                "local_buffer_staging": {
                    "enabled": enabled,
                    "max_bytes": 48 * 1024,
                    "batch_bytes": 32 * 1024,
                    "acquire_timeout_s": 2.0,
                },
            }
        )
        client._store = TimedStore(client._store)
        clients.append(client)
        return client

    def new_keys(count):
        batch = [f"{prefix}/{len(keys) + i}" for i in range(count)]
        keys.extend(batch)
        return batch

    yield make, new_keys
    for client in clients:
        if client._store is not None:
            try:
                client.clear(keys)
            finally:
                client.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_native_cross_client_roundtrip_and_output_lifetime(native_clients, enabled):
    make, new_keys = native_clients
    producer, consumer = make(enabled=enabled), make()
    values = [
        torch.arange(12, dtype=torch.float64).reshape(3, 4).t(),
        torch.tensor(7, dtype=torch.int64),
        torch.arange(11, dtype=torch.bfloat16),
        {"tensor": torch.arange(6).reshape(2, 3).t(), "array": np.arange(8)},
        torch.arange(65536, dtype=torch.uint8),
    ]
    keys = new_keys(len(values))
    meta = producer.put(keys[:-1], values[:-1])
    if enabled:
        assert producer._store.counts.get("register_buffer_calls", 0) == 0
    meta.extend(producer.put(keys[-1:], values[-1:]))
    actual = consumer.get(
        keys,
        [v.shape if isinstance(v, torch.Tensor) else None for v in values],
        [v.dtype if isinstance(v, torch.Tensor) else None for v in values],
        meta,
    )
    # Reuse and overwrite the entire native pool before checking the returned data.
    probe = consumer._store.batch_get_buffer([keys[-1]])
    try:
        assert probe[0] is not None
        memoryview(probe[0]).cast("B")[:] = bytes([165]) * 65536
    finally:
        probe.clear()
    producer.clear(keys)
    consumer.close()
    producer.close()
    for expected, result in zip(values, actual, strict=True):
        if isinstance(expected, dict):
            torch.testing.assert_close(result["tensor"], expected["tensor"])
            np.testing.assert_array_equal(result["array"], expected["array"])
        else:
            torch.testing.assert_close(result, expected)


def test_native_external_occupancy_fragmentation_and_coalescing(native_clients):
    make, new_keys = native_clients
    client = make()
    keys = new_keys(8)
    client.put(keys, [torch.zeros(8192, dtype=torch.uint8) for _ in keys])
    replicas = client._store.batch_get_replica_desc(keys)
    assert set(replicas) == set(keys)
    assert all(descs and all(desc.is_memory_replica() for desc in descs) for descs in replicas.values())
    handles = client._store.batch_get_buffer(keys)
    try:
        assert len(handles) == 8 and all(handle is not None for handle in handles)
        # Hold all 64 KiB outside TQ's budget; a logically admitted PUT must fall back.
        client.put(new_keys(1), [torch.ones(8192, dtype=torch.uint8)])
        # Free 32 KiB in separated 8 KiB holes: a 32 KiB allocation still cannot fit.
        for i in range(0, 8, 2):
            handles[i] = None
        client.put(new_keys(1), [torch.ones(32768, dtype=torch.uint8)])
    finally:
        handles.clear()
    # Adjacent frees coalesce; the same-sized PUT now succeeds without registration.
    client._store.counts.clear()
    client.put(new_keys(1), [torch.ones(32768, dtype=torch.uint8)])
    assert client._store.counts.get("register_buffer_calls", 0) == 0


def test_native_concurrent_puts_share_budget(native_clients):
    make, new_keys = native_clients
    client = make()
    batches = [new_keys(4) for _ in range(4)]
    values = [torch.arange(8192, dtype=torch.uint8) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda keys: client.put(keys, values), batches))
    assert client._store.counts.get("register_buffer_calls", 0) == 0
    for keys in batches:
        result = client.get(keys, [v.shape for v in values], [v.dtype for v in values])
        for expected, actual in zip(values, result, strict=True):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_zero_bytes_preserve_error_without_registration(native_clients, monkeypatch, enabled):
    from transfer_queue.storage.clients import mooncake_client

    monkeypatch.setattr(mooncake_client, "RETRY_DELAY_SECONDS", 0)
    make, new_keys = native_clients
    client = make(enabled=enabled)
    with pytest.raises(RuntimeError, match="error codes.*-600"):
        client.put(new_keys(1), [torch.empty(0)])
    assert client._store.counts.get("register_buffer_calls", 0) == 0
