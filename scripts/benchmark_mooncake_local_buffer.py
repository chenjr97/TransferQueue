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

"""Run a PUT A/B experiment for local-buffer staging against an existing Mooncake master."""

import argparse
import json
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient


class TimedStore:
    """Count Python-visible payload registrations and synchronous Store calls."""

    def __init__(self, store):
        self.store = store
        self.counts = Counter()
        self.lock = threading.Lock()

    def __getattr__(self, name):
        method = getattr(self.store, name)
        if name not in {
            "register_buffer",
            "unregister_buffer",
            "upsert_batch",
            "batch_upsert_from",
            "batch_get_into",
        }:
            return method

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                with self.lock:
                    self.counts[name + "_calls"] += 1
                    self.counts[name + "_s"] += elapsed

        return timed


def run(config, args, enabled):
    config = {**config, "local_buffer_staging": {**config["local_buffer_staging"], "enabled": enabled}}
    started = time.perf_counter()
    client = MooncakeStoreClient(config)
    setup_s = time.perf_counter() - started
    client._store = TimedStore(client._store)
    prefix = "tq-staging-bench-" + uuid.uuid4().hex
    keys = [[f"{prefix}/{lane}/{i}" for i in range(args.objects)] for lane in range(args.concurrency)]
    # Concurrent direct PUTs must not register overlapping source memory.
    values = [
        [torch.arange(args.bytes_per_object, dtype=torch.uint8) for _ in range(args.objects)]
        for _ in range(args.concurrency)
    ]
    if args.kind == "bytes":
        values = [[{"payload": value} for value in lane_values] for lane_values in values]
    shapes = [value.shape if isinstance(value, torch.Tensor) else None for value in values[0]]
    dtypes = [value.dtype if isinstance(value, torch.Tensor) else None for value in values[0]]
    metadata = [None] * args.concurrency
    process = psutil.Process()

    def call(operation, lane):
        started = time.perf_counter()
        if operation == "put":
            metadata[lane] = client.put(keys[lane], values[lane])
        else:
            client.get(keys[lane], shapes, dtypes, metadata[lane])
        return (time.perf_counter() - started) * 1000

    try:
        cold = {"put_ms": call("put", 0), "get_ms": call("get", 0)}
        for lane in range(1, args.concurrency):
            call("put", lane)
        received = client.get(keys[0], shapes, dtypes, metadata[0])
        for expected, actual in zip(values[0], received, strict=True):
            if args.kind == "bytes":
                expected, actual = expected["payload"], actual["payload"]
            torch.testing.assert_close(actual, expected)
        del received
        wire_bytes = sum(
            value.nbytes if isinstance(value, torch.Tensor) else meta["packed_size"]
            for value, meta in zip(values[0], metadata[0], strict=True)
        )
        output = {
            "mode": "staged_put" if enabled else "direct",
            "config": config,
            "setup_s": setup_s,
            "cold": cold,
        }
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for operation in ("put", "get"):
                for _ in range(args.warmup):
                    list(executor.map(lambda lane, operation=operation: call(operation, lane), range(args.concurrency)))
                client._store.counts.clear()
                samples = []
                rss_before = process.memory_info().rss
                rss_sampled_peak = rss_before
                started = time.perf_counter()
                for _ in range(args.iterations):
                    samples.extend(
                        executor.map(lambda lane, operation=operation: call(operation, lane), range(args.concurrency))
                    )
                    rss_sampled_peak = max(rss_sampled_peak, process.memory_info().rss)
                elapsed = time.perf_counter() - started
                output[operation] = {
                    "p50_ms": float(np.percentile(samples, 50)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "p99_ms": float(np.percentile(samples, 99)),
                    "wire_bytes_per_call": wire_bytes,
                    "throughput_bytes_s": wire_bytes * len(samples) / elapsed,
                    "rss_before": rss_before,
                    "rss_sampled_peak": rss_sampled_peak,
                    "rss_after": process.memory_info().rss,
                    "store": dict(client._store.counts),
                }
        return output
    finally:
        try:
            for lane_keys in keys:
                client.clear(lane_keys)
        finally:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON MooncakeStoreClient config")
    parser.add_argument("--kind", choices=["tensor", "bytes"], default="tensor")
    parser.add_argument("--mode", choices=["both", "direct", "staged"], default="both")
    parser.add_argument("--objects", type=int, default=400)
    parser.add_argument("--bytes-per-object", type=int, default=262144)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.objects, args.bytes_per_object, args.concurrency, args.iterations) <= 0 or args.warmup < 0:
        parser.error("counts and sizes must be positive; warmup must be nonnegative")
    config = json.loads(args.config.read_text())
    if config.get("use_gdr", False):
        parser.error("this benchmark measures CPU staging; set use_gdr=false")
    modes = [False, True] if args.mode == "both" else [args.mode == "staged"]
    for enabled in modes:
        print(
            json.dumps(
                {
                    "workload": vars(args) | {"config": str(args.config)},
                    "result": run(config, args, enabled),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
