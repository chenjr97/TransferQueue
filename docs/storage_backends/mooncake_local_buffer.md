# Mooncake local-buffer staging (CPU PUT/GET)

Mooncake allocates and registers a local buffer during `setup`. Its `upsert_batch`
API copies the payload into that buffer before transferring, so a PUT that goes
through it needs no `register_buffer`/`unregister_buffer` pair of its own. This
option routes CPU tensor and serialized-object PUT through `upsert_batch`,
trading one payload memory copy for the per-PUT registration. GET uses
`batch_get_buffer` to receive into the same native pool, copies into owned output
buffers, and releases the native handles before returning the byte budget.

Whether that trade wins depends on the transport and object size. The client
disables staging when the option is omitted; an explicit `enabled: true` enables
both PUT and GET. Measure the tradeoff on the target hardware first.
GDR tensor PUT and GET keep using the CUDA staging path.

## Configuration

Set `local_buffer_staging` in the `MooncakeStore` backend config
(see `transfer_queue/config.yaml`). Both settings are required when
`enabled: true`:

| Setting | Meaning |
| --- | --- |
| `max_bytes` | Shared in-flight PUT/GET byte budget B for one client |
| `acquire_timeout_s` | How long a worker waits for budget before raising `TimeoutError` |

Initialization requires `0 < B <= local_buffer_size`. Workers acquire a whole
worker batch's quota at once and release it before any fallback. Locks protect
accounting only; preparation, copying and transfer run concurrently.

B is **not** a physical occupancy bound. Mooncake's allocator bins, fragmentation
and other Store operations consume additional space, so `B <= local_buffer_size`
does not guarantee an allocation succeeds — leave headroom. B is admission control
that keeps concurrent PUTs and GETs from thrashing the allocator, not a capacity
guarantee.
Serialization buffers, GET outputs and oversized transfers stay outside B.
Separate clients have separate pools and budgets.

## Behavior

- Tensors become contiguous CPU tensors, then `uint8` byte views. This preserves
  arbitrary dtype/shape data, including scalars and bfloat16, without adding a
  metadata prefix. `uint8` is required: the pybind buffer binding takes
  `buffer_info.size` as a byte count, and it is an element count for wider dtypes.
- Serialized objects keep the existing packed wire format and `packed_size`
  metadata and the original key order.
- The existing `BATCH_SIZE_LIMIT` still partitions work by key count. If a worker
  batch's total **aligned** size exceeds B, the whole batch uses the original
  direct registration and `batch_upsert_from`/`batch_get_into` path. CPU staging
  does not split it further, preserving contiguous-region coalescing and batched
  transfer.
- A budget timeout raises `TimeoutError` without submitting that worker batch.
- `upsert_batch` returns one code for the whole batch. A nonzero code releases the
  budget and falls back to direct registration plus the existing per-key retry
  helper, so failures still get per-key diagnostics.
- GET copies raw bytes into the existing dtype-grouped output allocation, so
  tensors and decoded objects remain valid after native buffer reuse and client
  close. Serialization and metadata formats stay unchanged.
- If `batch_get_buffer` returns `None` for any key, GET releases all handles and
  the budget before falling back for the whole worker batch. The existing
  `batch_get_into` helper retains per-key retries and diagnostics. Unexpected
  result counts, payload-size mismatches and exceptions raise after cleanup.
- The native allocator rejects zero-byte objects. A batch containing one uses
  the original direct path, including its per-key retry and error behavior.

Replica selection stays in Mooncake. Failed staged reads fall back to the
existing direct GET path, including disk/offload reads. Disk reads may allocate
local-buffer memory of their own, competing with staged PUT/GET; neither the
budget nor the fallback reserves capacity for that. Concurrent disk/offload
behavior has not been validated.

## Measuring

Save a client config to JSON and point the benchmark at an existing test master:

```bash
python -m scripts.benchmark_mooncake_local_buffer --config /tmp/mooncake-client.json \
  --kind tensor --objects 400 --bytes-per-object 262144 --concurrency 4
python -m scripts.benchmark_mooncake_local_buffer --config /tmp/mooncake-client.json \
  --kind bytes --objects 400 --bytes-per-object 262144 --concurrency 4
```

Run with the intended RDMA protocol and NIC configuration; a TCP run exercises the
backend's local memcpy shortcut and says nothing about registration cost. Each JSON
line reports PUT and GET p50/p95/p99, wire-byte throughput, Python-visible
register/unregister counts and timing, budget snapshots and RSS samples. Staged
GET records `batch_get_buffer` calls; direct GET and fallback record
`batch_get_into` calls. Repeat
`--mode direct` and `--mode staged` in fresh processes, alternating their order, to
separate allocator and cache effects. Vary the workload relative to S to cover
large-batch and oversized-object cases. Worker time sums can exceed wall time, and
completion-time RSS samples can miss transient peaks. Fewer Python registration
calls alone does not imply higher throughput.

## Testing

CPU unit tests use a Store double with real pointer copies:

```bash
python -m pytest -q tests/test_mooncake_local_buffer.py tests/test_mooncake_utils.py
```

Native checks run against an existing test master using the same config format as
the benchmark; without the environment variable the module is skipped:

```bash
TQ_MOONCAKE_TEST_CONFIG=/tmp/mooncake-client.json \
  python -m pytest -q tests/test_mooncake_local_buffer_native.py
```

The native cases cover cross-client round trips, output lifetime, allocator
pressure from externally held handles, fragmentation and coalescing, and
concurrent PUT/GET budgeting and GET fallback under allocator pressure. TCP runs
establish API and allocator behavior only;
zero Python-visible payload registration calls do not establish native MR
lifetime or performance. Disk/offload concurrency and RDMA fault injection also
require the corresponding backend environment and hardware.
