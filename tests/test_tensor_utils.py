# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Unit tests for transfer_queue.utils.tensor_utils."""

import gc
import mmap
import sys
import weakref

import pytest
import torch

from transfer_queue.utils.tensor_utils import (
    allocate_empty_tensors,
    compute_stride,
    get_nbytes,
    merge_contiguous_memory,
)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mmap advice")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float64])
def test_mmap_group_views_keep_owner_alive(monkeypatch, dtype):
    owners = []
    original = torch.frombuffer

    def capture(owner, **kwargs):
        owners.append(weakref.ref(owner))
        return original(owner, **kwargs)

    monkeypatch.setattr(torch, "frombuffer", capture)
    count = 2 * 1024**2 // torch.empty((), dtype=dtype).element_size()
    tensors, pointers, regions, sizes = allocate_empty_tensors(
        [dtype, dtype], [(count // 2,), (count // 2,)], allocation="mmap_huge"
    )
    assert len(owners) == len(regions) == 1
    assert regions[0] % (2 * 1024**2) == 0
    assert sizes == [2 * 1024**2]
    assert pointers[1] - pointers[0] == 1024**2
    tensors[0].fill_(3)
    tensors[1].fill_(7)
    view = tensors[1][::2]
    del tensors
    gc.collect()
    assert owners[0]() is not None
    assert torch.all(view == 7)
    del view
    gc.collect()
    assert owners[0]() is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mmap advice")
def test_mmap_empty_and_small_groups_use_torch(monkeypatch):
    monkeypatch.setattr(torch, "frombuffer", lambda *a, **kw: pytest.fail("small group used mmap"))
    tensors, _, _, sizes = allocate_empty_tensors([torch.float32, torch.int64], [(0,), ()], allocation="mmap_huge")
    assert [t.shape for t in tensors] == [torch.Size([0]), torch.Size([])]
    assert sizes == [0, 8]


@pytest.mark.parametrize("invalid_allocation", ["typo", "mmap_nohuge"])
def test_allocation_validation(monkeypatch, invalid_allocation):
    from transfer_queue.utils import tensor_utils

    with pytest.raises(ValueError, match="Unknown buffer allocation"):
        allocate_empty_tensors([], [], allocation=invalid_allocation)
    monkeypatch.setattr(tensor_utils.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="require Linux"):
        allocate_empty_tensors([], [], allocation="mmap_huge")
    assert allocate_empty_tensors([], []) == ([], [], [], [])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mmap advice")
def test_mmap_huge_does_not_require_nohuge_advice(monkeypatch):
    monkeypatch.delattr(mmap, "MADV_NOHUGEPAGE", raising=False)
    tensors, _, _, sizes = allocate_empty_tensors([torch.uint8], [(2 * 1024**2,)], allocation="mmap_huge")
    tensors[0].fill_(7)
    assert sizes == [2 * 1024**2]
    assert torch.all(tensors[0] == 7)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mmap advice")
def test_madvise_failure_closes_mapping(monkeypatch):
    owners = []

    class FailingMapping(mmap.mmap):
        def madvise(self, *args):
            owners.append(self)
            raise OSError("advice failed")

    monkeypatch.setattr(mmap, "mmap", FailingMapping)
    with pytest.raises(OSError, match="advice failed"):
        allocate_empty_tensors([torch.uint8], [(2 * 1024**2,)], allocation="mmap_huge")
    assert owners[0].closed


class TestComputeStride:
    """Tests for compute_stride."""

    def test_3d(self):
        assert compute_stride((2, 3, 4)) == (12, 4, 1)

    def test_1d(self):
        assert compute_stride((5,)) == (1,)

    def test_scalar(self):
        assert compute_stride(()) == ()

    def test_2d(self):
        assert compute_stride((3, 5)) == (5, 1)


class TestGetNbytes:
    """Tests for get_nbytes."""

    def test_basic(self):
        dtypes = [torch.float32, torch.int32]
        shapes = [(2, 3), (4,)]
        result = get_nbytes(dtypes, shapes)
        assert result == [2 * 3 * 4, 4 * 4]  # float32=4, int32=4

    def test_scalar(self):
        dtypes = [torch.float64]
        shapes = [()]
        result = get_nbytes(dtypes, shapes)
        assert result == [8]  # scalar = 1 element

    def test_list_shape(self):
        dtypes = [torch.float32]
        shapes = [[]]  # list instead of tuple
        result = get_nbytes(dtypes, shapes)
        assert result == [4]

    def test_mixed_dtypes(self):
        dtypes = [torch.float16, torch.float32, torch.int64]
        shapes = [(10,), (10,), (10,)]
        result = get_nbytes(dtypes, shapes)
        assert result == [10 * 2, 10 * 4, 10 * 8]


class TestAllocateEmptyTensors:
    """Tests for allocate_empty_tensors."""

    def test_basic(self):
        dtypes = [torch.float32, torch.float32, torch.int32]
        shapes = [(2, 3), (4,), (5,)]
        tensors, ptrs, region_ptrs, region_sizes = allocate_empty_tensors(dtypes, shapes)

        assert len(tensors) == 3
        assert len(ptrs) == 3
        assert len(region_ptrs) == 2  # float32 group + int32 group
        assert len(region_sizes) == 2

        # Same dtype tensors share the same underlying storage
        assert tensors[0].untyped_storage().data_ptr() == region_ptrs[0]
        assert tensors[1].untyped_storage().data_ptr() == region_ptrs[0]
        assert tensors[2].untyped_storage().data_ptr() == region_ptrs[1]

        # Shapes are correct
        assert list(tensors[0].shape) == [2, 3]
        assert list(tensors[1].shape) == [4]
        assert list(tensors[2].shape) == [5]

    def test_scalar(self):
        dtypes = [torch.float32, torch.int32]
        shapes = [(), ()]
        tensors, ptrs, region_ptrs, region_sizes = allocate_empty_tensors(dtypes, shapes)

        assert len(tensors) == 2
        assert tensors[0].numel() == 1
        assert tensors[1].numel() == 1
        assert len(region_ptrs) == 2

    def test_empty(self):
        result = allocate_empty_tensors([], [])
        assert result == ([], [], [], [])

    def test_regions_complex(self):
        """Mixed dtypes and shapes: verify region counts, sizes, and per-tensor offsets."""
        dtypes = [
            torch.float32,  # group 0: (2, 3) -> 6 elements
            torch.int32,  # group 1: (4,) -> 4 elements
            torch.float32,  # group 0: scalar -> 1 element
            torch.float64,  # group 2: (2, 2) -> 4 elements
            torch.int32,  # group 1: (3, 2) -> 6 elements
        ]
        shapes = [(2, 3), (4,), (), (2, 2), (3, 2)]
        tensors, ptrs, region_ptrs, region_sizes = allocate_empty_tensors(dtypes, shapes)

        # 3 dtype groups in insertion order: float32, int32, float64
        assert len(region_ptrs) == 3
        assert len(region_sizes) == 3
        assert len(set(region_ptrs)) == 3  # distinct allocations

        # float32 region: 6 + 1 = 7 elements * 4 bytes = 28 bytes
        assert region_sizes[0] == 7 * 4
        # int32 region: 4 + 6 = 10 elements * 4 bytes = 40 bytes
        assert region_sizes[1] == 10 * 4
        # float64 region: 4 elements * 8 bytes = 32 bytes
        assert region_sizes[2] == 4 * 8

        # Per-tensor ptrs must lie inside their respective regions
        # tensor 0 (float32, shape (2,3), offset 0)
        assert ptrs[0] == region_ptrs[0]
        # tensor 1 (int32, shape (4,), offset 0)
        assert ptrs[1] == region_ptrs[1]
        # tensor 2 (float32, scalar, offset 6)
        assert ptrs[2] == region_ptrs[0] + 6 * 4
        # tensor 3 (float64, shape (2,2), offset 0)
        assert ptrs[3] == region_ptrs[2]
        # tensor 4 (int32, shape (3,2), offset 4)
        assert ptrs[4] == region_ptrs[1] + 4 * 4


class TestMergeContiguousMemory:
    """Tests for merge_contiguous_memory."""

    def test_basic_merge(self):
        ptrs = [0, 10, 30]
        sizes = [10, 20, 10]
        merged_ptrs, merged_sizes = merge_contiguous_memory(ptrs, sizes)
        # 0+10=10 (contiguous with 10), 10+20=30 (contiguous with 30) -> all merge into [0]
        assert merged_ptrs == [0]
        assert merged_sizes == [40]

    def test_no_contiguous(self):
        ptrs = [0, 100, 200]
        sizes = [50, 50, 50]
        merged_ptrs, merged_sizes = merge_contiguous_memory(ptrs, sizes)
        assert merged_ptrs == [0, 100, 200]
        assert merged_sizes == [50, 50, 50]

    def test_unsorted_input(self):
        ptrs = [100, 0, 50]
        sizes = [50, 50, 50]
        merged_ptrs, merged_sizes = merge_contiguous_memory(ptrs, sizes)
        # After sorting: 0, 50, 100; all contiguous -> merge into [0]
        assert merged_ptrs == [0]
        assert merged_sizes == [150]

    def test_single_region(self):
        ptrs = [10]
        sizes = [100]
        merged_ptrs, merged_sizes = merge_contiguous_memory(ptrs, sizes)
        assert merged_ptrs == [10]
        assert merged_sizes == [100]

    def test_empty(self):
        assert merge_contiguous_memory([], []) == ([], [])

    def test_mismatched_lengths_both_empty_not_triggered(self):
        # If one is empty and other is not, should raise ValueError
        with pytest.raises(ValueError, match="ptrs and sizes must have the same length"):
            merge_contiguous_memory([], [10])

        with pytest.raises(ValueError, match="ptrs and sizes must have the same length"):
            merge_contiguous_memory([0], [])

    def test_three_continuous(self):
        ptrs = [0, 10, 20]
        sizes = [10, 10, 10]
        merged_ptrs, merged_sizes = merge_contiguous_memory(ptrs, sizes)
        assert merged_ptrs == [0]
        assert merged_sizes == [30]
