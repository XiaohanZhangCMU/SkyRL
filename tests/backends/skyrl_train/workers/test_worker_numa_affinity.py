"""
Tests for NUMA affinity setup fallbacks.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/workers/test_worker_numa_affinity.py
"""

import ctypes.util
from types import SimpleNamespace

from skyrl.backends.skyrl_train.workers import worker as worker_module


def _dummy_actor(local_rank=0):
    return SimpleNamespace(_local_rank=local_rank)


def test_set_numa_affinity_skips_when_library_not_found(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(worker_module, "_SET_AFFINITY", False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(worker_module, "CDLL", lambda _: (_ for _ in ()).throw(AssertionError("CDLL should not run")))

    worker_module.DistributedTorchRayActor._set_numa_affinity(_dummy_actor(), 0)

    assert worker_module._SET_AFFINITY is True


def test_set_numa_affinity_skips_when_library_symbols_are_missing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(worker_module, "_SET_AFFINITY", False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libnuma.so.1")

    class FakeLibNuma:
        def __getattr__(self, name):
            raise AttributeError(f"undefined symbol: {name}")

    monkeypatch.setattr(worker_module, "CDLL", lambda _: FakeLibNuma())

    worker_module.DistributedTorchRayActor._set_numa_affinity(_dummy_actor(), 0)

    assert worker_module._SET_AFFINITY is True
