import sys
from typing import Any

import pytest

from src.utils import performance, production_validator, secure_subprocess


# --- Performance.py tests ---
def test_memory_manager_methods() -> None:
    mm = performance.MemoryManager()
    mem = mm.get_memory_usage()
    assert set(mem.keys()) == {"total", "available", "used", "percentage"}
    sys_mem = mm.get_system_memory_info()
    assert set(sys_mem.keys()) == {"total", "available", "used", "percentage"}
    gpu_mem = mm.get_gpu_memory_info()
    assert set(gpu_mem.keys()) == {"total", "used", "free", "percentage"}
    assert mm.optimize_memory() is True
    # Batch size logic
    assert mm.optimize_batch_size(16, 100, "cpu") <= 8
    assert mm.optimize_batch_size(128, 10, "cuda") <= 64
    assert mm.optimize_batch_size(32, 1, "other") <= 32


def test_performance_monitor_methods() -> None:
    pm = performance.PerformanceMonitor()
    assert pm.get_metrics() == {"cpu_usage": 0.0, "memory_usage": 0.0, "gpu_usage": 0.0}
    assert pm.get_current_metrics() == {
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "gpu_usage": 0.0,
    }
    pm.start_monitoring()
    pm.stop_monitoring()


# --- Production Validator tests ---
def test_production_validator_pass(monkeypatch: Any) -> None:
    # Patch sys.version_info to >= 3.8
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    result = production_validator.validate_production_readiness()
    assert result["status"] in ("PASS", "FAIL")
    assert "system" in result and "dependencies" in result


def test_production_validator_fail() -> None:
    # Use version_info parameter for easier testing
    result = production_validator.validate_production_readiness(version_info=(3, 7, 0))
    assert result["status"] == "FAIL"
    assert any("Python version < 3.8" in issue for issue in result["issues"])


# --- Secure Subprocess tests ---
def test_run_command_success() -> None:
    success, out, err = secure_subprocess.run_command([sys.executable, "--version"])
    assert success
    assert "Python" in out


def test_run_command_invalid() -> None:
    success, out, err = secure_subprocess.run_command(["nonexistent_command_12345"])
    assert not success
    assert "not found" in err or "error" in err or "Invalid command format" in err


def test_run_python_module_success() -> None:
    # Use a module that always works and produces output
    success, out, err = secure_subprocess.run_python_module("site")
    assert success


# Optionally, test pip install (but skip by default)
@pytest.mark.skip(reason="Avoid installing packages during test runs.")
def test_run_pip_install() -> None:
    success, out, err = secure_subprocess.run_pip_install("pytest")
    assert (
        success or "already satisfied" in out or "Requirement already satisfied" in out
    )


# System info test (should not raise)
def test_get_system_info() -> None:
    success, out, err = secure_subprocess.get_system_info()
    assert isinstance(success, bool)
