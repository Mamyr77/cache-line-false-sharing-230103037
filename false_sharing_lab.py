import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Твой точный GitHub-ник
GIT_USERNAME = "Mamyr77"

NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    affinity_supported = hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
    available_cores = sorted(list(os.sched_getaffinity(0))) if affinity_supported else list(range(logical_cores))

    return {
        "git_username": git_user.strip(),
        "logical_cores": logical_cores,
        "cache_line_size_bytes": line_size,
        "cpu_signature": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "target_cores": available_cores[:NUM_THREADS],
    }


def allocate_aligned_uint64(num_elements: int, alignment: int = 64) -> np.ndarray:
    """Выделение 64-байт выровненного массива uint64 для точного попадания в границы кэш-линий."""
    raw = np.zeros(num_elements * 8 + alignment, dtype=np.uint8)
    addr = raw.ctypes.data
    offset = (alignment - (addr % alignment)) % alignment
    aligned = np.frombuffer(raw.data, dtype=np.uint64, count=num_elements, offset=offset)
    aligned.fill(0)
    return aligned


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel (In-Place Memory Stores)
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    """
    Выполняет чтение-модификацию-запись (RMW) напрямую в buffer[slot] на каждой итерации.
    Зависимость по данным и запись в память на каждом шаге вынуждают CPU
    выдавать реальные L1 store-инструкции (и вызывать RFO при False Sharing).
    """
    state = uint64(thread_seed) | uint64(1)
    for _ in range(iters):
        # Xorshift-шаг зависит от текущего значения в памяти buffer[slot]
        val = buffer[slot]
        state ^= (state << uint64(13))
        state ^= (state >> uint64(7))
        state ^= (state << uint64(17))
        buffer[slot] = val + (state & uint64(0xFF)) + uint64(1)


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    """
    Контрольное ядро: накапливает результат в локальном регистре процессора
    и записывает в разделяемый массив buffer[slot] ровно 1 раз в конце.
    """
    state = uint64(thread_seed) | uint64(1)
    acc = buffer[slot]
    for _ in range(iters):
        state ^= (state << uint64(13))
        state ^= (state >> uint64(7))
        state ^= (state << uint64(17))
        acc += (state & uint64(0xFF)) + uint64(1)
    buffer[slot] = acc


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def get_context_switches():
    if resource is None:
        return {"voluntary": 0, "involuntary": 0}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "voluntary": int(usage.ru_nvcsw),
        "involuntary": int(usage.ru_nivcsw),
    }


def run_kernel_benchmark(kernel_fn, stride: int, ctx: dict):
    buf_len = NUM_THREADS * max(stride, 16) + 16
    buffer = allocate_aligned_uint64(buf_len, alignment=64)

    base_seed = ctx["instance_seed"]
    target_cores = ctx["target_cores"]
    affinity_supported = ctx["affinity_supported"]

    barrier = threading.Barrier(NUM_THREADS)
    threads = []

    def thread_runner(tid: int):
        if affinity_supported:
            try:
                os.sched_setaffinity(0, {target_cores[tid]})
            except OSError:
                pass
        slot = tid * stride
        t_seed = (int(base_seed) ^ ((tid + 1) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
        barrier.wait()
        kernel_fn(buffer, slot, ITERATIONS_PER_THREAD, np.uint64(t_seed))

    for tid in range(NUM_THREADS):
        t = threading.Thread(target=thread_runner, args=(tid,))
        threads.append(t)

    cs_before = get_context_switches()
    t_start = time.perf_counter_ns()

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    t_end = time.perf_counter_ns()
    cs_after = get_context_switches()

    elapsed_ns = t_end - t_start
    total_updates = NUM_THREADS * ITERATIONS_PER_THREAD
    ns_per_update = elapsed_ns / total_updates

    final_values = [int(buffer[tid * stride]) for tid in range(NUM_THREADS)]
    return {
        "elapsed_ns": int(elapsed_ns),
        "elapsed_ms": round(elapsed_ns / 1e6, 3),
        "ns_per_update": round(ns_per_update, 4),
        "voluntary_csw": cs_after["voluntary"] - cs_before["voluntary"],
        "involuntary_csw": cs_after["involuntary"] - cs_before["involuntary"],
        "final_values": final_values,
    }


def run_stride_benchmark(ctx: dict) -> dict:
    # Прогрев JIT-компиляции Numba перед замерами
    warm_buf = allocate_aligned_uint64(32, alignment=64)
    worker_strided_accumulate(warm_buf, 0, 100, ctx["instance_seed"])
    worker_private_accumulate(warm_buf, 0, 100, ctx["instance_seed"])

    results = {}
    for stride in (1, 4, 8, 16):
        results[f"stride_{stride}"] = run_kernel_benchmark(worker_strided_accumulate, stride, ctx)

    results["private_baseline"] = run_kernel_benchmark(worker_private_accumulate, 8, ctx)
    return results


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def audit_llvm_and_parity(ctx: dict, results: dict):
    # 1. Проверка точного побитового совпадения (Parity) между всеми режимами
    ref_vals = results["private_baseline"]["final_values"]
    parity_ok = True
    for name, data in results.items():
        if data["final_values"] != ref_vals:
            parity_ok = False
            raise AssertionError(f"Parity mismatch in {name}: {data['final_values']} != {ref_vals}")

        # 2. Инспекция LLVM IR на наличие нативных 64-битных store-инструкций и отсутствие pyobject-отката
        llvm_dict = worker_strided_accumulate.inspect_llvm()
        llvm_ir = next(iter(llvm_dict.values()))
        # Выделяем тело нативного ядра (до служебной CPython-обертки)
        # Берем строго первую функцию из LLVM IR (само нативное ядро без CPython-обертки)
        # Выделяем первую функцию define ... { ... } (само нативное ядро без CPython-обертки)
        func_start = llvm_ir.find("define ")
        func_end = llvm_ir.find("\n}\n", func_start)
        native_ir = llvm_ir[func_start:func_end + 3] if func_start != -1 and func_end != -1 else llvm_ir

        i64_store_count = native_ir.count("store i64") + native_ir.count("store atomic i64")
        has_pyobject = ("pyobject" in native_ir.lower()) or ("numba.object" in native_ir.lower())
        llvm_audit_passed = (i64_store_count > 0) and (not has_pyobject)

        assert llvm_audit_passed, f"LLVM IR audit failed (stores={i64_store_count}, pyobj={has_pyobject})."

    assert llvm_audit_passed, "LLVM IR audit failed: missing native 'store i64' or found Python object fallback."

    s1_ns = results["stride_1"]["elapsed_ns"]
    s8_ns = results["stride_8"]["elapsed_ns"]
    slowdown_s1_vs_s8 = round(s1_ns / s8_ns, 3)

    print("=== CACHE LINE & FALSE SHARING TELEMETRY ===")
    print(f"Git Handle          : {ctx['git_username']}")
    print(f"CPU Signature       : {ctx['cpu_signature']}")
    print(f"Logical Cores       : {ctx['logical_cores']} (Using {NUM_THREADS} threads)")
    print(f"L1 Cache Line Size  : {ctx['cache_line_size_bytes']} bytes")
    print(f"Instance Seed       : {ctx['instance_seed']}")
    print(f"LLVM 'store i64'    : {i64_store_count} (pyobject fallback: {has_pyobject})")
    print(f"State Parity Passed : {parity_ok}")
    print(f"Slowdown (S1 vs S8) : {slowdown_s1_vs_s8}x\n")

    print(f"{'Layout / Mode':<20} | {'Elapsed (ms)':>12} | {'ns / update':>12} | {'Vol CSW':>8} | {'Invol CSW':>9}")
    print("-" * 73)
    for key in ("stride_1", "stride_4", "stride_8", "stride_16", "private_baseline"):
        r = results[key]
        print(f"{key:<20} | {r['elapsed_ms']:>12.3f} | {r['ns_per_update']:>12.4f} | {r['voluntary_csw']:>8d} | {r['involuntary_csw']:>9d}")

    manifest = {
        "git_username": ctx["git_username"],
        "cpu_signature": ctx["cpu_signature"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size_bytes": ctx["cache_line_size_bytes"],
        "instance_seed": str(ctx["instance_seed"]),
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "slowdown_stride1_vs_stride8": slowdown_s1_vs_s8,
        "llvm_audit": {
            "i64_store_count": i64_store_count,
            "pyobject_fallback": has_pyobject,
            "llvm_audit_passed": llvm_audit_passed,
        },
        "benchmarks": {
            k: {
                "elapsed_ns": v["elapsed_ns"],
                "elapsed_ms": v["elapsed_ms"],
                "ns_per_update": v["ns_per_update"],
                "voluntary_csw": v["voluntary_csw"],
                "involuntary_csw": v["involuntary_csw"],
            }
            for k, v in results.items()
        },
        "verification_passed": bool(parity_ok and llvm_audit_passed),
    }

    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nSaved cache_lab_manifest.json successfully.")


if __name__ == "__main__":
    context = init_system_context(GIT_USERNAME)
    bench_results = run_stride_benchmark(context)
    audit_llvm_and_parity(context, bench_results)