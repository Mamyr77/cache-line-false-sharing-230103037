# Analytical & Forensic Engineering Report: Cache Line & False Sharing Mechanics

**Student Git Handle:** `Mamyr77`  
**Host Architecture:** `AMD64::Intel64 Family 6 Model 154 Stepping 3, GenuineIntel::12` (12 logical cores, 64-byte L1D cache line)  
**Instance Seed:** `12902023850628446492`  
**Workload:** 4 threads $\times$ 60,000,000 updates ($2.4 \times 10^8$ total memory operations)

---

### Question 1 (MESI Protocol State Transitions)
Under `stride = 1` (8-byte element spacing), `buffer[0]` (offset `0B`) and `buffer[1]` (offset `8B`) reside within the exact same 64-byte L1 data cache line (`CacheLineIndex = 0`). When Core 0 and Core 1 concurrently execute read-modify-write stores to `buffer[0]` and `buffer[1]`, the hardware cache coherency controller enforces the **MESI** protocol at 64-byte line granularity:

1. **Initial Read / Shared State (`S`):** Both Core 0 and Core 1 initially load the cache line containing `buffer[0..7]` into their private L1 data caches in the **Shared (`S`)** state.
2. **Core 0 Store & RFO Broadcast (`S -> M` on Core 0, `S -> I` on Core 1):** Before Core 0 can commit its store to `buffer[0]`, it must acquire exclusive ownership. Core 0 broadcasts a **Read-For-Ownership (RFO)** transaction on the interconnect/ring bus. Core 1 snoops this RFO, transitions its entire 64-byte copy of the line to **Invalid (`I`)**, and sends an invalidation acknowledgment. Core 0 transitions its line to **Modified (`M`)** and writes the new value of `buffer[0]`.
3. **Core 1 Immediate Store (`I -> M` on Core 1, `M -> I` on Core 0):** Immediately afterward, Core 1 attempts to commit its store to `buffer[1]`. Because its local copy of the cache line is now in the **Invalid (`I`)** state (a coherency write miss), Core 1 cannot drain the entry from its Store Buffer into L1D. Core 1 must issue its own **RFO** request over the interconnect.
4. **Modified Line Writeback / Intervention:** Core 0 snoops Core 1's RFO while holding the line in **Modified (`M`)** state. Core 0 must flush/transfer the dirty 64-byte line across the interconnect (and update L3/LLC) and downgrade its own copy to **Invalid (`I`)**. Core 1 receives the line, transitions to **Modified (`M`)**, and finally drains its store to `buffer[1]`.

**Why the Store Buffer Stalls:** Modern out-of-order cores (including Intel Alder Lake Family 6 Model 154) buffer speculative stores in a **Store Buffer (SB)** to hide L1 latency, allowing subsequent instructions to execute out of order while waiting for an RFO. However, because our kernel enforces a tight loop-carried data dependency (`val = buffer[slot]` followed by dependent arithmetic and `buffer[slot] = ...`), subsequent iterations cannot retire or forward valid data cleanly once the cache line is invalidated, and the Store Buffer rapidly fills up waiting for RFO ownership acknowledgments—stalling the pipeline on cross-core coherency latency.

---

### Question 2 (Quantitative Slowdown & Coherency Overhead)
From the executed benchmark telemetry (`cache_lab_manifest.json`), the measured wall-clock durations and per-update latencies across $2.4 \times 10^8$ total updates (4 threads $\times$ 60,000,000 iterations) are:

| Benchmark Regime | Element Stride | Byte Spacing | Elapsed Time (ns) | Elapsed Time (ms) | Avg Latency (ns / update) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`stride_1` (Contiguous)** | 1 | 8 B | `101,532,000` | `101.532 ms` | `0.4231 ns` |
| **`stride_4` (Half-Line)** | 4 | 32 B | `96,053,000` | `96.053 ms` | `0.4002 ns` |
| **`stride_8` (Line-Padded)** | 8 | 64 B | `97,304,000` | `97.304 ms` | `0.4054 ns` |
| **`stride_16` (Prefetch-Safe)** | 16 | 128 B | `91,772,000` | `91.772 ms` | `0.3824 ns` |
| **`private_baseline` (Register)** | 8 (once) | 64 B | `91,515,000` | `91.515 ms` | `0.3813 ns` |

- **Slowdown Factor (`stride_1` relative to `stride_8`):**
  $$\text{Slowdown} = \frac{101.532\text{ ms}}{97.304\text{ ms}} = \mathbf{1.043\times} \quad (+4.34\% \text{ overhead})$$
  *(Note: On modern hybrid/ring-bus Intel architectures with store-to-load forwarding and fast L3 snoop filters, `stride_1` vs `stride_16` exhibits a **$1.106\times$** slowdown, adding `0.0407 ns` of coherency/prefetch contention per memory operation).*
- **Coherency vs. L1 Store-Forwarding Breakdown:** Comparing `private_baseline` (`0.3813 ns/update`, pure ALU + register accumulation) against `stride_16` (`0.3824 ns/update`) shows that when cache lines are completely isolated from both false sharing and adjacent-line prefetching, L1 store-to-load forwarding hides virtually all L1D memory access latency (`+0.0011 ns/update`). Moving from `stride_16` (`0.3824 ns/update`) to `stride_1` (`0.4231 ns/update`) exposes the true cross-core MESI RFO and prefetcher contention penalty.

---

### Question 3 (L2 Spatial Prefetcher & 128-Byte Pairing)
In my benchmark run, **`stride_8` (64-byte spacing)** took **`97.304 ms` (`0.4054 ns/update`)**, whereas **`stride_16` (128-byte spacing)** dropped to **`91.772 ms` (`0.3824 ns/update`)**—an improvement of **`5.532 ms` (`5.7%` faster)**, bringing `stride_16` within `0.257 ms` of the register-privatized baseline (`91.515 ms`).

**Hardware Mechanism (128-Byte Sector Pairing):**
Even though the L1 data cache coherency granularity is 64 bytes (`L1 Cache Line Size : 64 bytes`), Intel processors include an **L2 Spatial Prefetcher (Adjacent Cache Line Prefetcher)** that pairs consecutive even and odd 64-byte cache lines into aligned **128-byte sectors** (Lines `0 & 1`, Lines `2 & 3`, etc.).
- Under **`stride_8` (64-byte padding)**, Thread 0 writes to Line 0 (`offset 0B`) and Thread 1 writes to Line 1 (`offset 64B`), which belong to the *same* 128-byte aligned sector. Similarly, Thread 2 (Line 2, `128B`) and Thread 3 (Line 3, `192B`) share the second 128-byte sector.
- When Core 0 triggers a miss or coherence transition on Line 0, the L2 spatial prefetcher speculatively fetches its buddy Line 1 into Core 0's L2 cache in Shared (`S`) state—even though Core 1 is actively modifying Line 1 in Modified (`M`) state. This causes **secondary coherency invalidations and snoop traffic** between cores sharing a 128-byte sector.
- Under **`stride_16` (128-byte padding)**, threads write to offsets `0B`, `128B`, `256B`, and `384B` (Lines `0, 2, 4, 6`). Each thread occupies its own independent 128-byte sector, completely eliminating adjacent-line prefetcher cross-talk.

---

### Question 4 (Compiler Register Promotion vs. Volatile Memory Stores)
Inspecting `worker_strided_accumulate.inspect_llvm()` confirms that Numba compiled the loop into native LLVM IR containing **`store i64`** (`i64_store_count = 1`) with **`pyobject fallback: False`**, proving that every loop iteration executes a genuine 64-bit machine memory store rather than a Python object call.

**Why Compilers Attempt Register Hoisting & How Our Loop Prevents It:**
In standard compiler optimization passes (such as LLVM's `LICM` — Loop-Invariant Code Motion and `mem2reg` / Scalar Replacement of Aggregates), if a compiler proves that a memory pointer (`buffer[slot]`) is not aliased within a single-threaded loop, it promotes `buffer[slot]` into a CPU register before the loop, accumulates all 60,000,000 additions in that register, and emits a single `store i64` after the loop exits (which is exactly what `worker_private_accumulate` does).

In `worker_strided_accumulate`, register promotion is prevented because:
1. `buffer` is passed as a mutable external NumPy array pointer (`uint64*`) into a `@njit(nogil=True)` function without `noalias` guarantees across iterations.
2. Each iteration explicitly performs an in-place load (`val = buffer[slot]`) followed by a dependent computation (`buffer[slot] = val + (state & 0xFF) + 1`) directly back to the memory address. Because LLVM cannot prove that another thread will not observe or modify `buffer[slot]` between iterations across a `nogil` boundary, LLVM preserves the per-iteration `load i64` and `store i64` instructions inside the loop body, forcing the CPU to execute 60,000,000 L1 memory stores per thread.

---

### Question 5 (CPython GIL Masking vs. `nogil=True` Execution)
If the identical `stride = 1` experiment were executed using standard CPython bytecode threads without `@njit(nogil=True)`, **hardware false sharing would completely disappear**, and multi-threaded execution would actually become **slower** than single-threaded execution as thread count increases:

1. **GIL Serialization Prevents Simultaneous Core Stores:** CPython enforces a process-wide mutex—the **Global Interpreter Lock (GIL)**—so only one thread can execute Python bytecode at any instant. Even across 12 logical CPU cores, Core 0 and Core 1 never execute stores to `buffer[0]` and `buffer[1]` simultaneously; instead, one thread holds the cache line exclusively in `M`/`E` state for an entire OS scheduling quantum (~5 ms) before releasing the GIL, eliminating per-iteration RFO ping-pong.
2. **GIL Lock Contention vs. Cache-Line Contention:** With `@njit(nogil=True)`, all 4 threads execute native machine instructions truly in parallel across 4 physical cores (`0` voluntary/involuntary context switches during the hot loop), making **hardware cache-line coherency (MESI RFO bus traffic)** the primary bottleneck (`0.4231 ns/update`). Conversely, pure CPython threads suffer from **software OS-level lock contention**—threads repeatedly sleep, wake up, fight over the GIL mutex, and trigger thousands of expensive OS context switches while running interpreted bytecode that is orders of magnitude slower (~50–100+ ns per loop iteration), completely masking nanosecond-scale L1 cache effects.