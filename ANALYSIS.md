# Analytical & Forensic Report: Cache Line & False Sharing Mechanics

**Git Handle:** `Mamyr77`  
**Host CPU:** `AMD64::Intel64 Family 6 Model 154 Stepping 3, GenuineIntel::12` (12 logical cores, 64-byte L1D cache line)  
**Instance Seed:** `12902023850628446492`  
**Workload:** 4 threads × 60,000,000 updates (240,000,000 total updates)

---

### Question 1 (MESI Protocol State Transitions)
With `stride = 1` (8-byte spacing), `buffer[0]` (offset `0B`) and `buffer[1]` (offset `8B`) share the same 64-byte L1 data cache line (`CacheLineIndex = 0`)[cite: 9]. When Core 0 writes to `buffer[0]` and Core 1 immediately writes to `buffer[1]`, the hardware cache coherency protocol transitions the line across states as follows:

1. **Initial Read (`S`):** Both Core 0 and Core 1 read the line into their private L1 data caches in the **Shared (`S`)** state.
2. **Core 0 Store (`S -> M` on Core 0, `S -> I` on Core 1):** To modify `buffer[0]`, Core 0 broadcasts a **Read-For-Ownership (RFO)** request on the processor interconnect. Core 1 snoops the RFO and marks its entire 64-byte line as **Invalid (`I`)**. Core 0 transitions its copy to **Modified (`M`)** and commits the write.
3. **Core 1 Store (`I -> M` on Core 1, `M -> I` on Core 0):** When Core 1 attempts to write `buffer[1]`, it hits an **Invalid (`I`)** cache line. Core 1 must stall draining its store buffer and broadcast its own **RFO** request.
4. **Modified Line Transfer:** Core 0 snoops Core 1's RFO while holding the line in **Modified (`M`)** state, flushes/forwards the dirty 64-byte line over the interconnect, and transitions its own line to **Invalid (`I`)**. Core 1 receives ownership (`M`) and finally writes `buffer[1]`.

**Store Buffer Stall:** Although modern out-of-order cores buffer pending writes in a Store Buffer to hide L1 latency, our loop reads `buffer[slot]` on every iteration (`val = buffer[slot]`) before writing back. Repeated cross-core invalidations prevent store-to-load forwarding from completing locally and fill the Store Buffer while waiting for RFO acknowledgments across the ring bus.

---

### Question 2 (Quantitative Slowdown & Coherency Overhead)
Measured execution times and per-update latencies from `cache_lab_manifest.json` (across $2.4 \times 10^8$ total updates):

| Layout / Mode | Stride | Byte Spacing | Elapsed (ns) | Elapsed (ms) | ns / update |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`stride_1` (Contiguous)** | 1 | 8 B | `103,481,000` | `103.481 ms` | `0.4312 ns` |
| **`stride_4` (Half-Line)** | 4 | 32 B | `97,140,000` | `97.140 ms` | `0.4047 ns` |
| **`stride_8` (Line-Padded)** | 8 | 64 B | `97,356,000` | `97.356 ms` | `0.4057 ns` |
| **`stride_16` (Prefetch-Safe)** | 16 | 128 B | `95,180,000` | `95.180 ms` | `0.3966 ns` |
| **`private_baseline` (Register)** | 8 | 64 B (1 write) | `96,317,000` | `96.317 ms` | `0.4013 ns` |

- **Slowdown Factor (`stride_1` vs `stride_8`):**
  $$\text{Slowdown} = \frac{103.481\text{ ms}}{97.356\text{ ms}} = \mathbf{1.063\times} \quad (+6.29\% \text{ overhead})$$
- **Latency Breakdown:** Comparing `private_baseline` (`0.4013 ns/update`) with `stride_16` (`0.3966 ns/update`) shows that when cache lines are isolated from false sharing and prefetcher interference, L1 store-to-load forwarding completely hides L1D store latency (matching pure ALU/register execution within measurement noise). In contrast, `stride_1` increases latency to `0.4312 ns/update` (`+0.0255 ns/update` over `stride_8`, and `+0.0346 ns/update` over `stride_16`) due to cross-core MESI RFO invalidations.

---

### Question 3 (L2 Spatial Prefetcher & 128-Byte Pairing)
In my run, **`stride_8` (64-byte padding)** took **`97.356 ms` (`0.4057 ns/update`)**, while **`stride_16` (128-byte padding)** reduced execution time to **`95.180 ms` (`0.3966 ns/update`)**, saving `2.176 ms`.

While the L1 coherency granularity is 64 bytes, Intel processors pair adjacent even and odd 64-byte lines into **128-byte aligned sectors** in the L2 spatial prefetcher[cite: 9, 10]. Under `stride_8`, Thread 0 (`0B`, Line 0) and Thread 1 (`64B`, Line 1) occupy two halves of the same 128-byte sector. When Core 0 misses or requests ownership on Line 0, the L2 prefetcher speculatively pulls buddy Line 1 into Core 0's cache in Shared (`S`) state, triggering unnecessary snoop traffic and secondary invalidations against Core 1's Modified (`M`) copy. Spacing threads by 128 bytes (`stride_16`) places each thread in a separate 128-byte sector, eliminating prefetcher cross-talk[cite: 10].

---

### Question 4 (Compiler Register Promotion vs. Volatile Memory Stores)
Inspecting `worker_strided_accumulate.inspect_llvm()` confirms that Numba generated native machine code containing **`store i64`** (`i64_store_count = 1`) with **`pyobject fallback: False`**, verifying that every loop iteration performs a real 64-bit memory store.

Normally, LLVM optimization passes (`LICM` / `mem2reg`) try to hoist memory loads/stores out of loops into a CPU register—which is what happens in `worker_private_accumulate`, where `acc` stays in a register for all 60,000,000 iterations and writes to memory once at the end[cite: 10]. In `worker_strided_accumulate`, LLVM cannot promote `buffer[slot]` to a register because `buffer` is passed as an external mutable array pointer into a `nogil=True` function without aliasing guarantees across threads, and each iteration explicitly reads `buffer[slot]` before writing the updated result back to memory.

---

### Question 5 (CPython GIL Masking vs. `nogil=True` Execution)
Running the same `stride = 1` benchmark with standard CPython threads (without `@njit(nogil=True)`) would **not** trigger hardware false sharing and would run significantly slower as thread count increases:

1. **GIL Serialization Eliminates Simultaneous Stores:** CPython's Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time[cite: 9]. Core 0 and Core 1 never write to `buffer[0]` and `buffer[1]` concurrently; one thread retains exclusive ownership of the cache line across its entire OS scheduling slice before releasing the GIL, preventing per-iteration RFO ping-pong.
2. **Software Lock Contention vs. Hardware Coherency Contention:** In `@njit(nogil=True)` mode, all 4 threads run natively on separate cores with `0` context switches (`Vol CSW = 0, Invol CSW = 0`), exposing nanosecond-scale L1 cache-line contention (`0.4312 ns/update`). Standard CPython threads are bottlenecked by OS-level GIL mutex contention, thread wakeups, context switches, and interpreter overhead (~50–100+ ns per iteration), completely masking hardware cache effects.с