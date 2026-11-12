# Appendix: Detailed Sweeps, Ablations, and Engineering Notes (Physical AI VLA on TPU v7x)

This document preserves the full 328-point measured sweep tables, hardware/software execution architecture, configurations that regressed latency (`TP=2` and dynamic `fp8` on `3B–4B` models at `B = 1`, synchronous HTTP vision offload on 1–2 cameras), and methodology caveats. For headline results and reproduction commands, see [`README.md`](./README.md).

---

## 1. Execution Architecture & Methodology Caveats

1. **Multi-process TensorCore isolation (`torchrun --nproc_per_node=8`):** Each Cloud TPU `v7x-8` host exposes `4` physical chips (`8` hardware TensorCores, `96 GiB` HBM3e per TensorCore). Assigning one OS process per TensorCore (`TORCH_TPU_TOPOLOGY=2,2,1,2` via [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py)) eliminates Python GIL contention and single-process PJRT dispatch serialization while sharing compiled XLA executables across ranks via TorchTPU's Tier-2 disk cache.
2. **Stage-faithful TorchTPU compiled kernels:** [`scripts/run_physical_ai_vla_v7x_triathlon.py`](./scripts/run_physical_ai_vla_v7x_triathlon.py) stages each upstream `.safetensors` checkpoint from GCS/FUSE into `/dev/shm` (`812–1,861` tensors per model) and executes dedicated `torch.compile(dynamic=False)` stage kernels matching each model's exact layer counts, hidden widths, GQA head counts, MLP activations (`SiLU`, `ReLU²`, `GELU`), vision token sequence lengths, and action-head decoding structure (4/10-step flow-matching ODE, 7-step KV-cached autoregressive decode, or 1-pass bidirectional chunking), populating shape-matched 2D weight matrices from the staged `.safetensors` pool. These benchmarks measure hardware execution latency, HBM capacity, ICI `all_reduce` scaling, and HTTP fleet serving behavior rather than closed-loop task success rates in simulation.
3. **Warm steady-state measurement:** All reported latencies are warm `p50` / `p99` measurements after XLA graph compilation. Cold first-call XLA compilation is excluded from steady-state serving metrics.
4. **Disaggregated vision preprocessing overlap:** In `multicam_resolution_sweep`, synchronous local CPU E2E (`local_cpu_preproc_ms + tpu_p50_ms`), synchronous remote HTTP E2E (`disagg_cpu_rtt_ms + tpu_p50_ms`), and double-buffered pipelined E2E (`max(tpu_p50_ms, disagg_cpu_rtt_ms)`) are all reported explicitly.

---

## 2. Summary of Configurations That Did Not Help at `B = 1` or Saturated

- **Intra-chip `TP=2` at `B = 1` on `3B–4.3B` flow-matching policies (`pi0.5`, `GR00T-N1.7-3B`, `Cosmos3-Edge`):** Sharding across 2 TensorCores slowed `B = 1` inference (`6.53 → 11.70 ms` on `pi0.5-Base`, `6.49 → 9.59 ms` on `pi0.5-LIBERO`, `9.28 → 10.79 ms` on `GR00T-N1.7-3B`, `11.38 → 13.00 ms` on `Cosmos3-Edge`) because per-layer ICI `all_reduce` latency (~0.3 ms across dozens of layers) exceeds the compute/HBM savings on compact `6.4–8.5 GiB` models at batch 1. `TP=2` only lowers `B = 1` latency on the memory-bandwidth-bound `7.54B` models (`54.49 → 34.56 ms` on `OpenVLA-7B`, `14.76 → 11.38 ms` on `OpenVLA-7B-OFT`), though it speeds up `B >= 4` (`1.66–2.17×` vs. single-TensorCore `DP=1`) across all 6 models.
- **Dynamic `fp8` (`torch._scaled_mm`) at `B = 1` on `3B–4.3B` policies:** Per-activation dynamic `amax` scaling and `m`-dimension 16-row padding made `fp8` slightly slower than `bfloat16` at `B = 1` on `pi0.5` (`7.54 ms` vs. `6.53 ms`) and `GR00T-N1.7-3B` (`10.96 ms` vs. `9.28 ms`). Dynamic `fp8` improved `B = 1` latency only on `OpenVLA-7B` (`42.87 ms` vs. `54.49 ms`, `1.27×` faster) and `OpenVLA-7B-OFT` (`14.17 ms` vs. `14.76 ms`), while halving weight HBM across all models (`6.74 → 3.37 GiB` on `pi0.5`, `14.05 → 7.20 GiB` on `OpenVLA-7B`) and increasing large-batch `B = 32` throughput on `pi0.5` (`85,322` vs. `79,247 actions/s` on `DP=8`).
- **Synchronous HTTP CPU vision offload on 1–2 camera views:** Offloading `1×224²` or `2×224²` JPEG decode/resize over HTTP increased synchronous preprocessing time (`4.47 ms` vs. `2.86 ms` local CPU at `1×224²`; `5.54 ms` vs. `4.90 ms` at `2×224²`) due to serialization and network RTT. Synchronous remote offload outperformed local TPU-host CPU only at `4` cameras (`7.57 ms` vs. `9.48 ms` at `4×224²`, `1.25×` faster). Double-buffered pipelining is required to hide CPU preprocessing behind TPU inference.
- **Single-host `DP=8` under `1,000 Hz` open-loop Poisson load:** A single 8-TensorCore host saturates under `1,000 Hz` Poisson arrivals on `GR00T-N1.7-3B` (`635.6 Hz` achieved, `218.40 ms` `p99`), `Cosmos3-Edge` (`588.8 Hz`, `220.11 ms` `p99`), and `OpenVLA-7B-OFT` (`489.3 Hz`, `261.74 ms` `p99`), and `pi0.5` tail latency rises to `91.0–91.4 ms` `p99`. Scaling to `2-Host DP=16` (`16` TensorCores) removes the queueing knee (`13.82–34.92 ms` `p99` at `917–945 Hz` achieved).
- **Autoregressive `OpenVLA-7B` above `280 Hz` fleet rate:** Because 7 sequential KV-cached decode steps take `54.49 ms` per request (`chunk = 1`), even `2-Host DP=16` saturates at `280.3 Hz` (`659.23 ms` `p50` / `675.32 ms` `p99` under `500–1,000 Hz` Poisson arrivals).

---

## 3. Batch Scaling & Precision Matrix (`bfloat16` vs. Dynamic `fp8`, `B = 1..32`)

Raw logs: [`results/host0_results.jsonl`](./results/host0_results.jsonl) and [`results/host1_results.jsonl`](./results/host1_results.jsonl).

| Model | Precision | Batch `B` | Cam × Res | Chunk × Steps | `1-TC` `p50` / `p99` (ms) | Pipelined E2E `p50` (ms) | `1-TC` Actions/s | `1-Host DP=8` Wave `p50` (ms) | `1-Host DP=8` Actions/s | `DP=8` Scaling Eff. | HBM / TC (GiB) |
| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi0.5-Base (3.6B)` | `BF16` | 1 | `2×224x224` | `50×4` | 6.53 / 7.20 | 6.53 | 7,660.7 | 10.19 | 39,254.0 | 64.1% | 6.74 |
| `pi0.5-Base (3.6B)` | `BF16` | 2 | `2×224x224` | `50×4` | 11.04 / 11.77 | 11.04 | 9,059.1 | 12.02 | 66,555.5 | 91.8% | 6.74 |
| `pi0.5-Base (3.6B)` | `BF16` | 4 | `2×224x224` | `50×4` | 20.60 / 21.19 | 20.60 | 9,710.0 | 21.35 | 74,941.5 | 96.5% | 6.74 |
| `pi0.5-Base (3.6B)` | `BF16` | 8 | `2×224x224` | `50×4` | 38.35 / 38.86 | 38.35 | 10,431.2 | 38.87 | 82,325.5 | 98.7% | 6.74 |
| `pi0.5-Base (3.6B)` | `BF16` | 16 | `2×224x224` | `50×4` | 83.54 / 83.64 | 83.54 | 9,576.8 | 84.02 | 76,172.5 | 99.4% | 6.74 |
| `pi0.5-Base (3.6B)` | `BF16` | 32 | `2×224x224` | `50×4` | 161.07 / 161.35 | 161.07 | 9,933.4 | 161.52 | 79,247.0 | 99.7% | 6.74 |
| `pi0.5-Base (3.6B)` | `FP8` | 1 | `2×224x224` | `50×4` | 7.54 / 7.89 | 7.54 | 6,634.8 | 10.22 | 39,139.0 | 73.7% | 3.37 |
| `pi0.5-Base (3.6B)` | `FP8` | 2 | `2×224x224` | `50×4` | 11.64 / 12.69 | 11.64 | 8,592.0 | 12.21 | 65,520.0 | 95.3% | 3.37 |
| `pi0.5-Base (3.6B)` | `FP8` | 4 | `2×224x224` | `50×4` | 20.28 / 20.44 | 20.28 | 9,863.4 | 20.76 | 77,071.5 | 97.7% | 3.37 |
| `pi0.5-Base (3.6B)` | `FP8` | 8 | `2×224x224` | `50×4` | 39.38 / 39.66 | 39.38 | 10,157.9 | 40.10 | 79,800.5 | 98.2% | 3.37 |
| `pi0.5-Base (3.6B)` | `FP8` | 16 | `2×224x224` | `50×4` | 76.58 / 76.61 | 76.58 | 10,447.1 | 77.15 | 82,955.5 | 99.3% | 3.37 |
| `pi0.5-Base (3.6B)` | `FP8` | 32 | `2×224x224` | `50×4` | 149.36 / 149.48 | 149.36 | 10,712.7 | 150.02 | **85,322.0** | 99.6% | 3.37 |
| `pi0.5-LIBERO (3.6B)` | `BF16` | 1 | `2×224x224` | `50×4` | 6.49 / 7.24 | 6.49 | 7,709.4 | 7.81 | 51,216.5 | 83.0% | 6.74 |
| `pi0.5-LIBERO (3.6B)` | `BF16` | 2 | `2×224x224` | `50×4` | 11.01 / 11.23 | 11.01 | 9,080.7 | 11.79 | 67,854.0 | 93.4% | 6.74 |
| `pi0.5-LIBERO (3.6B)` | `BF16` | 4 | `2×224x224` | `50×4` | 20.58 / 20.79 | 20.58 | 9,719.3 | 21.48 | 74,488.0 | 95.8% | 6.74 |
| `pi0.5-LIBERO (3.6B)` | `BF16` | 8 | `2×224x224` | `50×4` | 38.24 / 38.87 | 38.24 | 10,460.0 | 38.85 | 82,368.0 | 98.4% | 6.74 |
| `pi0.5-LIBERO (3.6B)` | `BF16` | 16 | `2×224x224` | `50×4` | 83.42 / 83.82 | 83.42 | 9,590.3 | 84.04 | 76,154.0 | 99.3% | 6.74 |
| `pi0.5-LIBERO (3.6B)` | `BF16` | 32 | `2×224x224` | `50×4` | 160.90 / 161.19 | 160.90 | 9,944.3 | 161.55 | 79,232.5 | 99.6% | 6.74 |
| `pi0.5-LIBERO (3.6B)` | `FP8` | 1 | `2×224x224` | `50×4` | 7.55 / 8.62 | 7.55 | 6,624.2 | 10.58 | 37,807.0 | 71.3% | 3.37 |
| `pi0.5-LIBERO (3.6B)` | `FP8` | 2 | `2×224x224` | `50×4` | 11.63 / 12.08 | 11.63 | 8,595.2 | 12.21 | 65,520.0 | 95.3% | 3.37 |
| `pi0.5-LIBERO (3.6B)` | `FP8` | 4 | `2×224x224` | `50×4` | 20.15 / 20.42 | 20.15 | 9,923.5 | 21.01 | 76,154.0 | 95.9% | 3.37 |
| `pi0.5-LIBERO (3.6B)` | `FP8` | 8 | `2×224x224` | `50×4` | 39.52 / 39.59 | 39.52 | 10,121.0 | 40.08 | 79,840.5 | 98.6% | 3.37 |
| `pi0.5-LIBERO (3.6B)` | `FP8` | 16 | `2×224x224` | `50×4` | 76.59 / 77.02 | 76.59 | 10,445.8 | 77.26 | 82,837.0 | 99.1% | 3.37 |
| `pi0.5-LIBERO (3.6B)` | `FP8` | 32 | `2×224x224` | `50×4` | 149.26 / 149.53 | 149.26 | 10,719.7 | 149.88 | **85,401.5** | 99.6% | 3.37 |
| `GR00T-N1.7-3B` | `BF16` | 1 | `2×256x256` | `50×4` | 9.28 / 9.56 | 9.28 | 5,388.9 | 10.64 | 37,594.0 | 87.2% | 6.44 |
| `GR00T-N1.7-3B` | `BF16` | 2 | `2×256x256` | `50×4` | 14.24 / 14.62 | 14.24 | 7,022.0 | 15.29 | 52,322.0 | 93.1% | 6.44 |
| `GR00T-N1.7-3B` | `BF16` | 4 | `2×256x256` | `50×4` | 24.54 / 25.34 | 24.54 | 8,150.3 | 25.15 | **63,618.5** | 97.6% | 6.44 |
| `GR00T-N1.7-3B` | `BF16` | 8 | `2×256x256` | `50×4` | 55.01 / 55.19 | 55.01 | 7,271.2 | 55.88 | 57,265.5 | 98.4% | 6.44 |
| `GR00T-N1.7-3B` | `BF16` | 16 | `2×256x256` | `50×4` | 108.75 / 108.98 | 108.75 | 7,356.4 | 109.54 | 58,426.0 | 99.3% | 6.44 |
| `GR00T-N1.7-3B` | `BF16` | 32 | `2×256x256` | `50×4` | 221.34 / 221.77 | 221.34 | 7,228.7 | 222.25 | 57,593.0 | 99.6% | 6.44 |
| `GR00T-N1.7-3B` | `FP8` | 1 | `2×256x256` | `50×4` | 10.96 / 11.45 | 10.96 | 4,563.1 | 13.28 | 30,120.5 | 82.5% | 3.22 |
| `GR00T-N1.7-3B` | `FP8` | 2 | `2×256x256` | `50×4` | 16.39 / 16.86 | 16.39 | 6,100.0 | 17.42 | 45,924.0 | 94.1% | 3.22 |
| `GR00T-N1.7-3B` | `FP8` | 4 | `2×256x256` | `50×4` | 26.70 / 26.98 | 26.70 | 7,489.8 | 27.50 | 58,182.0 | 97.1% | 3.22 |
| `GR00T-N1.7-3B` | `FP8` | 8 | `2×256x256` | `50×4` | 61.29 / 61.54 | 61.29 | 6,526.3 | 62.15 | 51,488.5 | 98.6% | 3.22 |
| `GR00T-N1.7-3B` | `FP8` | 16 | `2×256x256` | `50×4` | 115.26 / 115.54 | 115.26 | 6,941.0 | 116.07 | 55,139.0 | 99.3% | 3.22 |
| `GR00T-N1.7-3B` | `FP8` | 32 | `2×256x256` | `50×4` | 226.06 / 226.26 | 226.06 | 7,077.8 | 226.92 | 56,407.5 | 99.6% | 3.22 |
| `Cosmos3-Edge-DROID` | `BF16` | 1 | `2×256x256` | `32×4` | 11.38 / 11.90 | 11.38 | 2,811.1 | 12.39 | 20,661.8 | 91.9% | 8.50 |
| `Cosmos3-Edge-DROID` | `BF16` | 2 | `2×256x256` | `32×4` | 16.16 / 16.57 | 16.16 | 3,960.8 | 16.72 | 30,622.1 | 96.6% | 8.50 |
| `Cosmos3-Edge-DROID` | `BF16` | 4 | `2×256x256` | `32×4` | 25.81 / 25.94 | 25.81 | 4,959.1 | 26.39 | 38,802.6 | 97.8% | 8.50 |
| `Cosmos3-Edge-DROID` | `BF16` | 8 | `2×256x256` | `32×4` | 42.94 / 44.91 | 42.94 | 5,961.9 | 43.53 | **47,048.0** | 98.6% | 8.50 |
| `Cosmos3-Edge-DROID` | `BF16` | 16 | `2×256x256` | `32×4` | 94.00 / 94.49 | 94.00 | 5,446.9 | 94.56 | 43,316.5 | 99.4% | 8.50 |
| `Cosmos3-Edge-DROID` | `BF16` | 32 | `2×256x256` | `32×4` | 185.64 / 187.07 | 185.64 | 5,516.1 | 186.48 | 43,929.6 | 99.5% | 8.50 |
| `Cosmos3-Edge-DROID` | `FP8` | 1 | `2×256x256` | `32×4` | 11.43 / 11.74 | 11.43 | 2,799.4 | 13.52 | 18,935.0 | 84.5% | 4.25 |
| `Cosmos3-Edge-DROID` | `FP8` | 2 | `2×256x256` | `32×4` | 16.18 / 17.27 | 16.18 | 3,955.6 | 17.21 | 29,750.1 | 94.0% | 4.25 |
| `Cosmos3-Edge-DROID` | `FP8` | 4 | `2×256x256` | `32×4` | 27.04 / 27.20 | 27.04 | 4,734.5 | 27.67 | 37,007.7 | 97.7% | 4.25 |
| `Cosmos3-Edge-DROID` | `FP8` | 8 | `2×256x256` | `32×4` | 49.62 / 52.04 | 49.62 | 5,159.0 | 50.30 | 40,715.8 | 98.7% | 4.25 |
| `Cosmos3-Edge-DROID` | `FP8` | 16 | `2×256x256` | `32×4` | 93.83 / 93.91 | 93.83 | 5,456.7 | 94.49 | 43,348.5 | 99.3% | 4.25 |
| `Cosmos3-Edge-DROID` | `FP8` | 32 | `2×256x256` | `32×4` | 183.82 / 184.14 | 183.82 | 5,570.8 | 184.51 | 44,398.7 | 99.6% | 4.25 |
| `OpenVLA-7B (AR)` | `BF16` | 1 | `2×224x224` | `1×7` | 54.49 / 54.79 | 54.49 | 18.4 | 56.25 | 142.2 | 96.9% | 14.05 |
| `OpenVLA-7B (AR)` | `BF16` | 2 | `2×224x224` | `1×7` | 76.36 / 77.05 | 76.36 | 26.2 | 77.25 | 207.1 | 98.8% | 14.05 |
| `OpenVLA-7B (AR)` | `BF16` | 4 | `2×224x224` | `1×7` | 128.75 / 128.81 | 128.75 | 31.1 | 129.22 | 247.6 | 99.6% | 14.05 |
| `OpenVLA-7B (AR)` | `BF16` | 8 | `2×224x224` | `1×7` | 211.36 / 211.70 | 211.36 | 37.9 | 212.13 | 301.7 | 99.6% | 14.05 |
| `OpenVLA-7B (AR)` | `BF16` | 16 | `2×224x224` | `1×7` | 382.40 / 382.57 | 382.40 | 41.8 | 382.96 | 334.2 | 99.9% | 14.05 |
| `OpenVLA-7B (AR)` | `BF16` | 32 | `2×224x224` | `1×7` | 729.98 / 730.19 | 729.98 | 43.8 | 730.33 | 350.5 | 100.0% | 14.05 |
| `OpenVLA-7B (AR)` | `FP8` | 1 | `2×224x224` | `1×7` | **42.87** / 45.03 | 42.87 | 23.3 | 43.99 | 181.9 | 97.4% | 7.20 |
| `OpenVLA-7B (AR)` | `FP8` | 2 | `2×224x224` | `1×7` | 63.47 / 63.73 | 63.47 | 31.5 | 64.16 | 249.4 | 98.9% | 7.20 |
| `OpenVLA-7B (AR)` | `FP8` | 4 | `2×224x224` | `1×7` | 106.98 / 107.08 | 106.98 | 37.4 | 107.93 | 296.5 | 99.1% | 7.20 |
| `OpenVLA-7B (AR)` | `FP8` | 8 | `2×224x224` | `1×7` | 180.39 / 180.91 | 180.39 | 44.3 | 181.19 | 353.2 | 99.6% | 7.20 |
| `OpenVLA-7B (AR)` | `FP8` | 16 | `2×224x224` | `1×7` | 327.25 / 327.58 | 327.25 | 48.9 | 328.07 | 390.2 | 99.8% | 7.20 |
| `OpenVLA-7B (AR)` | `FP8` | 32 | `2×224x224` | `1×7` | 627.90 / 628.05 | 627.90 | 51.0 | 628.55 | **407.3** | 99.9% | 7.20 |
| `OpenVLA-7B-OFT` | `BF16` | 1 | `2×224x224` | `50×1` | 14.76 / 14.85 | 14.76 | 3,386.9 | 15.76 | 25,380.5 | 93.7% | 14.50 |
| `OpenVLA-7B-OFT` | `BF16` | 2 | `2×224x224` | `50×1` | 38.54 / 38.74 | 38.54 | 2,594.9 | 39.57 | 20,217.5 | 97.4% | 14.50 |
| `OpenVLA-7B-OFT` | `BF16` | 4 | `2×224x224` | `50×1` | 62.86 / 63.11 | 62.86 | 3,181.9 | 63.59 | 25,161.0 | 98.8% | 14.50 |
| `OpenVLA-7B-OFT` | `BF16` | 8 | `2×224x224` | `50×1` | 124.72 / 124.91 | 124.72 | 3,207.2 | 125.67 | 25,463.5 | 99.2% | 14.50 |
| `OpenVLA-7B-OFT` | `BF16` | 16 | `2×224x224` | `50×1` | 245.14 / 245.25 | 245.14 | 3,263.5 | 245.59 | 26,059.5 | 99.8% | 14.50 |
| `OpenVLA-7B-OFT` | `BF16` | 32 | `2×224x224` | `50×1` | 483.66 / 488.38 | 483.66 | 3,308.1 | 484.54 | 26,417.0 | 99.8% | 14.50 |
| `OpenVLA-7B-OFT` | `FP8` | 1 | `2×224x224` | `50×1` | **14.17** / 14.76 | 14.17 | 3,527.8 | 15.59 | 25,657.5 | 90.9% | 7.25 |
| `OpenVLA-7B-OFT` | `FP8` | 2 | `2×224x224` | `50×1` | 30.42 / 30.74 | 30.42 | 3,287.0 | 31.25 | 25,600.0 | 97.4% | 7.25 |
| `OpenVLA-7B-OFT` | `FP8` | 4 | `2×224x224` | `50×1` | 63.47 / 63.64 | 63.47 | 3,151.0 | 64.37 | 24,856.5 | 98.6% | 7.25 |
| `OpenVLA-7B-OFT` | `FP8` | 8 | `2×224x224` | `50×1` | 101.07 / 101.31 | 101.07 | 3,957.8 | 102.00 | 31,372.5 | 99.1% | 7.25 |
| `OpenVLA-7B-OFT` | `FP8` | 16 | `2×224x224` | `50×1` | 192.89 / 193.23 | 192.89 | 4,147.4 | 193.55 | 33,066.5 | 99.7% | 7.25 |
| `OpenVLA-7B-OFT` | `FP8` | 32 | `2×224x224` | `50×1` | 382.53 / 382.78 | 382.53 | 4,182.7 | 383.01 | **33,419.5** | 99.9% | 7.25 |

---

## 4. Intra-Chip `TP=2` (`tpu_dist` ICI `all_reduce`) Sharding Matrix

Each physical `v7x` chip pairs 2 TensorCores (`192 GiB` HBM3e) over intra-package ICI. Relative to `1-TC (DP=1)` `bfloat16`:

| Model | Batch `B` | `1-TC (DP=1)` `BF16` `p50` (ms) | `1-Chip TP=2 (2 TC)` `p50` / `p99` (ms) | `TP=2` vs. `1-TC` Speedup | `1-Chip TP=2` Actions/s | `1-Host (4×TP=2)` Actions/s | `TP=2` HBM / TC (GiB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi0.5-Base (3.6B)` | 1 | 6.53 | 11.70 / 13.82 | 0.56× | 4,274.7 | 17,099.0 | 3.50 |
| `pi0.5-Base (3.6B)` | 4 | 20.60 | 12.40 / 12.56 | 1.66× | 16,124.4 | 64,497.6 | 3.50 |
| `pi0.5-Base (3.6B)` | 8 | 38.35 | 22.48 / 22.70 | 1.71× | 17,794.1 | 71,176.6 | 3.50 |
| `pi0.5-Base (3.6B)` | 16 | 83.54 | 41.28 / 41.52 | 2.02× | 19,379.3 | 77,517.2 | 3.50 |
| `pi0.5-LIBERO (3.6B)` | 1 | 6.49 | 9.59 / 9.98 | 0.68× | 5,212.8 | 20,851.3 | 3.50 |
| `pi0.5-LIBERO (3.6B)` | 4 | 20.58 | 12.14 / 12.48 | 1.70× | 16,469.4 | 65,877.7 | 3.50 |
| `pi0.5-LIBERO (3.6B)` | 8 | 38.24 | 22.27 / 22.84 | 1.72× | 17,963.5 | 71,854.1 | 3.50 |
| `pi0.5-LIBERO (3.6B)` | 16 | 83.42 | 41.37 / 41.52 | 2.02× | 19,337.0 | 77,347.9 | 3.50 |
| `GR00T-N1.7-3B` | 1 | 9.28 | 10.79 / 13.17 | 0.86× | 4,635.8 | 18,543.4 | 3.35 |
| `GR00T-N1.7-3B` | 4 | 24.54 | 13.60 / 13.87 | 1.80× | 14,709.5 | 58,837.8 | 3.35 |
| `GR00T-N1.7-3B` | 8 | 55.01 | 25.32 / 25.91 | 2.17× | 15,799.8 | 63,199.3 | 3.35 |
| `GR00T-N1.7-3B` | 16 | 108.75 | 57.08 / 59.56 | 1.91× | 14,016.3 | 56,065.0 | 3.35 |
| `Cosmos3-Edge-DROID` | 1 | 11.38 | 13.00 / 16.25 | 0.88× | 2,461.7 | 9,846.8 | 4.42 |
| `Cosmos3-Edge-DROID` | 4 | 25.81 | 14.59 / 22.31 | 1.77× | 8,774.2 | 35,096.8 | 4.42 |
| `Cosmos3-Edge-DROID` | 8 | 42.94 | 24.26 / 28.17 | 1.77× | 10,553.0 | 42,211.9 | 4.42 |
| `Cosmos3-Edge-DROID` | 16 | 94.00 | 45.85 / 46.38 | 2.05× | 11,167.9 | 44,671.7 | 4.42 |
| `OpenVLA-7B (AR)` | 1 | 54.49 | **34.56** / 40.35 | **1.58×** | 28.9 | 115.7 | 7.30 |
| `OpenVLA-7B (AR)` | 4 | 128.75 | 68.66 / 71.62 | 1.88× | 58.3 | 233.0 | 7.30 |
| `OpenVLA-7B (AR)` | 8 | 211.36 | 113.04 / 113.33 | 1.87× | 70.8 | 283.1 | 7.30 |
| `OpenVLA-7B (AR)` | 16 | 382.40 | 204.01 / 204.61 | 1.87× | 78.4 | 313.7 | 7.30 |
| `OpenVLA-7B-OFT` | 1 | 14.76 | **11.38** / 12.61 | **1.30×** | 4,393.5 | 17,574.0 | 7.54 |
| `OpenVLA-7B-OFT` | 4 | 62.86 | 35.55 / 35.84 | 1.77× | 5,625.8 | 22,503.2 | 7.54 |
| `OpenVLA-7B-OFT` | 8 | 124.72 | 67.57 / 67.98 | 1.85× | 5,919.9 | 23,679.7 | 7.54 |
| `OpenVLA-7B-OFT` | 16 | 245.14 | 130.66 / 131.06 | 1.88× | 6,122.9 | 24,491.6 | 7.54 |

---

## 5. Two-Host Slice `DP=16` (`8` Chips = `16` TensorCores) Batch, Concurrency & Poisson Sweeps

Raw logs: [`results/multihost_dp16_results.jsonl`](./results/multihost_dp16_results.jsonl).

### A. `2-Host DP=16` Batch Scaling (`Concurrency = 16`, `bfloat16`)

| Model | Batch `B` | TPU Service `p50` (ms) | Cross-Host RPC E2E `p50` / `p95` / `p99` (ms) | `2-Host DP=16` Trajectories/s | `2-Host DP=16` Actions/s | Control Targets Met (`100 / 50 / 20 Hz`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| `pi0.5-Base (3.6B)` | 1 | 7.24 | 8.72 / 12.74 / 12.95 | 1,340.47 | 67,023.4 | `100 / 50 / 20 Hz` |
| `pi0.5-Base (3.6B)` | 4 | 20.82 | 22.99 / 26.94 / 29.92 | 2,360.61 | 118,030.6 | `— / — / 20 Hz` |
| `pi0.5-Base (3.6B)` | 8 | 38.30 | 40.51 / 44.34 / 44.98 | 2,907.55 | **145,377.4** | `— / — / 20 Hz` |
| `pi0.5-Base (3.6B)` | 16 | 83.49 | 85.20 / 89.22 / 89.84 | 2,866.56 | 143,327.9 | `— / — / —` |
| `pi0.5-LIBERO (3.6B)` | 1 | 7.15 | 8.68 / 13.48 / 13.96 | 1,372.26 | 68,613.2 | `100 / 50 / 20 Hz` |
| `pi0.5-LIBERO (3.6B)` | 4 | 20.83 | 23.06 / 26.92 / 27.13 | 2,385.12 | 119,255.9 | `— / — / 20 Hz` |
| `pi0.5-LIBERO (3.6B)` | 8 | 38.27 | 40.47 / 44.28 / 44.76 | 2,869.96 | **143,497.8** | `— / — / 20 Hz` |
| `pi0.5-LIBERO (3.6B)` | 16 | 83.54 | 85.51 / 89.56 / 90.27 | 2,867.78 | 143,389.1 | `— / — / —` |
| `GR00T-N1.7-3B` | 1 | 9.74 | 11.68 / 20.51 / 21.18 | 997.14 | 49,857.0 | `— / 50 / 20 Hz` |
| `GR00T-N1.7-3B` | 4 | 24.39 | 26.21 / 29.91 / 30.48 | 2,118.55 | 105,927.4 | `— / — / 20 Hz` |
| `GR00T-N1.7-3B` | 8 | 54.95 | 56.52 / 60.86 / 61.39 | 2,109.39 | 105,469.6 | `— / — / —` |
| `GR00T-N1.7-3B` | 16 | 108.68 | 110.09 / 114.71 / 115.04 | 2,224.97 | **111,248.3** | `— / — / —` |
| `Cosmos3-Edge-DROID` | 1 | 11.71 | 13.02 / 17.90 / 18.81 | 952.59 | 30,482.9 | `— / 50 / 20 Hz` |
| `Cosmos3-Edge-DROID` | 4 | 25.91 | 27.91 / 31.37 / 34.03 | 2,021.99 | 64,703.7 | `— / — / 20 Hz` |
| `Cosmos3-Edge-DROID` | 8 | 42.88 | 45.27 / 48.76 / 49.29 | 2,609.11 | **83,491.6** | `— / — / 20 Hz` |
| `Cosmos3-Edge-DROID` | 16 | 93.92 | 95.93 / 99.89 / 100.16 | 2,560.72 | 81,943.0 | `— / — / —` |
| `OpenVLA-7B (AR)` | 1 | 54.81 | 57.24 / 60.87 / 60.96 | 262.03 | 262.0 | `— / — / —` |
| `OpenVLA-7B (AR)` | 4 | 128.73 | 130.42 / 134.21 / 136.12 | 476.69 | 476.7 | `— / — / —` |
| `OpenVLA-7B (AR)` | 8 | 211.33 | 214.33 / 217.07 / 217.81 | 588.15 | 588.2 | `— / — / —` |
| `OpenVLA-7B (AR)` | 16 | 382.21 | 384.18 / 387.73 / 388.00 | 660.00 | **660.0** | `— / — / —` |
| `OpenVLA-7B-OFT` | 1 | 15.02 | 18.31 / 20.83 / 21.36 | 762.77 | 38,138.5 | `— / 50 / 20 Hz` |
| `OpenVLA-7B-OFT` | 4 | 62.58 | 64.88 / 68.17 / 68.67 | 938.35 | 46,917.4 | `— / — / —` |
| `OpenVLA-7B-OFT` | 8 | 124.56 | 126.88 / 130.35 / 130.98 | 977.36 | 48,867.8 | `— / — / —` |
| `OpenVLA-7B-OFT` | 16 | 244.47 | 246.28 / 249.97 / 251.46 | 1,023.63 | **51,181.4** | `— / — / —` |

### B. Closed-Loop Fleet Concurrency (`CON = 1..64`: `1-Host DP=8` vs. `2-Host DP=16`)

| Model | Concurrency `CON` | `1-Host DP=8` E2E `p50` / `p99` (ms) | `1-Host DP=8` Actions/s | `2-Host DP=16` E2E `p50` / `p99` (ms) | `2-Host DP=16` Actions/s | `2-Host / 1-Host` Throughput Ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi0.5-Base (3.6B)` | 1 | 7.73 / 8.31 | 6,412.2 | 7.81 / 8.67 | 6,338.6 | 0.99× |
| `pi0.5-Base (3.6B)` | 4 | 8.82 / 14.78 | 18,245.8 | 8.47 / 13.08 | 19,285.1 | 1.06× |
| `pi0.5-Base (3.6B)` | 8 | 13.10 / 19.11 | 22,997.4 | 8.51 / 14.88 | 33,992.0 | 1.48× |
| `pi0.5-Base (3.6B)` | 16 | 15.95 / 25.12 | 33,648.6 | 8.87 / 14.13 | 64,212.4 | 1.91× |
| `pi0.5-Base (3.6B)` | 32 | 26.42 / 59.08 | 38,411.3 | 14.48 / 20.82 | **77,557.4** | 2.02× |
| `pi0.5-Base (3.6B)` | 64 | 54.93 / 89.21 | 39,013.2 | 26.40 / 61.03 | 74,270.8 | 1.90× |
| `pi0.5-LIBERO (3.6B)` | 1 | 7.53 / 8.44 | 6,498.2 | 7.91 / 8.68 | 6,314.2 | 0.97× |
| `pi0.5-LIBERO (3.6B)` | 4 | 8.73 / 13.11 | 18,136.2 | 8.49 / 12.70 | 20,139.6 | 1.11× |
| `pi0.5-LIBERO (3.6B)` | 8 | 12.07 / 14.97 | 26,935.8 | 8.50 / 12.82 | 38,753.6 | 1.44× |
| `pi0.5-LIBERO (3.6B)` | 16 | 15.60 / 24.41 | 36,505.8 | 9.05 / 13.08 | 59,562.8 | 1.63× |
| `pi0.5-LIBERO (3.6B)` | 32 | 29.21 / 45.88 | 36,468.3 | 15.11 / 22.68 | 73,607.5 | 2.02× |
| `pi0.5-LIBERO (3.6B)` | 64 | 60.36 / 91.26 | 39,626.6 | 25.80 / 48.68 | **78,322.7** | 1.98× |
| `GR00T-N1.7-3B` | 1 | 10.24 / 11.16 | 4,789.7 | 10.66 / 11.68 | 4,672.0 | 0.98× |
| `GR00T-N1.7-3B` | 4 | 14.75 / 16.84 | 13,227.6 | 10.95 / 14.15 | 17,376.5 | 1.31× |
| `GR00T-N1.7-3B` | 8 | 15.23 / 17.77 | 20,305.0 | 10.99 / 14.36 | 31,294.5 | 1.54× |
| `GR00T-N1.7-3B` | 16 | 21.43 / 29.73 | 28,837.7 | 13.11 / 18.29 | 49,810.4 | 1.73× |
| `GR00T-N1.7-3B` | 32 | 37.59 / 72.36 | 28,399.4 | 19.98 / 29.55 | 58,773.3 | 2.07× |
| `GR00T-N1.7-3B` | 64 | 85.30 / 119.17 | 31,309.8 | 31.19 / 49.11 | **60,988.8** | 1.95× |
| `Cosmos3-Edge-DROID` | 1 | 12.56 / 14.91 | 2,506.6 | 12.83 / 13.94 | 2,461.7 | 0.98× |
| `Cosmos3-Edge-DROID` | 4 | 13.20 / 17.12 | 8,799.8 | 13.02 / 17.12 | 8,959.1 | 1.02× |
| `Cosmos3-Edge-DROID` | 8 | 16.91 / 19.28 | 13,238.9 | 13.17 / 16.85 | 17,017.6 | 1.29× |
| `Cosmos3-Edge-DROID` | 16 | 24.46 / 32.29 | 16,690.2 | 13.12 / 17.71 | 27,403.0 | 1.64× |
| `Cosmos3-Edge-DROID` | 32 | 47.35 / 58.90 | 18,040.3 | 23.57 / 30.83 | 33,811.8 | 1.87× |
| `Cosmos3-Edge-DROID` | 64 | 95.41 / 119.47 | 18,207.0 | 46.98 / 56.60 | **35,737.6** | 1.96× |
| `OpenVLA-7B (AR)` | 1 | 55.26 / 56.53 | 18.0 | 55.69 / 56.65 | 17.9 | 0.99× |
| `OpenVLA-7B (AR)` | 4 | 55.89 / 70.59 | 67.1 | 56.02 / 60.79 | 70.7 | 1.05× |
| `OpenVLA-7B (AR)` | 8 | 60.09 / 80.09 | 117.1 | 56.30 / 57.01 | 139.2 | 1.19× |
| `OpenVLA-7B (AR)` | 16 | 109.96 / 132.98 | 130.3 | 56.50 / 60.47 | 260.1 | 2.00× |
| `OpenVLA-7B (AR)` | 32 | 220.69 / 238.59 | 135.7 | 109.97 / 115.95 | 276.2 | 2.04× |
| `OpenVLA-7B (AR)` | 64 | 440.97 / 463.83 | 140.6 | 220.62 / 227.82 | **279.8** | 1.99× |
| `OpenVLA-7B-OFT` | 1 | 15.59 / 17.59 | 3,178.3 | 15.69 / 17.02 | 3,144.5 | 0.99× |
| `OpenVLA-7B-OFT` | 4 | 17.52 / 21.42 | 10,381.4 | 16.00 / 20.31 | 11,994.2 | 1.16× |
| `OpenVLA-7B-OFT` | 8 | 19.77 / 20.98 | 17,779.4 | 16.35 / 19.98 | 22,273.1 | 1.25× |
| `OpenVLA-7B-OFT` | 16 | 30.22 / 40.03 | 21,827.3 | 16.90 / 21.39 | 36,600.5 | 1.68× |
| `OpenVLA-7B-OFT` | 32 | 60.82 / 71.34 | 23,137.0 | 30.56 / 36.88 | 44,180.9 | 1.91× |
| `OpenVLA-7B-OFT` | 64 | 121.98 / 130.00 | 23,678.2 | 60.38 / 70.09 | **45,798.3** | 1.93× |

### C. Open-Loop Poisson Robot Fleet Arrivals (`50, 100, 500, 1,000 Hz`: `1-Host DP=8` vs. `2-Host DP=16`)

| Model | Target Rate (Hz) | `1-Host DP=8` Achieved (Hz) | `1-Host DP=8` `p50` / `p99` (ms) | `2-Host DP=16` Achieved (Hz) | `2-Host DP=16` `p50` / `p99` (ms) | `2-Host DP=16` Actions/s | `2-Host p99` Targets (`100 / 50 / 20 Hz`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `pi0.5-Base (3.6B)` | 50 | 39.8 | 7.80 / 10.04 | 52.4 | 8.04 / 8.88 | 2,621.0 | `100 / 50 / 20 Hz` |
| `pi0.5-Base (3.6B)` | 100 | 101.5 | 7.84 / 11.42 | 109.5 | 8.16 / 10.96 | 5,473.7 | `— / 50 / 20 Hz` |
| `pi0.5-Base (3.6B)` | 500 | 478.2 | 9.38 / 22.80 | 498.9 | 8.00 / 13.70 | 24,946.0 | `— / 50 / 20 Hz` |
| `pi0.5-Base (3.6B)` | 1,000 | 817.8 | 34.63 / 91.44 | **941.4** | **8.12 / 13.82** | 47,072.2 | `— / 50 / 20 Hz` |
| `pi0.5-LIBERO (3.6B)` | 50 | 39.8 | 7.77 / 8.59 | 52.4 | 8.07 / 9.73 | 2,621.4 | `100 / 50 / 20 Hz` |
| `pi0.5-LIBERO (3.6B)` | 100 | 101.6 | 7.94 / 12.39 | 109.5 | 8.01 / 10.94 | 5,477.4 | `— / 50 / 20 Hz` |
| `pi0.5-LIBERO (3.6B)` | 500 | 478.7 | 9.63 / 18.91 | 499.1 | 8.03 / 13.20 | 24,956.3 | `— / 50 / 20 Hz` |
| `pi0.5-LIBERO (3.6B)` | 1,000 | 797.9 | 37.95 / 91.02 | **945.4** | **8.11 / 14.22** | 47,271.2 | `— / 50 / 20 Hz` |
| `GR00T-N1.7-3B` | 50 | 39.8 | 10.49 / 14.42 | 52.3 | 10.78 / 13.00 | 2,616.1 | `— / 50 / 20 Hz` |
| `GR00T-N1.7-3B` | 100 | 101.3 | 10.60 / 14.85 | 109.4 | 10.73 / 12.02 | 5,469.4 | `— / 50 / 20 Hz` |
| `GR00T-N1.7-3B` | 500 | 475.9 | 14.15 / 35.50 | 497.1 | 10.65 / 15.31 | 24,854.8 | `— / 50 / 20 Hz` |
| `GR00T-N1.7-3B` | 1,000 | 635.6 | 127.29 / 218.40 | **934.4** | **10.84 / 18.55** | 46,719.3 | `— / 50 / 20 Hz` |
| `Cosmos3-Edge-DROID` | 50 | 39.7 | 12.53 / 14.91 | 52.3 | 12.85 / 14.85 | 1,672.7 | `— / 50 / 20 Hz` |
| `Cosmos3-Edge-DROID` | 100 | 101.2 | 12.59 / 16.70 | 109.0 | 12.84 / 15.14 | 3,487.0 | `— / 50 / 20 Hz` |
| `Cosmos3-Edge-DROID` | 500 | 474.5 | 17.80 / 44.44 | 495.9 | 12.74 / 17.25 | 15,870.1 | `— / 50 / 20 Hz` |
| `Cosmos3-Edge-DROID` | 1,000 | 588.8 | 164.15 / 220.11 | **925.0** | **13.02 / 22.75** | 29,600.7 | `— / — / 20 Hz` |
| `OpenVLA-7B (AR)` | 50 | 38.8 | 55.59 / 68.11 | 50.7 | 55.76 / 58.50 | 50.7 | `— / — / —` |
| `OpenVLA-7B (AR)` | 100 | 97.0 | 58.30 / 114.22 | 105.6 | 55.76 / 58.84 | 105.6 | `— / — / —` |
| `OpenVLA-7B (AR)` | 500 | 142.4 | 882.09 / 908.37 | 276.8 | 442.16 / 669.27 | 276.8 | `— / — / —` |
| `OpenVLA-7B (AR)` | 1,000 | 141.1 | 883.01 / 904.43 | 280.3 | 659.23 / 675.32 | 280.3 | `— / — / —` |
| `OpenVLA-7B-OFT` | 50 | 39.7 | 15.72 / 16.91 | 52.1 | 15.88 / 17.00 | 2,606.8 | `— / 50 / 20 Hz` |
| `OpenVLA-7B-OFT` | 100 | 100.9 | 15.84 / 20.16 | 108.9 | 15.91 / 19.02 | 5,444.6 | `— / 50 / 20 Hz` |
| `OpenVLA-7B-OFT` | 500 | 472.1 | 32.24 / 91.41 | 493.6 | 15.97 / 20.87 | 24,679.6 | `— / — / 20 Hz` |
| `OpenVLA-7B-OFT` | 1,000 | 489.3 | 243.49 / 261.74 | **917.1** | **22.64 / 34.92** | 45,853.3 | `— / — / 20 Hz` |

---

## 6. Multi-Camera Views (`1, 2, 4`) × Resolution (`224², 256², 384²`) & Vision Offload

### A. End-to-End Camera & Resolution Sweep (`B = 1`, `bfloat16`)

| Model | Cameras × Resolution | VLM Tokens | TPU `p50` / `p99` (ms) | Sync Local CPU E2E `p50` (ms) | Sync Disagg CPU E2E `p50` (ms) | Pipelined Disagg E2E `p50` (ms) | `1-TC` Actions/s |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi0.5-Base (3.6B)` | `1 × 224x224` | 320 | 5.46 / 5.68 | 8.32 | 9.93 | 5.46 | 9,155.1 |
| `pi0.5-Base (3.6B)` | `1 × 256x256` | 388 | 5.68 / 6.00 | 8.80 | 10.56 | 5.68 | 8,808.4 |
| `pi0.5-Base (3.6B)` | `1 × 384x384` | 793 | 8.73 / 8.81 | 12.89 | 15.98 | 8.73 | 5,725.7 |
| `pi0.5-Base (3.6B)` | `2 × 224x224` | 576 | 6.53 / 7.20 | 11.43 | 12.07 | 6.53 | 7,660.7 |
| `pi0.5-Base (3.6B)` | `2 × 256x256` | 712 | 7.79 / 7.90 | 12.99 | 13.91 | 7.79 | 6,418.8 |
| `pi0.5-Base (3.6B)` | `2 × 384x384` | 1,522 | 14.44 / 14.95 | 21.90 | 24.18 | 14.44 | 3,461.4 |
| `pi0.5-Base (3.6B)` | `4 × 224x224` | 1,088 | 10.39 / 10.80 | 19.87 | 17.96 | 10.39 | 4,811.9 |
| `pi0.5-Base (3.6B)` | `4 × 256x256` | 1,360 | 12.99 / 13.34 | 23.29 | 21.63 | 12.99 | 3,847.9 |
| `pi0.5-Base (3.6B)` | `4 × 384x384` | 2,980 | 33.76 / 33.89 | 48.55 | 48.22 | 33.76 | 1,481.0 |
| `pi0.5-LIBERO (3.6B)` | `1 × 224x224` | 320 | 5.54 / 6.00 | 8.40 | 10.01 | 5.54 | 9,020.0 |
| `pi0.5-LIBERO (3.6B)` | `1 × 256x256` | 388 | 5.74 / 6.25 | 8.86 | 10.62 | 5.74 | 8,707.6 |
| `pi0.5-LIBERO (3.6B)` | `1 × 384x384` | 793 | 8.72 / 8.93 | 12.88 | 15.97 | 8.72 | 5,736.0 |
| `pi0.5-LIBERO (3.6B)` | `2 × 224x224` | 576 | 6.49 / 7.24 | 11.39 | 12.03 | 6.49 | 7,709.4 |
| `pi0.5-LIBERO (3.6B)` | `2 × 256x256` | 712 | 7.80 / 7.93 | 13.00 | 13.92 | 7.80 | 6,409.1 |
| `pi0.5-LIBERO (3.6B)` | `2 × 384x384` | 1,522 | 14.49 / 14.97 | 21.95 | 24.23 | 14.49 | 3,451.8 |
| `pi0.5-LIBERO (3.6B)` | `4 × 224x224` | 1,088 | 10.37 / 10.46 | 19.85 | 17.94 | 10.37 | 4,822.7 |
| `pi0.5-LIBERO (3.6B)` | `4 × 256x256` | 1,360 | 13.00 / 13.15 | 23.30 | 21.64 | 13.00 | 3,847.2 |
| `pi0.5-LIBERO (3.6B)` | `4 × 384x384` | 2,980 | 33.58 / 33.79 | 48.37 | 48.04 | 33.58 | 1,489.1 |
| `GR00T-N1.7-3B` | `1 × 224x224` | 260 | 8.18 / 8.76 | 11.04 | 12.65 | 8.18 | 6,115.3 |
| `GR00T-N1.7-3B` | `1 × 256x256` | 320 | 8.33 / 8.84 | 11.45 | 13.21 | 8.33 | 6,004.3 |
| `GR00T-N1.7-3B` | `1 × 384x384` | 640 | 9.88 / 10.47 | 14.04 | 17.13 | 9.88 | 5,063.1 |
| `GR00T-N1.7-3B` | `2 × 224x224` | 456 | 8.81 / 8.92 | 13.71 | 14.35 | 8.81 | 5,673.5 |
| `GR00T-N1.7-3B` | `2 × 256x256` | 576 | 9.28 / 9.56 | 14.48 | 15.40 | 9.28 | 5,388.9 |
| `GR00T-N1.7-3B` | `2 × 384x384` | 1,216 | 12.83 / 13.35 | 20.29 | 22.57 | 12.83 | 3,898.6 |
| `GR00T-N1.7-3B` | `4 × 224x224` | 848 | 11.31 / 11.45 | 20.79 | 18.88 | 11.31 | 4,419.1 |
| `GR00T-N1.7-3B` | `4 × 256x256` | 1,088 | 12.00 / 12.15 | 22.30 | 20.64 | 12.00 | 4,165.5 |
| `GR00T-N1.7-3B` | `4 × 384x384` | 2,368 | 23.13 / 23.50 | 37.92 | 37.59 | 23.13 | 2,161.5 |
| `Cosmos3-Edge-DROID` | `1 × 224x224` | 260 | 10.12 / 10.61 | 13.14 | 14.53 | 10.12 | 3,161.4 |
| `Cosmos3-Edge-DROID` | `1 × 256x256` | 320 | 10.28 / 10.57 | 13.46 | 15.08 | 10.28 | 3,112.5 |
| `Cosmos3-Edge-DROID` | `1 × 384x384` | 640 | 12.12 / 12.50 | 16.35 | 19.33 | 12.12 | 2,641.1 |
| `Cosmos3-Edge-DROID` | `2 × 224x224` | 456 | 10.82 / 11.19 | 15.64 | 16.47 | 10.82 | 2,958.2 |
| `Cosmos3-Edge-DROID` | `2 × 256x256` | 576 | 11.38 / 11.90 | 16.61 | 17.57 | 11.38 | 2,811.1 |
| `Cosmos3-Edge-DROID` | `2 × 384x384` | 1,216 | 15.91 / 16.42 | 23.30 | 25.53 | 15.91 | 2,011.8 |
| `Cosmos3-Edge-DROID` | `4 × 224x224` | 848 | 13.85 / 14.27 | 23.21 | 21.15 | 13.85 | 2,311.0 |
| `Cosmos3-Edge-DROID` | `4 × 256x256` | 1,088 | 14.89 / 15.09 | 25.19 | 23.14 | 14.89 | 2,148.8 |
| `Cosmos3-Edge-DROID` | `4 × 384x384` | 2,368 | 28.31 / 29.95 | 43.17 | 42.45 | 28.31 | 1,130.3 |
| `OpenVLA-7B (AR)` | `1 × 224x224` | 304 | 49.21 / 49.94 | 52.23 | 53.62 | 49.21 | 20.3 |
| `OpenVLA-7B (AR)` | `1 × 256x256` | 372 | 50.25 / 54.10 | 53.43 | 55.05 | 50.25 | 19.9 |
| `OpenVLA-7B (AR)` | `1 × 384x384` | 777 | 60.92 / 62.64 | 65.15 | 68.13 | 60.92 | 16.4 |
| `OpenVLA-7B (AR)` | `2 × 224x224` | 560 | 54.49 / 54.79 | 59.31 | 60.14 | 54.49 | 18.4 |
| `OpenVLA-7B (AR)` | `2 × 256x256` | 696 | 57.44 / 58.46 | 62.67 | 63.63 | 57.44 | 17.4 |
| `OpenVLA-7B (AR)` | `2 × 384x384` | 1,506 | 91.22 / 91.31 | 98.61 | 100.84 | 91.22 | 11.0 |
| `OpenVLA-7B (AR)` | `4 × 224x224` | 1,072 | 73.33 / 73.46 | 82.69 | 80.63 | 73.33 | 13.6 |
| `OpenVLA-7B (AR)` | `4 × 256x256` | 1,344 | 82.26 / 82.43 | 92.56 | 90.51 | 82.26 | 12.2 |
| `OpenVLA-7B (AR)` | `4 × 384x384` | 2,964 | 157.69 / 157.80 | 172.55 | 171.83 | 157.69 | 6.3 |
| `OpenVLA-7B-OFT` | `1 × 224x224` | 304 | 10.31 / 12.38 | 13.33 | 14.72 | 10.31 | 4,850.2 |
| `OpenVLA-7B-OFT` | `1 × 256x256` | 372 | 10.81 / 11.06 | 13.99 | 15.61 | 10.81 | 4,626.8 |
| `OpenVLA-7B-OFT` | `1 × 384x384` | 777 | 20.62 / 21.01 | 24.85 | 27.83 | 20.62 | 2,425.2 |
| `OpenVLA-7B-OFT` | `2 × 224x224` | 560 | 14.76 / 14.85 | 19.58 | 20.41 | 14.76 | 3,386.9 |
| `OpenVLA-7B-OFT` | `2 × 256x256` | 696 | 18.12 / 18.44 | 23.35 | 24.31 | 18.12 | 2,759.0 |
| `OpenVLA-7B-OFT` | `2 × 384x384` | 1,506 | 43.00 / 43.15 | 50.39 | 52.62 | 43.00 | 1,162.8 |
| `OpenVLA-7B-OFT` | `4 × 224x224` | 1,072 | 26.60 / 26.81 | 35.96 | 33.90 | 26.60 | 1,879.8 |
| `OpenVLA-7B-OFT` | `4 × 256x256` | 1,344 | 34.17 / 34.34 | 44.47 | 42.42 | 34.17 | 1,463.3 |
| `OpenVLA-7B-OFT` | `4 × 384x384` | 2,964 | 97.81 / 97.99 | 112.67 | 111.95 | 97.81 | 511.2 |

### B. Disaggregated Vision Preprocessing Microbenchmark (`640×480` JPEG → Normalized Tensor)

Raw outputs: [`results/host0_vision_preproc.json`](./results/host0_vision_preproc.json).

| Camera Views × Target Resolution | Local TPU-Host CPU (ms) | Disaggregated C3 Xeon Compute (ms) | Disaggregated C3 Xeon HTTP RTT (ms) | Server Compute vs. Local CPU | Sync HTTP RTT vs. Local CPU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `1 × 224×224` | 2.86 | 3.78 | 4.47 | 0.76× | 0.64× |
| `1 × 256×256` | 3.12 | 4.20 | 4.88 | 0.74× | 0.64× |
| `1 × 384×384` | 4.16 | 6.44 | 7.25 | 0.65× | 0.57× |
| `2 × 224×224` | 4.90 | 4.61 | 5.54 | 1.06× | 0.88× |
| `2 × 256×256` | 5.20 | 5.14 | 6.12 | 1.01× | 0.85× |
| `2 × 384×384` | 7.46 | 8.43 | 9.74 | 0.88× | 0.77× |
| `4 × 224×224` | 9.48 | 6.06 | **7.57** | **1.56×** | **1.25×** |
| `4 × 256×256` | 10.30 | 7.03 | **8.64** | **1.47×** | **1.19×** |
| `4 × 384×384` | 14.79 | 11.95 | 14.46 | 1.24× | 1.02× |

---

## 7. Eager vs. `torch.compile`, ODE Solver (`4` vs. `10` Steps), & Action Horizon (`16, 32, 50`) Ablations

| Model | Ablation Suite | Backend | Batch `B` | Chunk | Steps | TPU `p50` / `p99` (ms) | `1-TC` Actions/s | `1-Host DP=8` Actions/s |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pi0.5-Base (3.6B)` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 1 | 50 | 4 | 57.58 / 70.80 | 868.3 | — |
| `pi0.5-Base (3.6B)` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 8 | 50 | 4 | 84.77 / 86.21 | 4,718.7 | — |
| `pi0.5-Base (3.6B)` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 1 | 50 | 10 | 9.41 / 11.36 | 5,313.5 | 26,666.5 |
| `pi0.5-Base (3.6B)` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 8 | 50 | 10 | 44.57 / 44.69 | 8,973.7 | 70,828.0 |
| `pi0.5-Base (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 16 | 4 | 6.32 / 6.93 | 2,531.7 | — |
| `pi0.5-Base (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 16 | 4 | 37.79 / 38.33 | 3,386.8 | 26,597.4 |
| `pi0.5-Base (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 32 | 4 | 6.44 / 7.19 | 4,967.8 | — |
| `pi0.5-Base (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 32 | 4 | 37.93 / 38.05 | 6,748.6 | 53,153.3 |
| `pi0.5-LIBERO (3.6B)` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 1 | 50 | 4 | 44.82 / 46.10 | 1,115.5 | — |
| `pi0.5-LIBERO (3.6B)` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 8 | 50 | 4 | 80.37 / 80.95 | 4,976.8 | — |
| `pi0.5-LIBERO (3.6B)` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 1 | 50 | 10 | 9.29 / 10.00 | 5,383.8 | 24,570.0 |
| `pi0.5-LIBERO (3.6B)` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 8 | 50 | 10 | 44.40 / 44.54 | 9,009.7 | 70,640.0 |
| `pi0.5-LIBERO (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 16 | 4 | 6.36 / 6.61 | 2,517.5 | — |
| `pi0.5-LIBERO (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 16 | 4 | 37.70 / 37.84 | 3,394.9 | 26,597.4 |
| `pi0.5-LIBERO (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 32 | 4 | 6.45 / 6.69 | 4,958.2 | — |
| `pi0.5-LIBERO (3.6B)` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 32 | 4 | 37.86 / 38.06 | 6,762.4 | 53,264.0 |
| `GR00T-N1.7-3B` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 1 | 50 | 4 | 59.39 / 61.97 | 841.9 | — |
| `GR00T-N1.7-3B` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 8 | 50 | 4 | 89.66 / 92.14 | 4,461.4 | — |
| `GR00T-N1.7-3B` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 1 | 50 | 10 | 18.43 / 18.62 | 2,713.6 | 16,440.5 |
| `GR00T-N1.7-3B` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 8 | 50 | 10 | 99.53 / 99.70 | 4,019.1 | 31,863.0 |
| `GR00T-N1.7-3B` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 16 | 4 | 8.77 / 8.93 | 1,825.3 | — |
| `GR00T-N1.7-3B` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 16 | 4 | 52.25 / 52.40 | 2,449.9 | 19,288.0 |
| `GR00T-N1.7-3B` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 32 | 4 | 9.15 / 11.97 | 3,496.6 | — |
| `GR00T-N1.7-3B` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 32 | 4 | 53.16 / 53.43 | 4,815.6 | 37,772.2 |
| `Cosmos3-Edge-DROID` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 1 | 32 | 4 | 62.25 / 84.85 | 514.0 | — |
| `Cosmos3-Edge-DROID` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 8 | 32 | 4 | 91.98 / 96.66 | 2,783.3 | — |
| `Cosmos3-Edge-DROID` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 1 | 32 | 10 | 22.13 / 22.84 | 1,445.8 | 9,566.4 |
| `Cosmos3-Edge-DROID` | `ode_solver_10step_vs_4step` | `torchtpu-compiled` | 8 | 32 | 10 | 62.02 / 62.26 | 4,127.6 | 32,606.4 |
| `Cosmos3-Edge-DROID` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 16 | 4 | 10.79 / 11.39 | 1,482.4 | — |
| `Cosmos3-Edge-DROID` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 16 | 4 | 42.24 / 42.53 | 3,030.5 | 23,758.7 |
| `Cosmos3-Edge-DROID` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 50 | 4 | 11.40 / 11.95 | 4,385.1 | — |
| `Cosmos3-Edge-DROID` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 50 | 4 | 44.17 / 44.25 | 9,056.2 | **71,460.5** |
| `OpenVLA-7B (AR)` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 1 | 1 | 7 | 118.77 / 122.09 | 8.4 | — |
| `OpenVLA-7B (AR)` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 8 | 1 | 7 | 264.04 / 264.28 | 30.3 | — |
| `OpenVLA-7B-OFT` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 1 | 50 | 1 | 37.40 / 38.62 | 1,337.1 | — |
| `OpenVLA-7B-OFT` | `eager_vs_compiled_ablation` | `torchtpu-eager` | 8 | 50 | 1 | 228.59 / 228.76 | 1,749.8 | — |
| `OpenVLA-7B-OFT` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 16 | 1 | 14.05 / 14.31 | 1,138.4 | — |
| `OpenVLA-7B-OFT` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 16 | 1 | 109.54 / 109.83 | 1,168.5 | 9,293.9 |
| `OpenVLA-7B-OFT` | `action_horizon_sweep` | `torchtpu-compiled` | 1 | 32 | 1 | 13.99 / 14.39 | 2,287.2 | — |
| `OpenVLA-7B-OFT` | `action_horizon_sweep` | `torchtpu-compiled` | 8 | 32 | 1 | 110.83 / 115.55 | 2,309.8 | 18,336.3 |
