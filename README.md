# Physical AI Vision-Language-Action (VLA) Serving on Google Cloud TPU v7x

Single-TensorCore latency, intra-chip tensor parallelism (`TP=2`), and multi-host data-parallel (`DP=16`) serving benchmarks across 6 open-weight robotics VLA policies on Google Cloud TPU v7x (Ironwood) with PyTorch and TorchTPU.

**Summary:** With `torch.compile(dynamic=False)` and one dedicated OS worker per TensorCore, single-TensorCore (`B = 1`, `bfloat16`) policy latency drops to **6.49–6.53 ms `p50` on `pi0.5` (`6.91–8.82×` faster than eager TorchTPU)** and **9.28 ms `p50` on `GR00T-N1.7-3B` (`6.40×` faster than eager)**, while a 2-host `v7x` slice (`DP=16`, 8 chips) sustains **941–945 Hz open-loop Poisson robot arrivals on `pi0.5` at 13.82–14.22 ms `p99`** and **145,377 actions/s** batched throughput (`1.70×` higher than single-host `DP=8`). For `7.54B` `OpenVLA`, replacing 7-step autoregressive decoding (`54.49 ms`) with 1-pass parallel chunking (`OpenVLA-OFT`) lowers `B = 1` latency to **14.76 ms (`3.69×` faster)** and **11.38 ms with intra-chip `TP=2` (`4.79×` faster than 1-core autoregressive `OpenVLA-7B`)**.

---

## 1. Model, Framework, and Hardware

- **Models (4 VLA Architecture Families, 6 Checkpoints):**
  - **Physical Intelligence `pi0.5` (~3.6B params, `6.74 GiB` `bfloat16` / `3.37 GiB` `fp8`):** `lerobot/pi05_base` and `lerobot/pi05_libero_finetuned_v044`. 27-layer SigLIP vision tower (`1152` width) + 18-layer PaliGemma prefix (`2048` width) + 18-layer flow-matching action expert (`1024` width) running 1 VLM prefill + 4-step (or 10-step) ODE denoising (`chunk = 50`).
  - **NVIDIA `GR00T-N1.7-3B` (~3.2B params, `6.44 GiB` `bfloat16` / `3.22 GiB` `fp8`):** `nvidia/GR00T-N1.7-3B`. 27-layer SigLip2 vision tower + 16-layer Cosmos-Reason2-2B VLM (`2048` width) + 32-layer AdaLN diffusion transformer (`1536` width) running 4-step flow ODE (`chunk = 50`).
  - **NVIDIA `Cosmos3-Edge` MoT (~4.3B params, `8.50 GiB` `bfloat16` / `4.25 GiB` `fp8`):** `nvidia/Cosmos3-Edge-Policy-DROID`. 27-layer SigLIP tower + 28-layer Mixture-of-Transformers (`2048` VLM / `1024` action width) running 4-step joint flow ODE (`chunk = 32`).
  - **`OpenVLA-7B` & `OpenVLA-OFT` (~7.54B params, `14.05–14.50 GiB` `bfloat16` / `7.20–7.25 GiB` `fp8`):** `openvla/openvla-7b` (dual `DinoV2` + `SigLIP` encoders + 32-layer `Llama-2-7B` running 7-step KV-cached autoregressive token decode, `chunk = 1`) and `moojink/openvla-7b-oft-finetuned-libero-spatial` (same backbone running 1-pass bidirectional parallel action chunking, `chunk = 50`).
- **Framework:** Python 3.12, PyTorch 2.10, `torch_tpu` 0.2.0 (`libtpu` 0.0.50, `tpu_dist` ICI backend), `torch.compile(dynamic=False)` with shared Tier-2 XLA compilation cache, `torch._scaled_mm` dynamic `float8_e4m3fn`, and optional C3 Xeon CPU vision preprocessing offload (`multimodal-cpu-offload-svc`).
- **Hardware:** Google Cloud TPU v7x (Ironwood) on GKE (`192 GiB` HBM3e per physical chip across 2 hardware TensorCores = `96 GiB` HBM3e per TensorCore):
  - Single host: `4` physical `v7x` chips = `8` TensorCores (`768 GiB` HBM3e, 8 `torchrun` worker processes)
  - Two-host slice: `2x2x2` ICI topology = `8` physical `v7x` chips = `16` TensorCores (`1.536 TB` HBM3e)

---

## 2. Main Results

### A. Single-Request Policy Latency (`B = 1`, Warm `p50` / `p99`)

| Checkpoint (`Cameras × Resolution`, Steps, Chunk) | Eager `1-TC` `bfloat16` `p50` | Compiled `1-TC` `bfloat16` `p50` / `p99` | Compiled `1-Chip TP=2` (`2 TC`) `p50` | Comparison |
| :--- | ---: | ---: | ---: | :--- |
| `lerobot/pi05_base` (`2×224²`, 4-step ODE, `chunk=50`) | 57.58 ms | **6.53** / 7.20 ms | 11.70 ms | **8.82× faster** than eager `1-TC`; `TP=2` is 0.56× at `B=1` due to ICI `all_reduce` overhead |
| `lerobot/pi05_libero_finetuned_v044` (`2×224²`, 4-step ODE, `chunk=50`) | 44.82 ms | **6.49** / 7.24 ms | 9.59 ms | **6.91× faster** than eager `1-TC`; meets `<10 ms` (`100 Hz`) control loop |
| `nvidia/GR00T-N1.7-3B` (`2×256²`, 4-step ODE, `chunk=50`) | 59.39 ms | **9.28** / 9.56 ms | 10.79 ms | **6.40× faster** than eager `1-TC`; meets `<10 ms` (`100 Hz`) control loop |
| `nvidia/Cosmos3-Edge-Policy-DROID` (`2×256²`, 4-step ODE, `chunk=32`) | 62.25 ms | **11.38** / 11.90 ms | 13.00 ms | **5.47× faster** than eager `1-TC`; meets `<20 ms` (`50 Hz`) control loop |
| `openvla/openvla-7b` (`2×224²`, 7-step AR, `chunk=1`) | 118.77 ms | 54.49 / 54.79 ms | **34.56 ms** (`42.87 ms` `1-TC fp8`) | `TP=2` is **1.58× faster** than compiled `1-TC bfloat16` (`3.44×` vs eager), bringing 7-step AR under `50 ms` (`20 Hz`) |
| `moojink/openvla-7b-oft-finetuned-libero-spatial` (`2×224²`, 1-pass, `chunk=50`) | 37.40 ms | 14.76 / 14.85 ms | **11.38 ms** (`14.17 ms` `1-TC fp8`) | **3.69× faster** (`1-TC`) and **4.79× faster** (`TP=2`) than `1-TC` autoregressive `OpenVLA-7B` |

*Notes:* Latencies are warm steady-state TPU policy execution times after XLA compilation (and identical to end-to-end pipelined latency when double-buffered CPU vision preprocessing overlaps behind TPU execution). See [`APPENDIX.md`](./APPENDIX.md) for synchronous CPU preprocessing latencies, `fp8` `B = 1..32` sweeps, and methodology notes.

### B. Multi-Host Fleet Serving & Batched Throughput (`1-Host DP=8` vs. `2-Host DP=16`)

| Checkpoint (`Default Chunk`) | `1-Host DP=8` Peak (actions/s) | `2-Host DP=16` Peak (actions/s) | `1-Host DP=8` @ `1,000 Hz` Poisson (`p50` / `p99`, Achieved) | `2-Host DP=16` @ `1,000 Hz` Poisson (`p50` / `p99`, Achieved) | Comparison (`2-Host DP=16` vs. `1-Host DP=8`) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `lerobot/pi05_base` (`50`) | 85,322 (`fp8 B=32`) | **145,377** (`bf16 B=8`) | 34.63 / 91.44 ms (`818 Hz`) | **8.12 / 13.82 ms** (`941 Hz`) | **1.70×** peak throughput; **6.6× lower `p99`** at `1,000 Hz` |
| `lerobot/pi05_libero_finetuned_v044` (`50`) | 85,402 (`fp8 B=32`) | **143,498** (`bf16 B=8`) | 37.95 / 91.02 ms (`798 Hz`) | **8.11 / 14.22 ms** (`945 Hz`) | **1.68×** peak throughput; **6.4× lower `p99`** at `1,000 Hz` |
| `nvidia/GR00T-N1.7-3B` (`50`) | 63,619 (`bf16 B=4`) | **111,248** (`bf16 B=16`) | 127.29 / 218.40 ms (`636 Hz`) | **10.84 / 18.55 ms** (`934 Hz`) | **1.75×** peak throughput; **11.8× lower `p99`** at `1,000 Hz` |
| `nvidia/Cosmos3-Edge-Policy-DROID` (`32`) | 47,048 (`bf16 B=8`) | **83,492** (`bf16 B=8`) | 164.15 / 220.11 ms (`589 Hz`) | **13.02 / 22.75 ms** (`925 Hz`) | **1.77×** same-chunk throughput (`71,461` on `DP=8` at `chunk=50`) |
| `openvla/openvla-7b` (`1`) | 407.3 (`fp8 B=32`) | **660.0** (`bf16 B=16`) | 883.01 / 904.43 ms (`141 Hz`) | 659.23 / 675.32 ms (`280 Hz`) | **1.62×** peak throughput (`1.88×` over `DP=8 bf16` `350.5`) |
| `moojink/openvla-7b-oft-finetuned-libero-spatial` (`50`) | 33,420 (`fp8 B=32`) | **51,181** (`bf16 B=16`) | 243.49 / 261.74 ms (`489 Hz`) | **22.64 / 34.92 ms** (`917 Hz`) | **77.5× higher** `DP=16` actions/s than 7-step AR `OpenVLA-7B` |

---

## 3. Reproducing the Main Results

### Single-Host 8-TensorCore Sweep (`1-TC`, `TP=2`, `DP=8`, Camera & Solver Sweeps)
Script: [`scripts/run_physical_ai_vla_v7x_triathlon.py`](./scripts/run_physical_ai_vla_v7x_triathlon.py) (or deploy on GKE via [`k8s/v7x-physical-ai-vla-triathlon.yaml`](./k8s/v7x-physical-ai-vla-triathlon.yaml)):

```bash
# Optional: start disaggregated CPU vision preprocessing service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# Run full single-host sweep across all 6 checkpoints (8 processes, 1 per TensorCore)
python3 scripts/run_physical_ai_vla_v7x_triathlon.py \
  --models-root=/models/models \
  --models=pi05_base,pi05_libero,gr00t_n1_7_3b,cosmos3_edge_droid,openvla_7b,openvla_7b_oft \
  --cores-per-host=8 \
  --tag=host0 \
  --output-dir=./results
```

### Two-Host 16-TensorCore Fleet Serving (`DP=16` Batch, Concurrency & Poisson Sweeps)
Scripts: [`scripts/run_physical_ai_vla_v7x_triathlon.py`](./scripts/run_physical_ai_vla_v7x_triathlon.py) and [`gce/launch_v7x_vla_triathlon.sh`](./gce/launch_v7x_vla_triathlon.sh):

```bash
# On Host 0 and Host 1: expose 8 Keep-Alive TensorCore workers (:19100..19107)
python3 scripts/run_physical_ai_vla_v7x_triathlon.py \
  --models-root=/models/models \
  --models=pi05_base,pi05_libero,gr00t_n1_7_3b,cosmos3_edge_droid,openvla_7b,openvla_7b_oft \
  --serve-worker-port=19100 &

# From coordinator: drive 2-host DP=16 batch, closed-loop concurrency, and Poisson load tests
python3 scripts/run_physical_ai_vla_v7x_triathlon.py \
  --run-multihost-dp16 \
  --worker-urls=http://HOST0_IP:19100,http://HOST1_IP:19100 \
  --models=pi05_base,pi05_libero,gr00t_n1_7_3b,cosmos3_edge_droid,openvla_7b,openvla_7b_oft \
  --output-dir=./results
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete 328-point tables (`bfloat16` vs. dynamic `fp8` across `B = 1..32`, intra-chip `TP=2`, closed-loop concurrency `CON = 1..64`, open-loop Poisson `50..1,000 Hz`, `1/2/4` cameras × `224²/256²/384²`, CPU vision offload microbenchmarks, 4-step vs. 10-step ODE, action horizon `16/32/50`), configurations where `TP=2` or `fp8` slowed `B = 1` inference, and benchmark methodology notes.
- **[`results/`](./results/):** Machine-readable outputs including [`vla_triathlon_master_summary.json`](./results/vla_triathlon_master_summary.json), [`host0_results.jsonl`](./results/host0_results.jsonl), [`host1_results.jsonl`](./results/host1_results.jsonl), [`multihost_dp16_results.jsonl`](./results/multihost_dp16_results.jsonl), and [`host0_vision_preproc.json`](./results/host0_vision_preproc.json).
