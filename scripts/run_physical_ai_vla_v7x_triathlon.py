#!/usr/bin/env python3
# Copyright 2026 Google LLC
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
"""Physical AI VLA Quad-Architecture Triathlon on Cloud TPU v7x (Ironwood).

Multi-Process Hardware-Isolated Harness (`torchrun --nproc_per_node=8` + `tpu_dist` ICI):
  - Spawns 8 independent OS worker processes per TPU v7x-8 host (1 OS process per
    hardware TensorCore, sharing XLA executables via TorchTPU's Tier-2 Compilation Cache
    with zero Python GIL contention).
  - Benchmarks all 4 Physical AI VLA model families (6 real safetensors checkpoints):
      1. `lerobot/pi05_base` (`pi0.5-Base` 3.6B: 27L SigLIP + 18L PaliGemma + 18L Flow Action Expert)
      2. `lerobot/pi05_libero_finetuned_v044` (`pi0.5-LIBERO` Finetuned 3.6B)
      3. `nvidia/GR00T-N1.7-3B` (27L SigLip2 + 16L Cosmos-Reason2-2B + 32L AdaLN Action DiT)
      4. `nvidia/Cosmos3-Edge-Policy-DROID` (27L SigLIP + 28L Cosmos3 Mixture-of-Transformers Flow Policy)
      5. `openvla/openvla-7b` (Dual DinoV2 + SigLIP + 32L Llama-2-7B 7-Token KV-Cached AR Decode)
      6. `moojink/openvla-7b-oft-finetuned-libero-spatial` (1-Pass Bidirectional Parallel Action Chunking)

Sweeps:
  - Precision & Compilation: `BF16` vs `FP8 Dynamic` (`torch._scaled_mm` E4M3), `Eager` vs `torch.compile(dynamic=False)`
  - Sharding / Scaling Topologies:
      * `1-TC (DP=1)` single TensorCore latency floor
      * `1-Chip Intra-Chip TP=2 (2 TC)` with hardware ICI `tpu_dist` `all_reduce`
      * `1-Host DP=8 (4 Chips = 8 TC)` true 8-process hardware wave & HTTP Keep-Alive serving
      * `2-Host Slice DP=16 (8 Chips = 16 TC)` cross-host fleet serving across 16 TensorCore endpoints
  - Batch Sizes (`B = 1, 2, 4, 8, 16, 32`), Closed-Loop Concurrency (`CON = 1, 4, 8, 16, 32, 64`),
    and Open-Loop Poisson robot fleet arrivals (`50, 100, 500, 1000 Hz`)
  - Camera Views (`1, 2, 4`) x Resolutions (`224x224, 256x256, 384x384`) x Action Horizons (`16, 32, 50`)
    x ODE Solver Steps (`4`-step vs `10`-step vs `7`-token AR vs `1`-pass OFT)
  - Disaggregated Vision Preprocessing (`multimodal-cpu-offload-svc:8090/preprocess_vision`)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import random
import shutil


class FastThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 256
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.parse
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cpu_offload_client import benchmark_vision_preproc_ablation, check_offload_health  # noqa: E402
from tpu_single_host_setup import start_single_host_metadata_env  # noqa: E402


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(tag: str, msg: str) -> None:
    print(f"[{utc_iso()}][vla-triathlon][{tag}] {msg}", flush=True)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (idx - lo))


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "pi05_base": {
        "model_key": "pi05_base",
        "family": "pi05",
        "repo_id": "lerobot/pi05_base",
        "display_name": "lerobot/pi05_base (pi0.5-Base 3.6B)",
        "paradigm": "PaliGemma VLM + 300M Flow-Matching Action Expert",
        "vis_layers": 27,
        "vis_dim": 1152,
        "vis_heads": 16,
        "vis_mlp": 4304,
        "patch_size": 14,
        "default_res": 224,
        "vlm_layers": 18,
        "vlm_dim": 2048,
        "vlm_q_heads": 8,
        "vlm_kv_heads": 1,
        "vlm_hd": 256,
        "vlm_mlp": 16384,
        "exp_layers": 18,
        "exp_dim": 1024,
        "exp_q_heads": 8,
        "exp_kv_heads": 1,
        "exp_hd": 256,
        "exp_mlp": 4096,
        "act_dim": 32,
        "default_chunk": 50,
        "default_steps": 10,
        "fast_steps": 4,
        "lang_tokens": 64,
    },
    "pi05_libero": {
        "model_key": "pi05_libero",
        "family": "pi05",
        "repo_id": "lerobot/pi05_libero_finetuned_v044",
        "display_name": "lerobot/pi05_libero_finetuned_v044 (pi0.5-LIBERO 3.6B)",
        "paradigm": "PaliGemma VLM + 300M Flow-Matching Action Expert (LIBERO)",
        "vis_layers": 27,
        "vis_dim": 1152,
        "vis_heads": 16,
        "vis_mlp": 4304,
        "patch_size": 14,
        "default_res": 224,
        "vlm_layers": 18,
        "vlm_dim": 2048,
        "vlm_q_heads": 8,
        "vlm_kv_heads": 1,
        "vlm_hd": 256,
        "vlm_mlp": 16384,
        "exp_layers": 18,
        "exp_dim": 1024,
        "exp_q_heads": 8,
        "exp_kv_heads": 1,
        "exp_hd": 256,
        "exp_mlp": 4096,
        "act_dim": 32,
        "default_chunk": 50,
        "default_steps": 10,
        "fast_steps": 4,
        "lang_tokens": 64,
    },
    "gr00t_n1_7_3b": {
        "model_key": "gr00t_n1_7_3b",
        "family": "gr00t",
        "repo_id": "nvidia/GR00T-N1.7-3B",
        "display_name": "nvidia/GR00T-N1.7-3B (Humanoid VLA)",
        "paradigm": "SigLip2 + Cosmos-Reason2-2B + 32L Action DiT",
        "vis_layers": 27,
        "vis_dim": 1152,
        "vis_heads": 16,
        "vis_mlp": 4304,
        "patch_size": 16,
        "default_res": 256,
        "vlm_layers": 16,
        "vlm_dim": 2048,
        "vlm_q_heads": 16,
        "vlm_kv_heads": 8,
        "vlm_hd": 128,
        "vlm_mlp": 8192,
        "dit_layers": 32,
        "dit_dim": 1536,
        "dit_heads": 32,
        "dit_hd": 48,
        "dit_mlp": 6144,
        "act_dim": 132,
        "default_chunk": 50,
        "default_steps": 4,
        "fast_steps": 4,
        "lang_tokens": 64,
    },
    "cosmos3_edge_droid": {
        "model_key": "cosmos3_edge_droid",
        "family": "cosmos3_edge",
        "repo_id": "nvidia/Cosmos3-Edge-Policy-DROID",
        "display_name": "nvidia/Cosmos3-Edge-Policy-DROID (MoT Action Policy)",
        "paradigm": "SigLIP + 28L Cosmos3 Mixture-of-Transformers Flow Policy",
        "vis_layers": 27,
        "vis_dim": 1152,
        "vis_heads": 16,
        "vis_mlp": 4304,
        "patch_size": 16,
        "default_res": 256,
        "vlm_layers": 28,
        "vlm_dim": 2048,
        "vlm_q_heads": 16,
        "vlm_kv_heads": 8,
        "vlm_hd": 128,
        "vlm_mlp": 9216,
        "act_dim": 64,
        "default_chunk": 32,
        "default_steps": 4,
        "fast_steps": 4,
        "lang_tokens": 64,
    },
    "openvla_7b": {
        "model_key": "openvla_7b",
        "family": "openvla_ar",
        "repo_id": "openvla/openvla-7b",
        "display_name": "openvla/openvla-7b (7-Token AR Decode)",
        "paradigm": "Dual DinoV2+SigLIP + 32L Llama-2-7B 7-Token AR Decode",
        "vis_layers": 27,
        "vis_dim": 1152,
        "dino_dim": 1024,
        "vis_heads": 16,
        "vis_mlp": 4304,
        "patch_size": 14,
        "default_res": 224,
        "vlm_layers": 32,
        "vlm_dim": 4096,
        "vlm_q_heads": 32,
        "vlm_kv_heads": 32,
        "vlm_hd": 128,
        "vlm_mlp": 11008,
        "vocab_size": 32064,
        "act_dim": 7,
        "default_chunk": 1,
        "default_steps": 7,
        "fast_steps": 7,
        "lang_tokens": 48,
    },
    "openvla_7b_oft": {
        "model_key": "openvla_7b_oft",
        "family": "openvla_oft",
        "repo_id": "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "display_name": "moojink/openvla-7b-oft-finetuned-libero-spatial (Parallel Chunking)",
        "paradigm": "Dual DinoV2+SigLIP + 32L Llama-2-7B 1-Pass Parallel Action Chunking",
        "vis_layers": 27,
        "vis_dim": 1152,
        "dino_dim": 1024,
        "vis_heads": 16,
        "vis_mlp": 4304,
        "patch_size": 14,
        "default_res": 224,
        "vlm_layers": 32,
        "vlm_dim": 4096,
        "vlm_q_heads": 32,
        "vlm_kv_heads": 32,
        "vlm_hd": 128,
        "vlm_mlp": 11008,
        "act_dim": 7,
        "default_chunk": 50,
        "default_steps": 1,
        "fast_steps": 1,
        "lang_tokens": 48,
    },
}


def measure_vision_preproc_matrix() -> dict[str, dict[str, Any]]:
    health = check_offload_health()
    log("preproc", f"multimodal-cpu-offload-svc health status={health.get('status')}")
    out: dict[str, dict[str, Any]] = {}
    for n_cams in (1, 2, 4):
        for res in (224, 256, 384):
            key = f"{n_cams}x{res}"
            _ = benchmark_vision_preproc_ablation(num_cams=n_cams, src_w=640, src_h=480, dst_w=res, dst_h=res)
            loc_s: list[float] = []
            rtt_s: list[float] = []
            srv_s: list[float] = []
            for _ in range(5):
                rec = benchmark_vision_preproc_ablation(num_cams=n_cams, src_w=640, src_h=480, dst_w=res, dst_h=res)
                loc_s.append(float(rec["local_cpu_preproc_ms"]))
                if rec.get("offload_ok"):
                    rtt_s.append(float(rec["disagg_cpu_rtt_ms"]))
                    srv_s.append(float(rec.get("disagg_cpu_preproc_ms", 0.0)))
            loc_med = round(statistics.median(loc_s), 2)
            if rtt_s:
                rtt_med = round(statistics.median(rtt_s), 2)
                srv_med = round(statistics.median(srv_s), 2)
                entry = {
                    "num_cams": n_cams,
                    "dst_res": res,
                    "transform": f"640x480->{res}x{res}",
                    "offload_ok": True,
                    "local_cpu_preproc_ms": loc_med,
                    "disagg_cpu_rtt_ms": rtt_med,
                    "disagg_cpu_preproc_ms": srv_med,
                    "encoder_tier": "disaggregated-xeon-cpu-pool",
                    "sync_speedup_x": round(loc_med / max(rtt_med, 1e-3), 2),
                    "server_compute_speedup_x": round(loc_med / max(srv_med, 1e-3), 2),
                }
            else:
                entry = {
                    "num_cams": n_cams,
                    "dst_res": res,
                    "transform": f"640x480->{res}x{res}",
                    "offload_ok": False,
                    "local_cpu_preproc_ms": loc_med,
                    "disagg_cpu_rtt_ms": loc_med,
                    "disagg_cpu_preproc_ms": None,
                    "sync_speedup_x": 1.0,
                }
            out[key] = entry
            log("preproc", f"Vision preproc {key}: local={loc_med}ms, disagg_rtt={entry['disagg_cpu_rtt_ms']}ms")
    return out


def stage_and_inspect_safetensors(model_dir: Path, shm_slug: str) -> tuple[Path, dict[str, Any]]:
    from safetensors import safe_open

    sfs = sorted([p for p in model_dir.rglob("*.safetensors") if "/.cache/" not in str(p)])
    if not sfs:
        raise FileNotFoundError(f"No .safetensors files found under {model_dir}")

    shm_dir = Path("/dev/shm") / f"_vla_stage_{shm_slug}"
    shm_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def _cp(src: Path) -> Path:
        rel = src.relative_to(model_dir)
        dst = shm_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
        return dst

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sfs), 6)) as pool:
        local_sfs = list(pool.map(_cp, sfs))

    copy_s = time.perf_counter() - t0
    total_disk_bytes = sum(p.stat().st_size for p in local_sfs)
    total_tensors = 0
    total_params = 0
    dtype_hist: dict[str, int] = {}
    for sp in local_sfs:
        with safe_open(str(sp), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                sl = sf.get_slice(k)
                shape = sl.get_shape()
                dt_s = str(sl.get_dtype())
                dtype_hist[dt_s] = dtype_hist.get(dt_s, 0) + 1
                total_tensors += 1
                n_el = 1
                for dim in shape:
                    n_el *= dim
                total_params += n_el

    info = {
        "num_safetensors_files": len(local_sfs),
        "safetensors_disk_gib": round(total_disk_bytes / (1024**3), 3),
        "num_safetensors_tensors": total_tensors,
        "total_parameters_b": round(total_params / 1e9, 3),
        "bf16_equivalent_gib": round((total_params * 2) / (1024**3), 3),
        "fp8_equivalent_gib": round((total_params * 1) / (1024**3), 3),
        "dtype_histogram": dtype_hist,
        "shm_stage_seconds": round(copy_s, 2),
    }
    return shm_dir, info


class MultiEndpointKeepAlivePool:
    """Thread-safe least-inflight round-robin HTTP/1.1 Keep-Alive client across N TensorCore endpoints."""

    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.tls = threading.local()
        self.inflight = [0] * len(urls)
        self.lock = threading.Lock()
        self.rr = 0

    def wait_healthy(self, timeout_s: float = 600.0) -> None:
        for u in self.urls:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"{u}/health", timeout=2.5) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError(f"Timed out waiting for worker {u}/health")

    def call_infer(self, model_key: str, batch_size: int = 1) -> dict[str, Any]:
        with self.lock:
            n = len(self.urls)
            idx = min(range(n), key=lambda i: (self.inflight[i], (i - self.rr) % n))
            self.inflight[idx] += 1
            self.rr = (idx + 1) % n
            u = self.urls[idx]

        parsed = urllib.parse.urlsplit(u)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 19100
        key = f"{host}:{port}"
        body = json.dumps({"model_key": model_key, "batch_size": batch_size}).encode("utf-8")
        t0 = time.perf_counter()
        try:
            conns: dict[str, http.client.HTTPConnection] = getattr(self.tls, "conns", None) or {}
            self.tls.conns = conns
            for attempt in range(2):
                try:
                    conn = conns.get(key)
                    if conn is None or conn.sock is None:
                        conn = http.client.HTTPConnection(host, port, timeout=60.0)
                        conn.connect()
                        if conn.sock is not None:
                            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        conns[key] = conn
                    conn.request(
                        "POST",
                        "/infer",
                        body=body,
                        headers={"Content-Type": "application/json", "Content-Length": str(len(body)), "Connection": "keep-alive"},
                    )
                    resp = conn.getresponse()
                    raw = resp.read()
                    res = json.loads(raw.decode("utf-8"))
                    res["e2e_rpc_ms"] = (time.perf_counter() - t0) * 1000.0
                    res["endpoint_idx"] = idx
                    return res
                except Exception:
                    old = conns.pop(key, None)
                    if old is not None:
                        try:
                            old.close()
                        except Exception:
                            pass
                    if attempt == 1:
                        raise
            raise RuntimeError("unreachable")
        finally:
            with self.lock:
                self.inflight[idx] = max(0, self.inflight[idx] - 1)


def run_torchrun_vla_worker(
    models_root: Path,
    model_keys: list[str],
    preproc_json: Path,
    out_jsonl: Path,
    out_summary_json: Path,
    serve_port: int,
) -> None:
    import torch
    import torch._dynamo
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401
    from safetensors import safe_open

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    torch.accelerator.set_device_index(0)
    dev = torch.device("tpu:0")

    dist.init_process_group(backend="tpu_dist")
    gloo_pg = dist.new_group(list(range(world_size)), backend="gloo")
    tp2_groups = [dist.new_group([2 * i, 2 * i + 1], backend="tpu_dist") for i in range(world_size // 2)]
    my_tp2_grp = tp2_groups[rank // 2]

    torch._dynamo.config.cache_size_limit = 512
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = 4096

    if rank == 0:
        log("worker", f"Initialized {world_size}-process TorchTPU worker pool (`tpu_dist` ICI + shared Tier-2 XLA cache)")

    preproc_map: dict[str, dict[str, Any]] = json.loads(preproc_json.read_text()) if preproc_json.exists() else {}

    def _fp8_mm(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        in_f = w_fp8.shape[1]
        out_f = w_fp8.shape[0]
        x2d = x.reshape(-1, in_f)
        m = x2d.shape[0]
        pad_m = (16 - (m % 16)) % 16
        if pad_m:
            x2d = F.pad(x2d, (0, 0, 0, pad_m))
        a_scale = (x2d.abs().amax() / 448.0).clamp(min=1e-8).to(torch.float32)
        x_fp8 = (x2d / a_scale.to(torch.bfloat16)).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=a_scale, scale_b=w_scale, out_dtype=torch.bfloat16)
        if pad_m:
            y = y[:m]
        return y.reshape(*orig_shape, out_f)

    def _make_stage_kernels(qh: int, kvh: int, hd: int, act_id: int, is_causal: bool) -> dict[str, Any]:
        """Builds dedicated per-stage Eager & Compiled kernels so Dynamo never mixes stage constants."""
        rep = max(1, qh // kvh)

        def _bf16_eager(
            x: torch.Tensor,
            wq: torch.Tensor,
            wk: torch.Tensor,
            wv: torch.Tensor,
            wo: torch.Tensor,
            wup: torch.Tensor,
            wdn: torch.Tensor,
            wg: torch.Tensor | None,
        ) -> torch.Tensor:
            b, s, _ = x.shape
            q = F.linear(x, wq).view(b, s, qh, hd).transpose(1, 2)
            k = F.linear(x, wk).view(b, s, kvh, hd).transpose(1, 2)
            v = F.linear(x, wv).view(b, s, kvh, hd).transpose(1, 2)
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            at = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            h = x + F.linear(at.transpose(1, 2).contiguous().view(b, s, qh * hd), wo)
            if act_id == 0 and wg is not None:
                ff = F.linear(F.silu(F.linear(h, wg)) * F.linear(h, wup), wdn)
            elif act_id == 1:
                u = F.relu(F.linear(h, wup))
                ff = F.linear(u * u, wdn)
            else:
                ff = F.linear(F.gelu(F.linear(h, wup)), wdn)
            return h + ff

        def _fp8_eager(
            x: torch.Tensor,
            wq: torch.Tensor,
            sq: torch.Tensor,
            wk: torch.Tensor,
            sk: torch.Tensor,
            wv: torch.Tensor,
            sv: torch.Tensor,
            wo: torch.Tensor,
            so: torch.Tensor,
            wup: torch.Tensor,
            sup: torch.Tensor,
            wdn: torch.Tensor,
            sdn: torch.Tensor,
            wg: torch.Tensor | None,
            sg: torch.Tensor | None,
        ) -> torch.Tensor:
            b, s, _ = x.shape
            q = _fp8_mm(x, wq, sq).view(b, s, qh, hd).transpose(1, 2)
            k = _fp8_mm(x, wk, sk).view(b, s, kvh, hd).transpose(1, 2)
            v = _fp8_mm(x, wv, sv).view(b, s, kvh, hd).transpose(1, 2)
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            at = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            h = x + _fp8_mm(at.transpose(1, 2).contiguous().view(b, s, qh * hd), wo, so)
            if act_id == 0 and wg is not None and sg is not None:
                ff = _fp8_mm(F.silu(_fp8_mm(h, wg, sg)) * _fp8_mm(h, wup, sup), wdn, sdn)
            elif act_id == 1:
                u = F.relu(_fp8_mm(h, wup, sup))
                ff = _fp8_mm(u * u, wdn, sdn)
            else:
                ff = _fp8_mm(F.gelu(_fp8_mm(h, wup, sup)), wdn, sdn)
            return h + ff

        return {
            "bf16_eager": _bf16_eager,
            "bf16_comp": torch.compile(_bf16_eager, dynamic=False),
            "fp8_eager": _fp8_eager,
            "fp8_comp": torch.compile(_fp8_eager, dynamic=False),
        }

    def _make_connector_kernels() -> dict[str, Any]:
        @torch.compile(dynamic=False)
        def proj_cat(v: torch.Tensor, w_vp: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
            return torch.cat([F.linear(v, w_vp), lang], dim=1)

        @torch.compile(dynamic=False)
        def dual_vis(v: torch.Tensor, w_vp: torch.Tensor, dv: torch.Tensor, w_dp: torch.Tensor) -> torch.Tensor:
            return F.linear(v, w_vp) + F.linear(dv, w_dp)

        @torch.compile(dynamic=False)
        def oft_pre(v_emb: torch.Tensor, lang: torch.Tensor, nact: torch.Tensor, w_aq: torch.Tensor) -> torch.Tensor:
            return torch.cat([v_emb, lang, F.linear(nact, w_aq)], dim=1)

        @torch.compile(dynamic=False)
        def oft_head(seq: torch.Tensor, chunk: int, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
            lat = seq[:, -chunk:, :]
            return F.linear(F.silu(F.linear(lat, w1)), w2)

        @torch.compile(dynamic=False)
        def flow_init_proj(
            h: torch.Tensor,
            w_vl: torch.Tensor,
            prop: torch.Tensor,
            w_st: torch.Tensor,
            nact: torch.Tensor,
            w_ac: torch.Tensor,
        ) -> torch.Tensor:
            cond = F.linear(h, w_vl).mean(dim=1, keepdim=True)
            return torch.cat([F.linear(prop, w_st) + cond, F.linear(nact, w_ac) + cond], dim=1)

        @torch.compile(dynamic=False)
        def flow_init_direct(
            h: torch.Tensor,
            prop: torch.Tensor,
            w_st: torch.Tensor,
            nact: torch.Tensor,
            w_ac: torch.Tensor,
        ) -> torch.Tensor:
            cond = h.mean(dim=1, keepdim=True)
            return torch.cat([F.linear(prop, w_st) + cond, F.linear(nact, w_ac) + cond], dim=1)

        @torch.compile(dynamic=False)
        def flow_out(z: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
            return F.linear(z[:, 1:, :], w_out)

        return {
            "proj_cat": proj_cat,
            "dual_vis": dual_vis,
            "oft_pre": oft_pre,
            "oft_head": oft_head,
            "flow_init_proj": flow_init_proj,
            "flow_init_direct": flow_init_direct,
            "flow_out": flow_out,
        }

    def _make_ar_kernels(qh: int, kvh: int, hd: int) -> dict[str, Any]:
        @torch.compile(dynamic=False)
        def kv_proj_bf16(h: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            b, s, _ = h.shape
            return (
                F.linear(h, wk).view(b, s, kvh, hd).transpose(1, 2),
                F.linear(h, wv).view(b, s, kvh, hd).transpose(1, 2),
            )

        @torch.compile(dynamic=False)
        def kv_proj_fp8(
            h: torch.Tensor, wk: torch.Tensor, sk: torch.Tensor, wv: torch.Tensor, sv: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            b, s, _ = h.shape
            return (
                _fp8_mm(h, wk, sk).view(b, s, kvh, hd).transpose(1, 2),
                _fp8_mm(h, wv, sv).view(b, s, kvh, hd).transpose(1, 2),
            )

        def ar_bf16_eager(
            xt: torch.Tensor,
            kc: torch.Tensor,
            vc: torch.Tensor,
            wq: torch.Tensor,
            wk: torch.Tensor,
            wv: torch.Tensor,
            wo: torch.Tensor,
            wg: torch.Tensor,
            wup: torch.Tensor,
            wdn: torch.Tensor,
        ) -> torch.Tensor:
            b, s, _ = xt.shape
            q = F.linear(xt, wq).view(b, s, qh, hd).transpose(1, 2)
            kt = F.linear(xt, wk).view(b, s, kvh, hd).transpose(1, 2)
            vt = F.linear(xt, wv).view(b, s, kvh, hd).transpose(1, 2)
            at = F.scaled_dot_product_attention(q, torch.cat([kc, kt], dim=2), torch.cat([vc, vt], dim=2), is_causal=False)
            h = xt + F.linear(at.transpose(1, 2).contiguous().view(b, s, qh * hd), wo)
            return h + F.linear(F.silu(F.linear(h, wg)) * F.linear(h, wup), wdn)

        def ar_fp8_eager(
            xt: torch.Tensor,
            kc: torch.Tensor,
            vc: torch.Tensor,
            wq: torch.Tensor,
            sq: torch.Tensor,
            wk: torch.Tensor,
            sk: torch.Tensor,
            wv: torch.Tensor,
            sv: torch.Tensor,
            wo: torch.Tensor,
            so: torch.Tensor,
            wg: torch.Tensor,
            sg: torch.Tensor,
            wup: torch.Tensor,
            sup: torch.Tensor,
            wdn: torch.Tensor,
            sdn: torch.Tensor,
        ) -> torch.Tensor:
            b, s, _ = xt.shape
            q = _fp8_mm(xt, wq, sq).view(b, s, qh, hd).transpose(1, 2)
            kt = _fp8_mm(xt, wk, sk).view(b, s, kvh, hd).transpose(1, 2)
            vt = _fp8_mm(xt, wv, sv).view(b, s, kvh, hd).transpose(1, 2)
            at = F.scaled_dot_product_attention(q, torch.cat([kc, kt], dim=2), torch.cat([vc, vt], dim=2), is_causal=False)
            h = xt + _fp8_mm(at.transpose(1, 2).contiguous().view(b, s, qh * hd), wo, so)
            return h + _fp8_mm(F.silu(_fp8_mm(h, wg, sg)) * _fp8_mm(h, wup, sup), wdn, sdn)

        @torch.compile(dynamic=False)
        def sample_embed(h_last: torch.Tensor, lm_head: torch.Tensor, tok_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            tid = torch.argmax(F.linear(h_last, lm_head), dim=-1)
            return tid.to(torch.bfloat16), F.embedding(tid, tok_embed)

        return {
            "kv_bf16": kv_proj_bf16,
            "kv_fp8": kv_proj_fp8,
            "ar_bf16_eager": ar_bf16_eager,
            "ar_bf16_comp": torch.compile(ar_bf16_eager, dynamic=False),
            "ar_fp8_eager": ar_fp8_eager,
            "ar_fp8_comp": torch.compile(ar_fp8_eager, dynamic=False),
            "sample_embed": sample_embed,
        }

    def _load_weight_pool_from_shm(shm_dir: Path, max_tensors: int = 800) -> list[torch.Tensor]:
        pool: list[torch.Tensor] = []
        for sp in sorted(shm_dir.rglob("*.safetensors")):
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    sl = sf.get_slice(k)
                    if len(sl.get_shape()) == 2:
                        pool.append(sf.get_tensor(k).to(torch.bfloat16).clone())
                        if len(pool) >= max_tensors:
                            return pool
        return pool

    def _build_tpu_bank(
        spec: dict[str, Any],
        precision: str,
        weight_pool: list[torch.Tensor],
        tp_degree: int = 1,
    ) -> tuple[dict[str, Any], float]:
        tp = max(1, tp_degree)
        total_bytes = 0

        def _w(out_f: int, in_f: int, allow_fp8: bool = True) -> Any:
            nonlocal total_bytes
            matched: torch.Tensor | None = None
            for idx, t in enumerate(weight_pool):
                if tuple(t.shape) == (out_f, in_f):
                    matched = weight_pool.pop(idx)
                    break
            if matched is None:
                matched = torch.empty((out_f, in_f), dtype=torch.bfloat16, device="cpu").normal_(0.0, 0.02)
            else:
                matched = matched.contiguous()

            if precision == "FP8" and allow_fp8 and (out_f % 16 == 0) and (in_f % 16 == 0):
                w_max = matched.abs().amax().clamp(min=1e-6)
                sc = (w_max / 448.0).to(torch.float32).to(dev)
                qf = (matched / (w_max / 448.0).to(torch.bfloat16)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(dev)
                total_bytes += qf.numel() * 1 + 4
                return (qf, sc)
            t_d = matched.to(dev)
            total_bytes += t_d.numel() * 2
            return t_d

        def _make_stage(n_layers: int, d: int, qh: int, kvh: int, hd: int, mlp: int, act_id: int, is_causal: bool) -> dict[str, Any]:
            qh_s = max(1, qh // tp)
            kvh_s = max(1, kvh // tp)
            mlp_s = ((mlp // tp) // 16) * 16
            layers = []
            for _ in range(n_layers):
                lw: dict[str, Any] = {
                    "wq": _w(qh_s * hd, d),
                    "wk": _w(kvh_s * hd, d),
                    "wv": _w(kvh_s * hd, d),
                    "wo": _w(d, qh_s * hd),
                    "wup": _w(mlp_s, d),
                    "wdn": _w(d, mlp_s),
                    "wg": _w(mlp_s, d) if act_id == 0 else None,
                }
                layers.append(lw)
            return {
                "layers": layers,
                "kernels": _make_stage_kernels(qh_s, kvh_s, hd, act_id, is_causal),
                "qh": qh_s,
                "kvh": kvh_s,
                "hd": hd,
            }

        fam = spec["family"]
        vis_hd = spec["vis_dim"] // spec["vis_heads"]
        vlm_causal = fam != "openvla_oft"
        bank: dict[str, Any] = {
            "family": fam,
            "precision": precision,
            "tp_degree": tp,
            "conn": _make_connector_kernels(),
            "vis": _make_stage(spec["vis_layers"], spec["vis_dim"], spec["vis_heads"], spec["vis_heads"], vis_hd, spec["vis_mlp"], 2, False),
            "vis_proj": _w(spec["vlm_dim"], spec["vis_dim"], allow_fp8=False),
            "vlm": _make_stage(
                spec["vlm_layers"],
                spec["vlm_dim"],
                spec["vlm_q_heads"],
                spec["vlm_kv_heads"],
                spec["vlm_hd"],
                spec["vlm_mlp"],
                1 if fam == "cosmos3_edge" else 0,
                vlm_causal,
            ),
        }
        ad = spec["act_dim"]
        vd = spec["vlm_dim"]
        if fam == "pi05":
            ed = spec["exp_dim"]
            bank["vl_to_exp"] = _w(ed, vd, allow_fp8=False)
            bank["state_in"] = _w(ed, ad, allow_fp8=False)
            bank["act_in"] = _w(ed, ad, allow_fp8=False)
            bank["exp"] = _make_stage(spec["exp_layers"], ed, spec["exp_q_heads"], spec["exp_kv_heads"], spec["exp_hd"], spec["exp_mlp"], 0, False)
            bank["act_out"] = _w(ad, ed, allow_fp8=False)
        elif fam == "gr00t":
            dd = spec["dit_dim"]
            bank["vl_to_dit"] = _w(dd, vd, allow_fp8=False)
            bank["state_in"] = _w(dd, ad, allow_fp8=False)
            bank["act_in"] = _w(dd, ad, allow_fp8=False)
            bank["dit"] = _make_stage(spec["dit_layers"], dd, spec["dit_heads"], spec["dit_heads"], spec["dit_hd"], spec["dit_mlp"], 2, False)
            bank["act_out"] = _w(ad, dd, allow_fp8=False)
        elif fam == "cosmos3_edge":
            bank["state_in"] = _w(vd, ad, allow_fp8=False)
            bank["act_in"] = _w(vd, ad, allow_fp8=False)
            bank["mot"] = _make_stage(spec["vlm_layers"], vd, spec["vlm_q_heads"], spec["vlm_kv_heads"], spec["vlm_hd"], spec["vlm_mlp"], 1, False)
            bank["act_out"] = _w(ad, vd, allow_fp8=False)
        elif fam == "openvla_ar":
            bank["dino"] = _make_stage(24, 1024, 16, 16, 64, 4096, 2, False)
            bank["dino_proj"] = _w(vd, 1024, allow_fp8=False)
            bank["lm_head"] = _w(32064, vd, allow_fp8=False)
            bank["tok_embed"] = _w(32064, vd, allow_fp8=False)
            vlm_s = bank["vlm"]
            bank["ar_kernels"] = _make_ar_kernels(vlm_s["qh"], vlm_s["kvh"], vlm_s["hd"])
        elif fam == "openvla_oft":
            bank["dino"] = _make_stage(24, 1024, 16, 16, 64, 4096, 2, False)
            bank["dino_proj"] = _w(vd, 1024, allow_fp8=False)
            bank["act_q_in"] = _w(vd, ad, allow_fp8=False)
            bank["oft_mlp1"] = _w(vd, vd, allow_fp8=False)
            bank["oft_mlp2"] = _w(ad, vd, allow_fp8=False)

        torch.accelerator.synchronize()
        return bank, round(total_bytes / (1024.0**3), 2)

    def _run_stage(x: torch.Tensor, stg: dict[str, Any], precision: str, compiled: bool) -> torch.Tensor:
        ks = stg["kernels"]
        if precision == "BF16":
            fn = ks["bf16_comp"] if compiled else ks["bf16_eager"]
            for lw in stg["layers"]:
                x = fn(x, lw["wq"], lw["wk"], lw["wv"], lw["wo"], lw["wup"], lw["wdn"], lw["wg"])
            return x
        fn8 = ks["fp8_comp"] if compiled else ks["fp8_eager"]
        for lw in stg["layers"]:
            wq, sq = lw["wq"]
            wk, sk = lw["wk"]
            wv, sv = lw["wv"]
            wo, so = lw["wo"]
            wup, sup = lw["wup"]
            wdn, sdn = lw["wdn"]
            wg, sg = lw["wg"] if lw["wg"] is not None else (None, None)
            x = fn8(x, wq, sq, wk, sk, wv, sv, wo, so, wup, sup, wdn, sdn, wg, sg)
        return x

    def _forward_bank(
        bank: dict[str, Any],
        inp: tuple[torch.Tensor, ...],
        n_steps: int,
        compiled: bool,
        tp_sync: bool = False,
    ) -> torch.Tensor:
        vis, dino, lang, prop, nact = inp[:5]
        prec = bank["precision"]
        fam = bank["family"]
        conn = bank["conn"]

        v = _run_stage(vis, bank["vis"], prec, compiled)
        if tp_sync:
            dist.all_reduce(v, group=my_tp2_grp)

        if fam == "openvla_oft":
            dv = _run_stage(dino, bank["dino"], prec, compiled)
            if tp_sync:
                dist.all_reduce(dv, group=my_tp2_grp)
            v_emb = conn["dual_vis"](v, bank["vis_proj"], dv, bank["dino_proj"])
            seq = conn["oft_pre"](v_emb, lang, nact, bank["act_q_in"])
            seq = _run_stage(seq, bank["vlm"], prec, compiled)
            if tp_sync:
                dist.all_reduce(seq, group=my_tp2_grp)
            return conn["oft_head"](seq, nact.shape[1], bank["oft_mlp1"], bank["oft_mlp2"])

        if fam == "openvla_ar":
            dv = _run_stage(dino, bank["dino"], prec, compiled)
            if tp_sync:
                dist.all_reduce(dv, group=my_tp2_grp)
            v_emb = conn["dual_vis"](v, bank["vis_proj"], dv, bank["dino_proj"])
            vl_seq = torch.cat([v_emb, lang], dim=1)
            vlm = bank["vlm"]
            ark = bank["ar_kernels"]
            h = vl_seq
            k_caches: list[torch.Tensor] = []
            v_caches: list[torch.Tensor] = []
            if prec == "BF16":
                fn_pre = vlm["kernels"]["bf16_comp"] if compiled else vlm["kernels"]["bf16_eager"]
                fn_ar = ark["ar_bf16_comp"] if compiled else ark["ar_bf16_eager"]
                for lw in vlm["layers"]:
                    kc, vc = ark["kv_bf16"](h, lw["wk"], lw["wv"])
                    k_caches.append(kc)
                    v_caches.append(vc)
                    h = fn_pre(h, lw["wq"], lw["wk"], lw["wv"], lw["wo"], lw["wup"], lw["wdn"], lw["wg"])
                if tp_sync:
                    dist.all_reduce(h, group=my_tp2_grp)
                t_b16, cur = ark["sample_embed"](h[:, -1:, :], bank["lm_head"], bank["tok_embed"])
                preds = [t_b16]
                for _ in range(max(0, n_steps - 1)):
                    xh = cur
                    for idx, lw in enumerate(vlm["layers"]):
                        xh = fn_ar(xh, k_caches[idx], v_caches[idx], lw["wq"], lw["wk"], lw["wv"], lw["wo"], lw["wg"], lw["wup"], lw["wdn"])
                    if tp_sync:
                        dist.all_reduce(xh, group=my_tp2_grp)
                    tb, cur = ark["sample_embed"](xh, bank["lm_head"], bank["tok_embed"])
                    preds.append(tb)
                return torch.cat(preds, dim=1).unsqueeze(1)
            else:
                fn_pre8 = vlm["kernels"]["fp8_comp"] if compiled else vlm["kernels"]["fp8_eager"]
                fn_ar8 = ark["ar_fp8_comp"] if compiled else ark["ar_fp8_eager"]
                for lw in vlm["layers"]:
                    wq, sq = lw["wq"]
                    wk, sk = lw["wk"]
                    wv, sv = lw["wv"]
                    wo, so = lw["wo"]
                    wup, sup = lw["wup"]
                    wdn, sdn = lw["wdn"]
                    wg, sg = lw["wg"]
                    kc, vc = ark["kv_fp8"](h, wk, sk, wv, sv)
                    k_caches.append(kc)
                    v_caches.append(vc)
                    h = fn_pre8(h, wq, sq, wk, sk, wv, sv, wo, so, wup, sup, wdn, sdn, wg, sg)
                t_b16, cur = ark["sample_embed"](h[:, -1:, :], bank["lm_head"], bank["tok_embed"])
                preds = [t_b16]
                for _ in range(max(0, n_steps - 1)):
                    xh = cur
                    for idx, lw in enumerate(vlm["layers"]):
                        wq, sq = lw["wq"]
                        wk, sk = lw["wk"]
                        wv, sv = lw["wv"]
                        wo, so = lw["wo"]
                        wup, sup = lw["wup"]
                        wdn, sdn = lw["wdn"]
                        wg, sg = lw["wg"]
                        xh = fn_ar8(xh, k_caches[idx], v_caches[idx], wq, sq, wk, sk, wv, sv, wo, so, wg, sg, wup, sup, wdn, sdn)
                    tb, cur = ark["sample_embed"](xh, bank["lm_head"], bank["tok_embed"])
                    preds.append(tb)
                return torch.cat(preds, dim=1).unsqueeze(1)

        vl_seq = conn["proj_cat"](v, bank["vis_proj"], lang)
        h = _run_stage(vl_seq, bank["vlm"], prec, compiled)
        if tp_sync:
            dist.all_reduce(h, group=my_tp2_grp)
        dt_step = 1.0 / max(n_steps, 1)

        if fam == "pi05":
            z = conn["flow_init_proj"](h, bank["vl_to_exp"], prop, bank["state_in"], nact, bank["act_in"])
            for _ in range(n_steps):
                u = _run_stage(z, bank["exp"], prec, compiled)
                if tp_sync:
                    dist.all_reduce(u, group=my_tp2_grp)
                z = z + u * dt_step
            return conn["flow_out"](z, bank["act_out"])

        if fam == "gr00t":
            z = conn["flow_init_proj"](h, bank["vl_to_dit"], prop, bank["state_in"], nact, bank["act_in"])
            for _ in range(n_steps):
                u = _run_stage(z, bank["dit"], prec, compiled)
                if tp_sync:
                    dist.all_reduce(u, group=my_tp2_grp)
                z = z + u * dt_step
            return conn["flow_out"](z, bank["act_out"])

        z = conn["flow_init_direct"](h, prop, bank["state_in"], nact, bank["act_in"])
        for _ in range(n_steps):
            u = _run_stage(z, bank["mot"], prec, compiled)
            if tp_sync:
                dist.all_reduce(u, group=my_tp2_grp)
            z = z + u * dt_step
        return conn["flow_out"](z, bank["act_out"])

    def _make_inputs(
        spec: dict[str, Any],
        bsz: int,
        n_cams: int,
        res: int,
        act_chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        patch = spec["patch_size"]
        toks_per_cam = (res // patch) * (res // patch)
        vis_seq = n_cams * toks_per_cam
        vis = (torch.randn((bsz, vis_seq, spec["vis_dim"]), dtype=torch.bfloat16, device="cpu") * 0.02).to(dev)
        dino = None
        if spec["family"] in ("openvla_ar", "openvla_oft"):
            dino = (torch.randn((bsz, vis_seq, 1024), dtype=torch.bfloat16, device="cpu") * 0.02).to(dev)
        lang = (torch.randn((bsz, spec["lang_tokens"], spec["vlm_dim"]), dtype=torch.bfloat16, device="cpu") * 0.02).to(dev)
        prop = (torch.randn((bsz, 1, spec["act_dim"]), dtype=torch.bfloat16, device="cpu") * 0.02).to(dev)
        eff_chunk = 1 if spec["family"] == "openvla_ar" else act_chunk
        nact = (torch.randn((bsz, eff_chunk, spec["act_dim"]), dtype=torch.bfloat16, device="cpu") * 0.02).to(dev)
        torch.accelerator.synchronize()
        return vis, dino, lang, prop, nact, vis_seq + spec["lang_tokens"]

    def _stage_model_collective(mkey: str) -> tuple[list[torch.Tensor], dict[str, Any]]:
        spec = MODEL_SPECS[mkey]
        mdir = models_root / spec["repo_id"]
        shm_dir = Path("/dev/shm") / f"_vla_stage_{mkey}"
        obj_list: list[Any] = [None]
        if rank == 0:
            shm_dir, sf_info = stage_and_inspect_safetensors(mdir, mkey)
            obj_list[0] = sf_info
        dist.broadcast_object_list(obj_list, src=0, group=gloo_pg)
        dist.barrier(group=gloo_pg)
        sf_info = obj_list[0]
        wpool = _load_weight_pool_from_shm(shm_dir, max_tensors=800)
        dist.barrier(group=gloo_pg)
        if rank == 0:
            shutil.rmtree(shm_dir, ignore_errors=True)
        return wpool, sf_info

    # ------------------------------------------------------------------
    # Persistent Multi-Process HTTP Keep-Alive Mode (`--serve-worker-port`)
    # Each Rank `r = 0..7` serves its dedicated TensorCore on `serve_port + r`!
    # ------------------------------------------------------------------
    if serve_port > 0:
        my_port = serve_port + rank
        loaded_rpc: dict[str, dict[str, Any]] = {}
        for mkey in model_keys:
            spec = MODEL_SPECS[mkey]
            wpool, sf_info = _stage_model_collective(mkey)
            bank, mod_gib = _build_tpu_bank(spec, "BF16", wpool, tp_degree=1)
            del wpool
            gc.collect()
            hbm_gib = max(mod_gib, sf_info["bf16_equivalent_gib"])
            steps = spec["fast_steps"]
            chunk = 1 if spec["family"] == "openvla_ar" else spec["default_chunk"]
            res = spec["default_res"]
            in_cache: dict[int, Any] = {}
            with torch.no_grad():
                for bsz in (1, 4, 8, 16):
                    inp = _make_inputs(spec, bsz, 2, res, chunk)
                    in_cache[bsz] = inp
                    _ = _forward_bank(bank, inp, steps, compiled=True, tp_sync=False)
                    torch.accelerator.synchronize()
            dist.barrier(group=gloo_pg)
            loaded_rpc[mkey] = {
                "bank": bank,
                "in_cache": in_cache,
                "steps": steps,
                "chunk": chunk,
                "hbm_gib": round(hbm_gib, 2),
            }
            if rank == 0:
                log("rpc-server", f"Ready model={mkey} across all {world_size} TensorCore OS processes (ports {serve_port}..{serve_port+world_size-1})")

        tc_lock = threading.Lock()

        class _RankRpcHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path == "/health":
                    body = json.dumps({"status": "ok", "rank": rank, "models": list(loaded_rpc.keys())}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self) -> None:
                clen = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(clen) if clen > 0 else b"{}"
                req = json.loads(raw.decode("utf-8"))
                mkey = req.get("model_key", model_keys[0])
                bsz = int(req.get("batch_size", 1))
                entry = loaded_rpc[mkey]
                t_q0 = time.perf_counter()
                with tc_lock, torch.no_grad():
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    t0 = time.perf_counter()
                    _ = _forward_bank(entry["bank"], entry["in_cache"][bsz], entry["steps"], compiled=True, tp_sync=False)
                    torch.accelerator.synchronize()
                    tpu_ms = (time.perf_counter() - t0) * 1000.0
                resp = json.dumps(
                    {
                        "ok": True,
                        "rank": rank,
                        "queue_ms": round(q_ms, 3),
                        "tpu_ms": round(tpu_ms, 3),
                        "action_chunk": entry["chunk"],
                        "hbm_gib": entry["hbm_gib"],
                    }
                ).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                except BrokenPipeError:
                    pass

        srv = FastThreadingHTTPServer(("0.0.0.0", my_port), _RankRpcHandler)
        if rank == 0:
            log("rpc-server", f"All {world_size} TensorCore HTTP Keep-Alive servers listening on :{serve_port}..:{serve_port+world_size-1}")
        srv.serve_forever()
        return

    # ------------------------------------------------------------------
    # Standard Single-Host Deep Sweep Mode (`8` synchronized TensorCore processes)
    # ------------------------------------------------------------------
    all_rows: list[dict[str, Any]] = []
    if rank == 0:
        out_jsonl.unlink(missing_ok=True)

    def _append_row(row: dict[str, Any]) -> None:
        if rank != 0:
            return
        all_rows.append(row)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")
        out_summary_json.write_text(
            json.dumps(
                {
                    "updated_at_utc": utc_iso(),
                    "chip_type": "v7x",
                    "cores_per_host": world_size,
                    "execution_mode": "8-process-torchrun-tpu-dist-ici",
                    "vision_preproc_ablation": preproc_map,
                    "num_rows": len(all_rows),
                    "results": all_rows,
                },
                indent=2,
            )
        )

    for model_idx, mkey in enumerate(model_keys):
        torch._dynamo.reset()
        gc.collect()
        spec = MODEL_SPECS[mkey]
        if rank == 0:
            log(mkey, f"=== Stage & Inspect {spec['display_name']} ===")
        wpool_raw, sf_info = _stage_model_collective(mkey)
        if rank == 0:
            log(mkey, f"Safetensors metadata: {sf_info}")

        t_b0 = time.perf_counter()
        bf16_bank, bf16_mod_gib = _build_tpu_bank(spec, "BF16", list(wpool_raw), tp_degree=1)
        bf16_peak_gib = max(bf16_mod_gib, sf_info["bf16_equivalent_gib"])
        dist.barrier(group=gloo_pg)
        if rank == 0:
            log(mkey, f"Loaded 8x hardware-isolated BF16 banks across 8 TensorCores in {time.perf_counter()-t_b0:.2f}s ({bf16_peak_gib:.2f} GiB HBM/TC)")

        def _preproc_stats(n_cams: int, res: int) -> tuple[float, float]:
            vp = preproc_map.get(f"{n_cams}x{res}") or preproc_map.get(f"{n_cams}x{spec['default_res']}") or {}
            loc = float(vp.get("local_cpu_preproc_ms", 5.2))
            dis = float(vp.get("disagg_cpu_rtt_ms", loc))
            return loc, dis

        def _run_dp1_and_dp8_point(
            suite_tag: str,
            precision: str,
            compiled: bool,
            bank: dict[str, Any],
            n_cams: int,
            res: int,
            act_chunk: int,
            steps: int,
            bsz: int,
            peak_gib: float,
            measure_dp8: bool = True,
            n_iters: int = 15,
        ) -> None:
            backend = "torchtpu-compiled" if compiled else "torchtpu-eager"
            eff_chunk = 1 if spec["family"] == "openvla_ar" else act_chunk
            loc_pre, dis_rtt = _preproc_stats(n_cams, res)

            with torch.no_grad():
                inp = _make_inputs(spec, bsz, n_cams, res, eff_chunk)
                vl_toks = inp[5]
                # All 8 ranks warm up together so Tier-2 XLA cache populates cleanly!
                for _ in range(2):
                    _ = _forward_bank(bank, inp, steps, compiled=compiled, tp_sync=False)
                    torch.accelerator.synchronize()
                dist.barrier(group=gloo_pg)

                lats: list[float] = []
                if rank == 0:
                    for _ in range(n_iters):
                        t0 = time.perf_counter()
                        _ = _forward_bank(bank, inp, steps, compiled=compiled, tp_sync=False)
                        torch.accelerator.synchronize()
                        lats.append((time.perf_counter() - t0) * 1000.0)
                dist.barrier(group=gloo_pg)

                w_lats: list[float] = []
                if measure_dp8:
                    for _ in range(n_iters):
                        dist.barrier(group=gloo_pg)
                        tw0 = time.perf_counter()
                        _ = _forward_bank(bank, inp, steps, compiled=compiled, tp_sync=False)
                        torch.accelerator.synchronize()
                        dist.barrier(group=gloo_pg)
                        if rank == 0:
                            w_lats.append((time.perf_counter() - tw0) * 1000.0)

                del inp
                torch.accelerator.synchronize()

            if rank != 0:
                return

            p50 = pct(lats, 50)
            p95 = pct(lats, 95)
            p99 = pct(lats, 99)
            dp1_traj_s = (bsz * 1000.0) / max(p50, 1e-6)
            dp1_act_s = dp1_traj_s * eff_chunk
            e2e_loc = p50 + loc_pre
            e2e_dis_sync = p50 + dis_rtt
            e2e_dis_pipe = max(p50, dis_rtt)

            dp8_wave_p50 = round(pct(w_lats, 50), 2) if measure_dp8 else None
            dp8_traj_s = round((bsz * world_size * 1000.0) / max(dp8_wave_p50 or 1.0, 1e-6), 2) if measure_dp8 else None
            dp8_act_s = round((dp8_traj_s or 0.0) * eff_chunk, 1) if measure_dp8 else None
            dp8_eff_pct = round(100.0 * (dp8_traj_s or 0.0) / max(dp1_traj_s * world_size, 1e-6), 1) if measure_dp8 else None

            row = {
                "suite": suite_tag,
                "model_key": mkey,
                "repo_id": spec["repo_id"],
                "paradigm": spec["paradigm"],
                "topology": "1-Host DP=8 (8 TC)" if measure_dp8 else "1-TC (DP=1)",
                "precision": precision,
                "backend": backend,
                "num_cameras": n_cams,
                "resolution": f"{res}x{res}",
                "vlm_tokens": vl_toks,
                "action_chunk": eff_chunk,
                "solver_or_decode_steps": steps,
                "batch_size": bsz,
                "tpu_p50_ms": round(p50, 2),
                "tpu_p95_ms": round(p95, 2),
                "tpu_p99_ms": round(p99, 2),
                "local_cpu_preproc_ms": round(loc_pre, 2),
                "disagg_cpu_rtt_ms": round(dis_rtt, 2),
                "e2e_local_cpu_p50_ms": round(e2e_loc, 2),
                "e2e_disagg_cpu_p50_ms": round(e2e_dis_sync, 2),
                "e2e_disagg_pipelined_p50_ms": round(e2e_dis_pipe, 2),
                "sla_100hz_10ms_tpu": p50 <= 10.0,
                "sla_100hz_10ms_pipelined": e2e_dis_pipe <= 10.0,
                "sla_50hz_20ms_local": e2e_loc <= 20.0,
                "sla_50hz_20ms_disagg_sync": e2e_dis_sync <= 20.0,
                "sla_50hz_20ms_pipelined": e2e_dis_pipe <= 20.0,
                "sla_20hz_50ms_disagg_sync": e2e_dis_sync <= 50.0,
                "sla_20hz_50ms_pipelined": e2e_dis_pipe <= 50.0,
                "dp1_trajectories_per_s": round(dp1_traj_s, 2),
                "dp1_actions_per_s": round(dp1_act_s, 1),
                "host_dp8_wave_p50_ms": dp8_wave_p50,
                "host_dp8_trajectories_per_s": dp8_traj_s,
                "host_dp8_actions_per_s": dp8_act_s,
                "dp8_scaling_efficiency_pct": dp8_eff_pct,
                "peak_hbm_gib_per_tc": round(peak_gib, 2),
                "safetensors_tensors": sf_info["num_safetensors_tensors"],
                "status": "PASS",
            }
            _append_row(row)
            log(
                mkey,
                f"[{suite_tag}][{precision}][{backend}] cams={n_cams} {res}x{res} chunk={eff_chunk} steps={steps} B={bsz} "
                f"-> TPU_P50={row['tpu_p50_ms']}ms (P99={row['tpu_p99_ms']}ms) | PipeE2E={row['e2e_disagg_pipelined_p50_ms']}ms "
                f"| DP1={row['dp1_actions_per_s']} act/s | DP8={row['host_dp8_actions_per_s']} act/s ({dp8_eff_pct}%)",
            )

        def_res = spec["default_res"]
        def_chunk = spec["default_chunk"]
        fast_st = spec["fast_steps"]

        # 1. Batch Size Sweep (B = 1, 2, 4, 8, 16, 32) on BF16 Compiled + Eager anchor
        for bsz in (1, 2, 4, 8, 16, 32):
            _run_dp1_and_dp8_point(
                "batch_sweep_bf16_compiled", "BF16", True, bf16_bank, 2, def_res, def_chunk, fast_st, bsz, bf16_peak_gib, measure_dp8=True
            )
        for bsz in (1, 8):
            _run_dp1_and_dp8_point(
                "eager_vs_compiled_ablation", "BF16", False, bf16_bank, 2, def_res, def_chunk, fast_st, bsz, bf16_peak_gib, measure_dp8=False, n_iters=6
            )

        # 2. ODE Solver Steps (4-step vs 10-step) & Action Horizon Sweep (chunk = 16, 32, 50)
        if spec["family"] in ("pi05", "gr00t", "cosmos3_edge"):
            for bsz in (1, 8):
                _run_dp1_and_dp8_point(
                    "ode_solver_10step_vs_4step", "BF16", True, bf16_bank, 2, def_res, def_chunk, 10, bsz, bf16_peak_gib, measure_dp8=True
                )
        if spec["family"] != "openvla_ar":
            for ch in (16, 32, 50):
                if ch == def_chunk:
                    continue
                for bsz in (1, 8):
                    _run_dp1_and_dp8_point(
                        "action_horizon_sweep", "BF16", True, bf16_bank, 2, def_res, ch, fast_st, bsz, bf16_peak_gib, measure_dp8=(bsz == 8)
                    )

        # 3. Camera Views (1, 2, 4) x Resolutions (224, 256, 384)
        for n_cams in (1, 2, 4):
            for res in (224, 256, 384):
                if n_cams == 2 and res == def_res:
                    continue
                _run_dp1_and_dp8_point(
                    "multicam_resolution_sweep",
                    "BF16",
                    True,
                    bf16_bank,
                    n_cams,
                    res,
                    def_chunk,
                    fast_st,
                    1,
                    bf16_peak_gib,
                    measure_dp8=(res == def_res),
                    n_iters=12,
                )

        # 4. 1-Host Closed-Loop Concurrency (`CON = 1, 4, 8, 16, 32, 64`) & Open-Loop Poisson (`50..1000 Hz`)
        #    across all 8 independent TensorCore OS processes via HTTP Keep-Alive (`127.0.0.1:19200..19207`)!
        eff_c = 1 if spec["family"] == "openvla_ar" else def_chunk
        with torch.no_grad():
            ip1 = _make_inputs(spec, 1, 2, def_res, eff_c)
            _ = _forward_bank(bf16_bank, ip1, fast_st, compiled=True, tp_sync=False)
            torch.accelerator.synchronize()

        local_lock = threading.Lock()
        base_ep_port = 19200 + model_idx * 20
        ep_port = base_ep_port + rank

        class _LocalEpHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                clen = int(self.headers.get("Content-Length", "0") or 0)
                if clen > 0:
                    _ = self.rfile.read(clen)
                with local_lock, torch.no_grad():
                    ts = time.perf_counter()
                    _ = _forward_bank(bf16_bank, ip1, fast_st, compiled=True, tp_sync=False)
                    torch.accelerator.synchronize()
                    svc_ms = (time.perf_counter() - ts) * 1000.0
                resp = json.dumps({"ok": True, "tpu_ms": round(svc_ms, 3)}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                except BrokenPipeError:
                    pass

        ep_srv = FastThreadingHTTPServer(("127.0.0.1", ep_port), _LocalEpHandler)
        ep_thr = threading.Thread(target=ep_srv.serve_forever, daemon=True)
        ep_thr.start()
        dist.barrier(group=gloo_pg)

        if rank == 0:
            pool_ka = MultiEndpointKeepAlivePool([f"http://127.0.0.1:{base_ep_port + r}" for r in range(world_size)])
            pool_ka.wait_healthy(timeout_s=30.0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as wp:
                list(wp.map(lambda _: pool_ka.call_infer(mkey, 1), range(16)))

            for conc in (1, 4, 8, 16, 32, 64):
                n_reqs = max(24, conc * 3)
                t_w0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as cpool:
                    outs = list(cpool.map(lambda _: pool_ka.call_infer(mkey, 1), range(n_reqs)))
                wall_s = time.perf_counter() - t_w0
                tot_lats = [x["e2e_rpc_ms"] for x in outs]
                svc_lats = [x["tpu_ms"] for x in outs]
                traj_s = n_reqs / max(wall_s, 1e-6)
                _append_row(
                    {
                        "suite": "closed_loop_concurrency_dp8",
                        "model_key": mkey,
                        "repo_id": spec["repo_id"],
                        "paradigm": spec["paradigm"],
                        "topology": "1-Host DP=8 (8 TC)",
                        "precision": "BF16",
                        "backend": "torchtpu-compiled",
                        "concurrency": conc,
                        "batch_size": 1,
                        "num_cameras": 2,
                        "resolution": f"{def_res}x{def_res}",
                        "action_chunk": eff_c,
                        "solver_or_decode_steps": fast_st,
                        "tpu_service_p50_ms": round(pct(svc_lats, 50), 2),
                        "e2e_queue_plus_tpu_p50_ms": round(pct(tot_lats, 50), 2),
                        "e2e_queue_plus_tpu_p95_ms": round(pct(tot_lats, 95), 2),
                        "e2e_queue_plus_tpu_p99_ms": round(pct(tot_lats, 99), 2),
                        "sla_100hz_10ms": pct(tot_lats, 50) <= 10.0,
                        "sla_50hz_20ms": pct(tot_lats, 50) <= 20.0,
                        "sla_20hz_50ms": pct(tot_lats, 50) <= 50.0,
                        "trajectories_per_s": round(traj_s, 2),
                        "actions_per_s": round(traj_s * eff_c, 1),
                        "peak_hbm_gib_per_tc": round(bf16_peak_gib, 2),
                        "status": "PASS",
                    }
                )
                log(mkey, f"[DP=8 ClosedLoop] CON={conc} -> P50={pct(tot_lats,50):.2f}ms, P99={pct(tot_lats,99):.2f}ms, {traj_s*eff_c:.1f} act/s")

            for rate_hz in (50, 100, 500, 1000):
                dur_s = 1.5
                n_arrivals = min(500, max(25, int(rate_hz * dur_s)))
                rng = random.Random(42 + rate_hz)
                ia_list = [rng.expovariate(rate_hz) for _ in range(n_arrivals)]
                futs_p = []
                t_p0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=128) as ppool:
                    target_t = t_p0
                    for ia in ia_list:
                        target_t += ia
                        now_t = time.perf_counter()
                        if target_t > now_t:
                            time.sleep(target_t - now_t)
                        futs_p.append(ppool.submit(pool_ka.call_infer, mkey, 1))
                    outs_p = [f.result() for f in futs_p]
                wall_p = time.perf_counter() - t_p0
                tot_lats = [x["e2e_rpc_ms"] for x in outs_p]
                ach_hz = n_arrivals / max(wall_p, 1e-6)
                _append_row(
                    {
                        "suite": "open_loop_poisson_dp8",
                        "model_key": mkey,
                        "repo_id": spec["repo_id"],
                        "paradigm": spec["paradigm"],
                        "topology": "1-Host DP=8 (8 TC)",
                        "precision": "BF16",
                        "backend": "torchtpu-compiled",
                        "target_arrival_hz": rate_hz,
                        "achieved_rate_hz": round(ach_hz, 1),
                        "batch_size": 1,
                        "action_chunk": eff_c,
                        "solver_or_decode_steps": fast_st,
                        "e2e_p50_ms": round(pct(tot_lats, 50), 2),
                        "e2e_p95_ms": round(pct(tot_lats, 95), 2),
                        "e2e_p99_ms": round(pct(tot_lats, 99), 2),
                        "sla_100hz_10ms_p99": pct(tot_lats, 99) <= 10.0,
                        "sla_50hz_20ms_p99": pct(tot_lats, 99) <= 20.0,
                        "sla_20hz_50ms_p99": pct(tot_lats, 99) <= 50.0,
                        "actions_per_s": round(ach_hz * eff_c, 1),
                        "peak_hbm_gib_per_tc": round(bf16_peak_gib, 2),
                        "status": "PASS",
                    }
                )
                log(mkey, f"[DP=8 Poisson] Target={rate_hz}Hz -> Achieved={ach_hz:.1f}Hz, P50={pct(tot_lats,50):.2f}ms, P99={pct(tot_lats,99):.2f}ms")

        dist.barrier(group=gloo_pg)
        ep_srv.shutdown()
        ep_srv.server_close()
        del ip1, bf16_bank
        gc.collect()
        torch.accelerator.synchronize()

        # 5. Intra-Chip TP=2 (`1 Physical v7x Chip` = `2 TensorCores` via `tpu_dist` ICI `all_reduce`)
        torch._dynamo.reset()
        tp2_bank, tp2_mod_gib = _build_tpu_bank(spec, "BF16", list(wpool_raw), tp_degree=2)
        tp2_hbm_gib = max(tp2_mod_gib, round(sf_info["bf16_equivalent_gib"] * 0.52, 2))
        loc_pre, dis_rtt = _preproc_stats(2, def_res)
        with torch.no_grad():
            for bsz in (1, 4, 8, 16):
                inp_tp = _make_inputs(spec, bsz, 2, def_res, eff_c)
                for _ in range(3):
                    _ = _forward_bank(tp2_bank, inp_tp, fast_st, compiled=True, tp_sync=True)
                    torch.accelerator.synchronize()
                dist.barrier(group=gloo_pg)
                tp_lats = []
                for _ in range(15):
                    dist.barrier(group=gloo_pg)
                    t0 = time.perf_counter()
                    _ = _forward_bank(tp2_bank, inp_tp, fast_st, compiled=True, tp_sync=True)
                    torch.accelerator.synchronize()
                    dist.barrier(group=gloo_pg)
                    if rank == 0:
                        tp_lats.append((time.perf_counter() - t0) * 1000.0)
                del inp_tp
                if rank == 0:
                    p50_tp = pct(tp_lats, 50)
                    p99_tp = pct(tp_lats, 99)
                    chip_traj_s = (bsz * 1000.0) / max(p50_tp, 1e-6)
                    _append_row(
                        {
                            "suite": "intrachip_tp2_topology",
                            "model_key": mkey,
                            "repo_id": spec["repo_id"],
                            "paradigm": spec["paradigm"],
                            "topology": "1-Chip Intra-Chip TP=2 (2 TC)",
                            "precision": "BF16",
                            "backend": "torchtpu-compiled",
                            "num_cameras": 2,
                            "resolution": f"{def_res}x{def_res}",
                            "action_chunk": eff_c,
                            "solver_or_decode_steps": fast_st,
                            "batch_size": bsz,
                            "tpu_p50_ms": round(p50_tp, 2),
                            "tpu_p95_ms": round(pct(tp_lats, 95), 2),
                            "tpu_p99_ms": round(p99_tp, 2),
                            "e2e_local_cpu_p50_ms": round(p50_tp + loc_pre, 2),
                            "e2e_disagg_cpu_p50_ms": round(p50_tp + dis_rtt, 2),
                            "e2e_disagg_pipelined_p50_ms": round(max(p50_tp, dis_rtt), 2),
                            "sla_100hz_10ms_tpu": p50_tp <= 10.0,
                            "sla_100hz_10ms_pipelined": max(p50_tp, dis_rtt) <= 10.0,
                            "sla_50hz_20ms_pipelined": max(p50_tp, dis_rtt) <= 20.0,
                            "sla_20hz_50ms_pipelined": max(p50_tp, dis_rtt) <= 50.0,
                            "chip_tp2_trajectories_per_s": round(chip_traj_s, 2),
                            "chip_tp2_actions_per_s": round(chip_traj_s * eff_c, 1),
                            "host_4chip_tp2x4_actions_per_s": round(chip_traj_s * eff_c * (world_size // 2), 1),
                            "peak_hbm_gib_per_tc": tp2_hbm_gib,
                            "status": "PASS",
                        }
                    )
                    log(mkey, f"[TP=2 Intra-Chip ICI] B={bsz} -> TPU_P50={p50_tp:.2f}ms (P99={p99_tp:.2f}ms), HBM/TC={tp2_hbm_gib}GiB")
        del tp2_bank
        gc.collect()
        torch.accelerator.synchronize()

        # 6. FP8 Dynamic (`torch._scaled_mm` E4M3) Sweep across B = 1, 2, 4, 8, 16, 32 (DP=1 & DP=8)
        torch._dynamo.reset()
        fp8_bank, fp8_mod_gib = _build_tpu_bank(spec, "FP8", list(wpool_raw), tp_degree=1)
        fp8_peak_gib = max(fp8_mod_gib, sf_info["fp8_equivalent_gib"])
        for bsz in (1, 2, 4, 8, 16, 32):
            _run_dp1_and_dp8_point(
                "fp8_dynamic_scaled_mm", "FP8", True, fp8_bank, 2, def_res, def_chunk, fast_st, bsz, fp8_peak_gib, measure_dp8=True
            )
        del fp8_bank, wpool_raw
        gc.collect()
        torch.accelerator.synchronize()
        dist.barrier(group=gloo_pg)
        if rank == 0:
            log(mkey, f"=== Completed all single-host sweeps for {spec['display_name']} ===")

    dist.barrier(group=gloo_pg)
    dist.destroy_process_group()


def expand_worker_urls(raw_urls: str, ranks_per_host: int = 8) -> list[str]:
    """Expands base host URLs (`http://ip:19100`) into all `ranks_per_host` TensorCore endpoints (`19100..19107`)."""
    items = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if len(items) >= ranks_per_host:
        return items
    expanded: list[str] = []
    for u in items:
        p = urllib.parse.urlsplit(u)
        host = p.hostname or "127.0.0.1"
        base_port = p.port or 19100
        for r in range(ranks_per_host):
            expanded.append(f"{p.scheme or 'http'}://{host}:{base_port + r}")
    return expanded


def run_multihost_dp16_coordinator(
    worker_urls: list[str],
    model_keys: list[str],
    out_jsonl: Path,
    out_summary_json: Path,
) -> None:
    log("dp16-coord", f"Connecting to {len(worker_urls)} TPU v7x TensorCore endpoints across slice: {worker_urls[0]} .. {worker_urls[-1]}")
    pool_ka = MultiEndpointKeepAlivePool(worker_urls)
    pool_ka.wait_healthy(timeout_s=600.0)
    log("dp16-coord", f"All {len(worker_urls)} TensorCore HTTP Keep-Alive endpoints healthy!")

    rows: list[dict[str, Any]] = []
    if out_jsonl.exists():
        for ln in out_jsonl.read_text().splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _record(r: dict[str, Any]) -> None:
        rows.append(r)
        with open(out_jsonl, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(r) + "\n")
        out_summary_json.write_text(
            json.dumps(
                {
                    "updated_at_utc": utc_iso(),
                    "topology": f"2-Host Slice DP={len(worker_urls)} (8 Chips = {len(worker_urls)} TensorCores)",
                    "num_endpoints": len(worker_urls),
                    "num_rows": len(rows),
                    "results": rows,
                },
                indent=2,
            )
        )

    n_eps = len(worker_urls)
    for mkey in model_keys:
        spec = MODEL_SPECS[mkey]
        eff_c = 1 if spec["family"] == "openvla_ar" else spec["default_chunk"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_eps) as wpool:
            list(wpool.map(lambda _: pool_ka.call_infer(mkey, 1), range(n_eps * 2)))

        for bsz in (1, 4, 8, 16):
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_eps) as wpool:
                list(wpool.map(lambda _: pool_ka.call_infer(mkey, bsz), range(n_eps)))
                n_reqs = n_eps * 4
                t0 = time.perf_counter()
                outs = list(wpool.map(lambda _: pool_ka.call_infer(mkey, bsz), range(n_reqs)))
                wall_s = time.perf_counter() - t0
            e2e_l = [x["e2e_rpc_ms"] for x in outs]
            tpu_l = [x["tpu_ms"] for x in outs]
            traj_s = (n_reqs * bsz) / max(wall_s, 1e-6)
            rec = {
                "suite": "multihost_dp16_batch_scaling",
                "model_key": mkey,
                "repo_id": spec["repo_id"],
                "paradigm": spec["paradigm"],
                "topology": f"2-Host Slice DP={n_eps} ({n_eps} TC)",
                "precision": "BF16",
                "backend": "torchtpu-compiled",
                "concurrency": n_eps,
                "batch_size": bsz,
                "action_chunk": eff_c,
                "solver_or_decode_steps": spec["fast_steps"],
                "tpu_p50_ms": round(pct(tpu_l, 50), 2),
                "e2e_rpc_p50_ms": round(pct(e2e_l, 50), 2),
                "e2e_rpc_p95_ms": round(pct(e2e_l, 95), 2),
                "e2e_rpc_p99_ms": round(pct(e2e_l, 99), 2),
                "sla_100hz_10ms": pct(e2e_l, 50) <= 10.0,
                "sla_50hz_20ms": pct(e2e_l, 50) <= 20.0,
                "sla_20hz_50ms": pct(e2e_l, 50) <= 50.0,
                "slice_dp16_trajectories_per_s": round(traj_s, 2),
                "slice_dp16_actions_per_s": round(traj_s * eff_c, 1),
                "peak_hbm_gib_per_tc": outs[0].get("hbm_gib"),
                "status": "PASS",
            }
            _record(rec)
            log("dp16-coord", f"[{mkey}][DP=16 Batch] B={bsz} CON={n_eps} -> E2E_P50={rec['e2e_rpc_p50_ms']}ms, Slice={rec['slice_dp16_actions_per_s']} act/s")

        for conc in (1, 4, 8, 16, 32, 64):
            n_reqs = max(32, conc * 3)
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as cpool:
                outs = list(cpool.map(lambda _: pool_ka.call_infer(mkey, 1), range(n_reqs)))
            wall_s = time.perf_counter() - t0
            e2e_l = [x["e2e_rpc_ms"] for x in outs]
            tpu_l = [x["tpu_ms"] for x in outs]
            traj_s = n_reqs / max(wall_s, 1e-6)
            rec = {
                "suite": "multihost_dp16_closed_loop_concurrency",
                "model_key": mkey,
                "repo_id": spec["repo_id"],
                "paradigm": spec["paradigm"],
                "topology": f"2-Host Slice DP={n_eps} ({n_eps} TC)",
                "precision": "BF16",
                "backend": "torchtpu-compiled",
                "concurrency": conc,
                "batch_size": 1,
                "action_chunk": eff_c,
                "solver_or_decode_steps": spec["fast_steps"],
                "tpu_p50_ms": round(pct(tpu_l, 50), 2),
                "e2e_rpc_p50_ms": round(pct(e2e_l, 50), 2),
                "e2e_rpc_p95_ms": round(pct(e2e_l, 95), 2),
                "e2e_rpc_p99_ms": round(pct(e2e_l, 99), 2),
                "sla_100hz_10ms": pct(e2e_l, 50) <= 10.0,
                "sla_50hz_20ms": pct(e2e_l, 50) <= 20.0,
                "sla_20hz_50ms": pct(e2e_l, 50) <= 50.0,
                "slice_dp16_trajectories_per_s": round(traj_s, 2),
                "slice_dp16_actions_per_s": round(traj_s * eff_c, 1),
                "peak_hbm_gib_per_tc": outs[0].get("hbm_gib"),
                "status": "PASS",
            }
            _record(rec)
            log("dp16-coord", f"[{mkey}][DP=16 ClosedLoop] CON={conc} -> P50={rec['e2e_rpc_p50_ms']}ms, P99={rec['e2e_rpc_p99_ms']}ms, {rec['slice_dp16_trajectories_per_s']} traj/s")

        for rate_hz in (50, 100, 500, 1000):
            dur_s = 1.5
            n_arr = min(600, max(30, int(rate_hz * dur_s)))
            rng = random.Random(100 + rate_hz)
            ia_list = [rng.expovariate(rate_hz) for _ in range(n_arr)]
            futs = []
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=192) as ppool:
                nxt = t0
                for ia in ia_list:
                    nxt += ia
                    now = time.perf_counter()
                    if nxt > now:
                        time.sleep(nxt - now)
                    futs.append(ppool.submit(pool_ka.call_infer, mkey, 1))
                outs = [f.result() for f in futs]
            wall_s = time.perf_counter() - t0
            e2e_l = [x["e2e_rpc_ms"] for x in outs]
            ach_hz = n_arr / max(wall_s, 1e-6)
            rec = {
                "suite": "multihost_dp16_open_loop_poisson",
                "model_key": mkey,
                "repo_id": spec["repo_id"],
                "paradigm": spec["paradigm"],
                "topology": f"2-Host Slice DP={n_eps} ({n_eps} TC)",
                "precision": "BF16",
                "backend": "torchtpu-compiled",
                "target_arrival_hz": rate_hz,
                "achieved_rate_hz": round(ach_hz, 1),
                "batch_size": 1,
                "action_chunk": eff_c,
                "solver_or_decode_steps": spec["fast_steps"],
                "e2e_rpc_p50_ms": round(pct(e2e_l, 50), 2),
                "e2e_rpc_p95_ms": round(pct(e2e_l, 95), 2),
                "e2e_rpc_p99_ms": round(pct(e2e_l, 99), 2),
                "sla_100hz_10ms_p99": pct(e2e_l, 99) <= 10.0,
                "sla_50hz_20ms_p99": pct(e2e_l, 99) <= 20.0,
                "sla_20hz_50ms_p99": pct(e2e_l, 99) <= 50.0,
                "slice_dp16_actions_per_s": round(ach_hz * eff_c, 1),
                "peak_hbm_gib_per_tc": outs[0].get("hbm_gib"),
                "status": "PASS",
            }
            _record(rec)
            log("dp16-coord", f"[{mkey}][DP=16 Poisson] Target={rate_hz}Hz -> Achieved={rec['achieved_rate_hz']}Hz, P50={rec['e2e_rpc_p50_ms']}ms, P99={rec['e2e_rpc_p99_ms']}ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical AI VLA Quad-Architecture Triathlon on Cloud TPU v7x")
    parser.add_argument("--torchrun-worker", action="store_true")
    parser.add_argument("--models-root", default="/models/models")
    parser.add_argument("--models", default="pi05_base,pi05_libero,gr00t_n1_7_3b,cosmos3_edge_droid,openvla_7b,openvla_7b_oft")
    parser.add_argument("--cores-per-host", type=int, default=8)
    parser.add_argument("--output-dir", default="/tmp/vla_triathlon_results")
    parser.add_argument("--tag", default="host0")
    parser.add_argument("--preproc-json", default="")
    parser.add_argument("--serve-worker-port", type=int, default=0)
    parser.add_argument("--run-multihost-dp16", action="store_true")
    parser.add_argument("--worker-urls", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.run_multihost_dp16:
        urls = expand_worker_urls(args.worker_urls, ranks_per_host=args.cores_per_host)
        run_multihost_dp16_coordinator(
            worker_urls=urls,
            model_keys=model_keys,
            out_jsonl=out_dir / "multihost_dp16_results.jsonl",
            out_summary_json=out_dir / "multihost_dp16_summary.json",
        )
        return 0

    if args.torchrun_worker:
        run_torchrun_vla_worker(
            models_root=Path(args.models_root),
            model_keys=model_keys,
            preproc_json=Path(args.preproc_json),
            out_jsonl=out_dir / f"{args.tag}_results.jsonl",
            out_summary_json=out_dir / f"{args.tag}_summary.json",
            serve_port=args.serve_worker_port,
        )
        return 0

    preproc_path = out_dir / f"{args.tag}_vision_preproc.json"
    if args.serve_worker_port == 0 and not preproc_path.exists():
        preproc_map = measure_vision_preproc_matrix()
        preproc_path.write_text(json.dumps(preproc_map, indent=2))
    elif not preproc_path.exists():
        preproc_path.write_text("{}")

    child_env = start_single_host_metadata_env("v7x", port=18080)
    child_env["TORCH_TPU_TOPOLOGY"] = "2,2,1,2"
    sb_base = 29540 if args.serve_worker_port == 0 else 29560
    master_port = 29508 if args.serve_worker_port == 0 else 29518
    child_env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(f"127.0.0.1:{sb_base + i}" for i in range(args.cores_per_host))

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={args.cores_per_host}",
        f"--master_port={master_port}",
        str(Path(__file__).resolve()),
        "--torchrun-worker",
        f"--models-root={args.models_root}",
        f"--models={','.join(model_keys)}",
        f"--cores-per-host={args.cores_per_host}",
        f"--output-dir={out_dir}",
        f"--tag={args.tag}",
        f"--preproc-json={preproc_path}",
        f"--serve-worker-port={args.serve_worker_port}",
    ]
    log("parent", f"Spawning 8-process TorchTPU worker pool ({args.tag}): {' '.join(cmd)}")
    res = subprocess.run(cmd, env=child_env, check=False)
    log("parent", f"8-process TorchTPU worker pool ({args.tag}) exited with rc={res.returncode}")
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
