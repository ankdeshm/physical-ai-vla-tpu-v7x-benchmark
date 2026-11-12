#!/usr/bin/env bash
# Launch Physical AI VLA Quad-Architecture Triathlon directly on a 2-Host Cloud TPU v7x-8 Slice (GCE TPU VM).
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"
ZONE="${ZONE:-us-central1-c}"
TPU_NAME="${TPU_NAME:-tpu-v7x-8-vla-bench}"
GCS_BUCKET="${GCS_BUCKET:-YOUR_GCS_BUCKET}"
IMAGE_URI="${IMAGE_URI:-us-central1-docker.pkg.dev/${PROJECT_ID}/vllm/vllm-torchtpu-omni:v0.32.0}"

echo "[1/3] Syncing VLA Triathlon scripts to gs://${GCS_BUCKET}/vla-triathlon/scripts/ ..."
gcloud storage cp scripts/*.py "gs://${GCS_BUCKET}/vla-triathlon/scripts/" --project="${PROJECT_ID}"

echo "[2/3] Launching parallel single-host deep sweeps across Worker 0 & Worker 1 (16 TensorCores)..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=0 \
  --command="docker run --rm --privileged --net=host --shm-size=700g \
    -v /mnt/models:/models \
    -v /tmp/vla_triathlon_results:/tmp/vla_triathlon_results \
    ${IMAGE_URI} bash -lc '
      gcloud storage cp gs://${GCS_BUCKET}/vla-triathlon/scripts/*.py /tmp/ &&
      python3 -u /tmp/run_physical_ai_vla_v7x_triathlon.py \
        --models-root=/models/models \
        --models=pi05_base,pi05_libero,gr00t_n1_7_3b \
        --cores-per-host=8 \
        --output-dir=/tmp/vla_triathlon_results \
        --tag=host0
    '" &
PID_W0=$!

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=1 \
  --command="docker run --rm --privileged --net=host --shm-size=700g \
    -v /mnt/models:/models \
    -v /tmp/vla_triathlon_results:/tmp/vla_triathlon_results \
    ${IMAGE_URI} bash -lc '
      gcloud storage cp gs://${GCS_BUCKET}/vla-triathlon/scripts/*.py /tmp/ &&
      python3 -u /tmp/run_physical_ai_vla_v7x_triathlon.py \
        --models-root=/models/models \
        --models=cosmos3_edge_droid,openvla_7b,openvla_7b_oft \
        --cores-per-host=8 \
        --output-dir=/tmp/vla_triathlon_results \
        --tag=host1
    '" &
PID_W1=$!

wait "${PID_W0}" "${PID_W1}"
echo "[3/3] Completed Phase 1 single-host sweeps across all 6 VLA checkpoints!"
