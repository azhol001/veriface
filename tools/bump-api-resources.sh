#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-veriface-api}"
MEM="${MEM:-8Gi}"
CPU="${CPU:-4}"
TIMEOUT="${TIMEOUT:-600}"
CONCURRENCY="${CONCURRENCY:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"   # set to 1 to reduce cold starts

# Flags override env/defaults
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="${2}"; shift 2 ;;
    --service) SERVICE="${2}"; shift 2 ;;
    --mem|--memory) MEM="${2}"; shift 2 ;;
    --cpu) CPU="${2}"; shift 2 ;;
    --timeout) TIMEOUT="${2}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2}"; shift 2 ;;
    --min-instances) MIN_INSTANCES="${2}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

echo "Updating ${SERVICE} in ${REGION} -> mem=${MEM} cpu=${CPU} timeout=${TIMEOUT}s conc=${CONCURRENCY} min-instances=${MIN_INSTANCES}"
gcloud run services update "${SERVICE}" \
  --region "${REGION}" \
  --memory "${MEM}" \
  --cpu "${CPU}" \
  --timeout "${TIMEOUT}" \
  --concurrency "${CONCURRENCY}" \
  --min-instances "${MIN_INSTANCES}" \
  --set-env-vars VERIFACE_DEBUG=1

echo "Done. Latest ready revision:"
gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --format='value(status.latestReadyRevisionName)'
