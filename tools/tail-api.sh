#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-veriface-api}"
LEVEL="${LEVEL:-error}"   # info | warning | error | critical

while [[ $# -gt 0 ]]; do
  case "$1" in
    --level)   LEVEL="${2:-error}"; shift 2 ;;
    --region)  REGION="${2:-us-central1}"; shift 2 ;;
    --service) SERVICE="${2:-veriface-api}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

echo "Tailing logs for ${SERVICE} (severity >= ${LEVEL}) in ${REGION}"
echo "Press Ctrl+C to stop."

# Cloud Run now uses --log-filter instead of --filter or --level
gcloud beta run services logs tail "${SERVICE}" \
  --region "${REGION}" \
  --log-filter="severity>=\"${LEVEL^^}\""
