#!/usr/bin/env bash
set -euo pipefail

# Usage: tools/analyze.sh path/to/video.mp4 [--timeout 300] [--region us-central1] [--service veriface-api]
FILE="${1:-}"
TIMEOUT=300
REGION="us-central1"
SERVICE="veriface-api"

# Parse optional flags
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) TIMEOUT="${2:-300}"; shift 2 ;;
    --region)  REGION="${2:-us-central1}"; shift 2 ;;
    --service) SERVICE="${2:-veriface-api}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "${FILE}" ]]; then
  echo "Usage: $0 /path/to/video.(mp4|mov|mkv|webm) [--timeout 300] [--region us-central1] [--service veriface-api]"
  exit 1
fi
if [[ ! -f "${FILE}" ]]; then
  echo "File not found: ${FILE}"
  exit 1
fi

# Resolve API URL: prefer env, else gcloud
API_URL="${API_BASE_URL:-}"
if [[ -z "${API_URL}" ]]; then
  API_URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
fi

echo "Analyzing: ${FILE}"
echo "API_URL:   ${API_URL}"
echo "Timeout:   ${TIMEOUT}s"
echo

# Infer mime type (fallback to mp4)
MIME="video/mp4"
case "${FILE##*.}" in
  mp4) MIME="video/mp4" ;;
  mov) MIME="video/quicktime" ;;
  mkv) MIME="video/x-matroska" ;;
  webm) MIME="video/webm" ;;
esac

# POST the file
curl -v -X POST \
  --max-time "${TIMEOUT}" --connect-timeout 15 \
  -H "Accept: application/json" \
  -F "file=@${FILE};type=${MIME}" \
  "${API_URL}/analyze" --fail-with-body \
  -w "\nHTTP:%{http_code} TIME_TOTAL:%{time_total}s SIZE_UPLOAD:%{size_upload}B\n"
