#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-veriface-api}"
PRESET="balanced"   # strict | balanced | lenient
DEBUG="${DEBUG:-1}"
JSON_OVERRIDE=""

# Args: --preset strict|balanced|lenient  --json '{"lipsync":0.55,"blink":0.55,"voice":0.55}'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --json) JSON_OVERRIDE="$2"; shift 2 ;;
    --debug) DEBUG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

case "$PRESET" in
  strict)   THRESH='{"lipsync":0.65,"blink":0.65,"voice":0.65}' ;;
  balanced) THRESH='{"lipsync":0.55,"blink":0.55,"voice":0.55}' ;;
  lenient)  THRESH='{"lipsync":0.45,"blink":0.45,"voice":0.45}' ;;
  *) echo "Invalid --preset: $PRESET (use strict|balanced|lenient)"; exit 2 ;;
esac

if [[ -n "$JSON_OVERRIDE" ]]; then
  THRESH="$JSON_OVERRIDE"
fi

echo "Updating $SERVICE in $REGION"
echo "  VERIFACE_THRESHOLDS=$THRESH"
echo "  VERIFACE_DEBUG=$DEBUG"

# Use a safe delimiter (~) so JSON commas don't break parsing
gcloud run services update "$SERVICE" \
  --region "$REGION" \
  --set-env-vars ^~^VERIFACE_THRESHOLDS=$THRESH~VERIFACE_DEBUG=$DEBUG

echo "Done. Latest ready revision:"
gcloud run services describe "$SERVICE" --region "$REGION" \
  --format='value(status.latestReadyRevisionName)'

echo
echo "Current env on service:"
gcloud run services describe "$SERVICE" --region "$REGION" --format=json \
  | jq -r '.spec.template.spec.containers[0].env | map(select(.name=="VERIFACE_THRESHOLDS" or .name=="VERIFACE_DEBUG"))'
