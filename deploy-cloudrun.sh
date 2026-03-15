#!/bin/bash
set -e

# ============================================================
# Deploy HR-VITON to Google Cloud Run
# ============================================================

PROJECT_ID="project-b75a91e8-32a7-4b31-b8b"
REGION="us-central1"
SERVICE_NAME="hr-viton"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== HR-VITON Cloud Run Deployment ==="
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Service:  $SERVICE_NAME"
echo "Image:    $IMAGE_NAME"
echo ""

# Step 1: Set the active project
echo ">>> Setting GCP project..."
gcloud config set project "$PROJECT_ID"

# Step 2: Enable required APIs
echo ">>> Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com

# Step 3: Build the Docker image using Cloud Build
# (builds remotely — no need for local Docker)
echo ">>> Building Docker image with Cloud Build..."
gcloud builds submit \
    --tag "$IMAGE_NAME" \
    --dockerfile Dockerfile.cloudrun \
    --timeout=1800 \
    --machine-type=e2-highcpu-8

# Step 4: Deploy to Cloud Run
echo ">>> Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --region "$REGION" \
    --platform managed \
    --memory 8Gi \
    --cpu 4 \
    --timeout 600 \
    --concurrency 4 \
    --min-instances 0 \
    --max-instances 3 \
    --port 8080 \
    --allow-unauthenticated \
    --set-env-vars "TOCG_MODEL_PATH=models/tocg.onnx,GEN_MODEL_PATH=models/gen.onnx,UPLOAD_DIR=/tmp/tryon/uploads,OUTPUT_DIR=/tmp/tryon/outputs"

# Step 5: Get the service URL
echo ""
echo "=== Deployment Complete ==="
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format "value(status.url)")
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test with:"
echo "  curl ${SERVICE_URL}/health"
