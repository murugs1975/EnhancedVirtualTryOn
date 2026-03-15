# Enhanced Virtual Try-On

A production-ready virtual try-on system 


---

## Architecture

```
┌────────────┐        ┌──────────────────────────────────────────────────────┐
│  Frontend  │  POST  │                   FastAPI Backend                   │
│  (Next.js) │───────▶│                                                    │
│  port 3000 │◀───────│  ┌─────────────────┐    ┌───────────────────────┐  │
└────────────┘  PNG   │  │  Preprocessing  │    │   ONNX Inference      │  │
                      │  │  ─ MediaPipe    │───▶│   Stage 1: tocg.onnx  │  │
                      │  │  ─ SCHP parsing │    │   (ConditionGenerator)│  │
                      │  │  ─ Cloth mask   │    │          │            │  │
                      │  └─────────────────┘    │   Post-processing     │  │
                      │                         │   (flow warp, segmap) │  │
                      │                         │          │            │  │
                      │                         │   Stage 2: gen.onnx   │  │
                      │                         │   (SPADEGenerator)    │  │
                      │                         └───────────────────────┘  │
                      │                                   port 8000       │
                      └──────────────────────────────────────────────────────┘
```

The pipeline runs two neural-network stages:

1. **ConditionGenerator** (`tocg.onnx`) — predicts optical flow to warp the garment onto the body, plus a 13-channel segmentation map.
2. **SPADEGenerator** (`gen.onnx`) — synthesises the final try-on image from the warped garment, body parse, and agnostic person inputs.

All inference runs on **ONNX Runtime (CPU)** — no GPU or PyTorch required at runtime.

---

## Project Structure

```
EnhancedVirtualTryOn/
│
├── api/                          # FastAPI backend (production runtime)
│   ├── app.py                    #   Main application — API endpoints
│   ├── config.py                 #   Settings (model paths, resolution, ports)
│   ├── inference.py              #   Two-stage ONNX inference engine
│   ├── preprocessing.py          #   Image loading, normalisation, resampling
│   ├── postprocessing.py         #   Flow warping, segmap processing, occlusion
│   ├── simple_preprocess.py      #   Simplified preprocessing (MediaPipe + SCHP)
│   └── requirements.txt          #   Python dependencies
│
├── frontend/                     # Next.js + Tailwind CSS web UI
│   ├── src/
│   │   ├── pages/index.tsx       #     Main page — upload, preview, results
│   │   ├── components/
│   │   │   ├── ImageUploader.tsx  #     Drag-and-drop image upload
│   │   │   ├── LogPanel.tsx      #     Real-time pipeline log viewer
│   │   │   └── ResultViewer.tsx  #     Try-on result display
│   │   ├── styles/               #     Global CSS
│   │   └── utils/                #     Shared utilities
│   ├── package.json
│   ├── tailwind.config.js
│   └── Dockerfile
│
├── onnx_models/                  # Exported ONNX models (runtime)
│   ├── tocg.onnx                 #   ConditionGenerator (~180 MB)
│   ├── gen.onnx                  #   SPADEGenerator     (~383 MB)
│   └── schp_lip.onnx            #   SCHP human parser   (~254 MB)
│
├── checkpoints/                  # Original PyTorch weights (for export only)
│   ├── mtviton.pth               #   ConditionGenerator checkpoint
│   ├── gen.pth                   #   SPADEGenerator checkpoint
│   └── exp-schp-201908261155-lip.pth  # SCHP checkpoint
│
├── export_models.py              # Script: export tocg + gen to ONNX
├── export_schp.py                # Script: export SCHP parser to ONNX
│
├── model_architectures/           # PyTorch model definitions (used by export scripts)
│   ├── networks.py               #   ConditionGenerator architecture
│   └── network_generator.py      #   SPADEGenerator architecture
│
├── virtual-tryon/                # Research notebooks 
│   ├── virtual_tryon_v4.ipynb    #   Main notebook — HR-VITON inference
│   ├── virtual_tryon_v3.ipynb    #   Iteration 3 — dense-flow from scratch
│   ├── virtual_tryon_v2.ipynb    #   Iteration 2 — improved CP-VTON
│   ├── virtual_tryon.ipynb       #   Iteration 1 — LightTryOnNet baseline
│   └── README.md                 #   Detailed research notes + results
│
├── data/                         # VITON-HD dataset (not committed)
│
├── Dockerfile                    # API-only Docker image
├── Dockerfile.cloudrun           # Combined image (API + frontend + nginx)
├── docker-compose.yml            # Local multi-container setup
├── nginx.cloudrun.conf           # Nginx reverse-proxy config (Cloud Run)
├── supervisord.cloudrun.conf     # Process manager config (Cloud Run)
├── start-cloudrun.sh             # Cloud Run entrypoint script
├── deploy-cloudrun.sh            # GCP deployment script
├── cloudbuild.yaml               # Cloud Build config
├── .dockerignore
└── .gcloudignore
```

---

## Quick Start

### Docker Compose (recommended)

```bash
docker compose up --build
```

- **Frontend**: http://localhost:3001
- **API**: http://localhost:8020
- **API docs**: http://localhost:8020/docs

### Manual Setup

**Backend:**

```bash
cd api
pip install -r requirements.txt

# Ensure ONNX models are in api/models/ (or set env vars)
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check (model load status) |
| `GET` | `/model-status` | ONNX model input/output details |
| `POST` | `/tryon` | Full try-on with pre-processed VITON-HD inputs (5 files) |
| `POST` | `/tryon-simple` | Simplified try-on — upload person + cloth images only |
| `POST` | `/tryon-npy` | Try-on with raw `.npy` tensor inputs |
| `POST` | `/pipeline-preview` | Returns intermediate pipeline visualisations + logs |

### Simple Try-On Example

```bash
curl -X POST http://localhost:8000/tryon-simple \
  -F "person=@person.jpg" \
  -F "cloth=@garment.jpg" \
  --output result.png
```

---

## Model Export

To re-export ONNX models from PyTorch checkpoints:

```bash
# Export ConditionGenerator + SPADEGenerator
python export_models.py \
  --tocg-weights checkpoints/mtviton.pth \
  --gen-weights checkpoints/gen.pth \
  --output-dir onnx_models

# Export SCHP human parser (requires SCHP repo cloned alongside)
python export_schp.py
```

---

## Cloud Run Deployment

The project includes a combined Docker image (`Dockerfile.cloudrun`) that bundles the API, frontend, and nginx into a single container for Google Cloud Run:

```bash
# Build and deploy
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh
```

---

## Key Technologies

| Component | Technology |
|-----------|------------|
| Inference | ONNX Runtime (CPU) |
| Backend | FastAPI, Python 3.10 |
| Frontend | Next.js 14, React 18, Tailwind CSS |
| Pose Detection | MediaPipe Pose Landmarker |
| Human Parsing | SCHP (Self-Correction Human Parsing) |
| Container | Docker, Docker Compose |
| Cloud | Google Cloud Run, Cloud Build |

---

## References

- Lee et al., *High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions*, ECCV 2022 — [arXiv](https://arxiv.org/abs/2206.14180)
- Choi et al., *VITON-HD: High-Resolution Virtual Try-On*, CVPR 2021
- Park et al., *Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE)*, CVPR 2019
