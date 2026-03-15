"""
FastAPI application for HR-VITON virtual try-on inference.
Uses ONNX Runtime — no PyTorch dependency in production.
"""
import os
import uuid
import base64
import io
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image

from config import settings
from inference import get_inference
from simple_preprocess import preprocess_person_and_cloth
from preprocessing import downsample
from postprocessing import (
    upsample_flow_and_warp, apply_clothmask_composition,
    remove_overlap, softmax, gaussian_blur_segmap
)


def _tensor_to_rgb_uint8(tensor: np.ndarray) -> np.ndarray:
    arr = (tensor[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _mask_to_rgb_uint8(mask_tensor: np.ndarray) -> np.ndarray:
    mask = np.clip(mask_tensor[0, 0] * 255.0, 0, 255).astype(np.uint8)
    return np.stack([mask, mask, mask], axis=-1)


def _to_data_url(image_arr: np.ndarray, preview_size: tuple[int, int] = (576, 768)) -> str:
    img = Image.fromarray(image_arr)
    img.thumbnail(preview_size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print(f"BACKEND_STARTUP: Initializing lifespan in directory: {os.getcwd()}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    
    print(f"BACKEND_STARTUP: Model paths: TOCG={settings.TOCG_MODEL_PATH}, GEN={settings.GEN_MODEL_PATH}")
    print(f"BACKEND_STARTUP: Checking file existence...")
    for p in [settings.TOCG_MODEL_PATH, settings.GEN_MODEL_PATH, settings.POSE_MODEL_PATH]:
        exists = os.path.exists(p)
        print(f"BACKEND_STARTUP: Path '{p}' exists: {exists}")
        if exists:
            print(f"BACKEND_STARTUP: Size of '{p}': {os.path.getsize(p)} bytes")

    # Pre-load models
    try:
        print("BACKEND_STARTUP: Loading HR-VITON models...")
        engine = get_inference()
        if engine.is_loaded:
            print("BACKEND_STARTUP: HR-VITON models loaded successfully")
        else:
            print("BACKEND_STARTUP_WARNING: One or more ONNX models not found. Check model paths.")
    except Exception as e:
        print(f"BACKEND_STARTUP_ERROR: Failed to load models: {traceback.format_exc()}")
        
    yield


app = FastAPI(
    title="HR-VITON Virtual Try-On API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "HR-VITON Virtual Try-On API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "frontend": "http://localhost:3001",
    }


@app.get("/health")
async def health():
    engine = get_inference()
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded,
        "tocg_loaded": engine.tocg_session is not None,
        "gen_loaded": engine.gen_session is not None,
    }


@app.get("/model-status")
async def model_status():
    engine = get_inference()
    info = {"tocg": None, "gen": None}
    if engine.tocg_session:
        info["tocg"] = {
            "inputs": [{"name": i.name, "shape": i.shape} for i in engine.tocg_session.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape} for o in engine.tocg_session.get_outputs()],
        }
    if engine.gen_session:
        info["gen"] = {
            "inputs": [{"name": i.name, "shape": i.shape} for i in engine.gen_session.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape} for o in engine.gen_session.get_outputs()],
        }
    return info


@app.post("/tryon")
async def tryon(
    cloth: UploadFile = File(..., description="Cloth image"),
    cloth_mask: UploadFile = File(..., description="Cloth mask (binary)"),
    parse_agnostic: UploadFile = File(..., description="Parse-agnostic map"),
    densepose: UploadFile = File(..., description="DensePose image"),
    agnostic: UploadFile = File(..., description="Agnostic person image"),
):
    """
    Run virtual try-on with pre-processed VITON-HD inputs.

    All inputs should be images at 768x1024 resolution.
    Returns the try-on result as a PNG image.
    """
    engine = get_inference()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    request_id = str(uuid.uuid4())[:8]

    try:
        # Save uploaded files
        paths = {}
        for name, file in [
            ("cloth", cloth), ("cloth_mask", cloth_mask),
            ("parse_agnostic", parse_agnostic), ("densepose", densepose),
            ("agnostic", agnostic),
        ]:
            path = os.path.join(settings.UPLOAD_DIR, f"{request_id}_{name}.png")
            with open(path, "wb") as f:
                f.write(await file.read())
            paths[name] = path

        # Run inference
        result = engine.run_from_files(
            cloth_path=paths["cloth"],
            cloth_mask_path=paths["cloth_mask"],
            parse_agnostic_path=paths["parse_agnostic"],
            densepose_path=paths["densepose"],
            agnostic_path=paths["agnostic"],
        )

        # Save output
        output_path = os.path.join(settings.OUTPUT_DIR, f"{request_id}_output.png")
        Image.fromarray(result).save(output_path)

        return FileResponse(output_path, media_type="image/png", filename="tryon_result.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploads
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)


@app.post("/tryon-simple")
async def tryon_simple(
    person: UploadFile = File(..., description="Full-body person/model image"),
    cloth: UploadFile = File(..., description="Target cloth image"),
):
    """
    Simplified endpoint that accepts only person and cloth images.
    Internally builds proxy inputs required by HR-VITON ONNX pipeline.
    """
    engine = get_inference()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    request_id = str(uuid.uuid4())[:8]

    try:
        fine_h = settings.FINE_HEIGHT
        fine_w = settings.FINE_WIDTH
        import io

        person_img = Image.open(io.BytesIO(await person.read()))
        cloth_img = Image.open(io.BytesIO(await cloth.read()))
        preprocessed = preprocess_person_and_cloth(person_img, cloth_img, fine_h, fine_w)

        result = engine.run_from_preprocessed(
            cloth=preprocessed.cloth,
            cloth_mask=preprocessed.cloth_mask,
            parse_agnostic=preprocessed.parse_agnostic,
            densepose=preprocessed.densepose,
            agnostic=preprocessed.agnostic,
        )

        output_path = os.path.join(settings.OUTPUT_DIR, f"{request_id}_output_simple.png")
        Image.fromarray(result).save(output_path)
        return FileResponse(output_path, media_type="image/png", filename="tryon_result.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline-preview")
async def pipeline_preview(
    person: UploadFile = File(..., description="Full-body person/model image"),
    cloth: UploadFile = File(..., description="Target cloth image"),
):
    """Generate intermediate pipeline visualizations for UI preview."""
    engine = get_inference()
    if engine.tocg_session is None:
        raise HTTPException(status_code=503, detail="Condition model not loaded")

    try:
        fine_h = settings.FINE_HEIGHT
        fine_w = settings.FINE_WIDTH
        low_h = settings.LOW_HEIGHT
        low_w = settings.LOW_WIDTH

        person_img = Image.open(io.BytesIO(await person.read()))
        cloth_img = Image.open(io.BytesIO(await cloth.read()))

        # Collect all logs including detailed stage descriptions
        log_dicts = []

        # --- Stage: Input Preprocessing ---
        log_dicts.append({"step": "preprocessing", "level": "info",
                          "message": f"Resizing inputs to {fine_w}x{fine_h}. Running MediaPipe PoseLandmarker (heavy) for 33-point pose + segmentation mask."})

        preprocessed = preprocess_person_and_cloth(person_img, cloth_img, fine_h, fine_w)

        # Append preprocessing logs from simple_preprocess
        for l in preprocessed.logs:
            log_dicts.append({"step": l.step, "level": l.level, "message": l.message})

        # --- Stage: Downsampling ---
        log_dicts.append({"step": "downsampling", "level": "info",
                          "message": f"Downsampling all inputs from {fine_w}x{fine_h} to {low_w}x{low_h} for ConditionGenerator. Cloth+mask (bilinear+nearest), parse_agnostic (nearest), densepose (bilinear)."})

        cloth_down = downsample(preprocessed.cloth, low_h, low_w, 'bilinear')
        cloth_mask_down = downsample(preprocessed.cloth_mask, low_h, low_w, 'nearest')
        parse_agnostic_down = downsample(preprocessed.parse_agnostic, low_h, low_w, 'nearest')
        densepose_down = downsample(preprocessed.densepose, low_h, low_w, 'bilinear')

        # --- Stage: tocg model inference ---
        input1 = np.concatenate([cloth_down, cloth_mask_down], axis=1)
        input2 = np.concatenate([parse_agnostic_down, densepose_down], axis=1)
        log_dicts.append({"step": "tocg_model", "level": "info",
                          "message": f"Running ConditionGenerator ONNX model (tocg.onnx). Input1: cloth+mask [{input1.shape}], Input2: parse_agnostic+densepose [{input2.shape}]. Predicts 5-scale optical flow + 13-ch segmentation map."})

        flow, fake_segmap, _, warped_cm_lo = engine.tocg_session.run(
            None, {"input1": input1, "input2": input2}
        )

        log_dicts.append({"step": "tocg_model", "level": "info",
                          "message": f"tocg output: flow {flow.shape}, segmap {fake_segmap.shape}. Flow is at half-resolution (128x96) from the deepest encoder scale."})

        # --- Stage: Cloth mask composition ---
        fake_segmap = apply_clothmask_composition(fake_segmap, warped_cm_lo, mode='warp_grad')
        log_dicts.append({"step": "postprocessing", "level": "info",
                          "message": "Applied cloth mask composition (warp_grad mode): multiplied segmap channel 3 (upper-clothes) by warped cloth mask to constrain clothing region."})

        # --- Stage: Flow upsampling + warping ---
        flow_mean = float(np.abs(flow).mean())
        flow_max = float(np.abs(flow).max())
        log_dicts.append({"step": "warping", "level": "info",
                          "message": f"Upsampling flow from {flow.shape[1]}x{flow.shape[2]} to {fine_h}x{fine_w} (bilinear interpolation, matching F.interpolate reference). Flow magnitude: mean={flow_mean:.4f}, max={flow_max:.4f}."})
        if flow_mean < 0.005:
            log_dicts.append({"step": "warping", "level": "warn",
                              "message": "Flow is near-zero. Warped cloth may look unchanged."})

        warped_cloth, warped_clothmask = upsample_flow_and_warp(
            flow,
            preprocessed.cloth,
            preprocessed.cloth_mask,
            fine_h,
            fine_w,
            low_h,
            low_w,
        )

        log_dicts.append({"step": "warping", "level": "info",
                          "message": f"Warped cloth image and mask at full resolution ({fine_w}x{fine_h}) using cv2.remap with bilinear interpolation (matching F.grid_sample reference)."})

        # --- Stage: Occlusion handling ---
        from preprocessing import upsample
        fake_segmap_fine = upsample(fake_segmap, fine_h, fine_w, 'bilinear')
        fake_segmap_blurred = gaussian_blur_segmap(fake_segmap_fine)
        seg_softmax_val = softmax(fake_segmap_blurred, axis=1)
        warped_clothmask = remove_overlap(seg_softmax_val, warped_clothmask)
        warped_cloth = warped_cloth * warped_clothmask + np.ones_like(warped_cloth) * (1 - warped_clothmask)

        log_dicts.append({"step": "occlusion", "level": "info",
                          "message": "Applied occlusion handling: upsampled 13-ch segmap to full res, applied 15x15 Gaussian blur (sigma=3.0), computed softmax, subtracted body-part overlap from cloth mask. Non-cloth regions composited with white background."})

        cloth_mask_img = _mask_to_rgb_uint8(preprocessed.cloth_mask)
        agnostic_img = _tensor_to_rgb_uint8(preprocessed.agnostic)
        warped_cloth_img = _tensor_to_rgb_uint8(warped_cloth)

        return {
            "cloth_mask": _to_data_url(cloth_mask_img),
            "agnostic": _to_data_url(agnostic_img),
            "warped_cloth": _to_data_url(warped_cloth_img),
            "logs": log_dicts,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Pipeline preview error: {tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/tryon-npy")
async def tryon_npy(
    cloth: UploadFile = File(..., description="Cloth tensor (.npy, [1,3,1024,768])"),
    cloth_mask: UploadFile = File(..., description="Cloth mask tensor (.npy, [1,1,1024,768])"),
    parse_agnostic: UploadFile = File(..., description="Parse-agnostic tensor (.npy, [1,13,1024,768])"),
    densepose: UploadFile = File(..., description="DensePose tensor (.npy, [1,3,1024,768])"),
    agnostic: UploadFile = File(..., description="Agnostic tensor (.npy, [1,3,1024,768])"),
):
    """
    Run try-on with pre-processed numpy tensors.
    Useful for testing with exact VITON-HD data.
    """
    engine = get_inference()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    request_id = str(uuid.uuid4())[:8]

    try:
        import io

        cloth_arr = np.load(io.BytesIO(await cloth.read()))
        cloth_mask_arr = np.load(io.BytesIO(await cloth_mask.read()))
        parse_agnostic_arr = np.load(io.BytesIO(await parse_agnostic.read()))
        densepose_arr = np.load(io.BytesIO(await densepose.read()))
        agnostic_arr = np.load(io.BytesIO(await agnostic.read()))

        result = engine.run_from_preprocessed(
            cloth_arr, cloth_mask_arr, parse_agnostic_arr,
            densepose_arr, agnostic_arr
        )

        output_path = os.path.join(settings.OUTPUT_DIR, f"{request_id}_output.png")
        Image.fromarray(result).save(output_path)

        return FileResponse(output_path, media_type="image/png", filename="tryon_result.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
