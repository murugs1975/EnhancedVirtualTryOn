"""Utilities to derive HR-VITON proxy inputs from person + cloth images."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# CIHP 20-class → HR-VITON 13-channel mapping (matches original cp_dataset_test.py)
# Channel 0: background (CIHP 0=background, 10=jumpsuit/neck)
# Channel 1: hair       (CIHP 1=hat, 2=hair)
# Channel 2: face       (CIHP 4=sunglasses, 13=face)
# Channel 3: upper      (CIHP 5=upper-clothes, 6=dress, 7=coat)
# Channel 4: bottom     (CIHP 9=pants, 12=skirt)
# Channel 5: left arm   (CIHP 14)
# Channel 6: right arm  (CIHP 15)
# Channel 7: left leg   (CIHP 16)
# Channel 8: right leg  (CIHP 17)
# Channel 9: left shoe  (CIHP 18)
# Channel 10: right shoe (CIHP 19)
# Channel 11: socks     (CIHP 8)
# Channel 12: noise     (CIHP 3=glove, 11=scarf)
CIHP_TO_13CH = {
    0: 0, 1: 1, 2: 1, 3: 12, 4: 2, 5: 3, 6: 3, 7: 3,
    8: 11, 9: 4, 10: 0, 11: 12, 12: 4, 13: 2, 14: 5, 15: 6,
    16: 7, 17: 8, 18: 9, 19: 10,
}


@dataclass
class LogEntry:
    step: str       # e.g., "cloth_mask", "pose_detection", "agnostic"
    level: str      # "info", "warn", "error"
    message: str


@dataclass
class PreprocessedInputs:
    cloth: np.ndarray
    cloth_mask: np.ndarray
    parse_agnostic: np.ndarray
    densepose: np.ndarray
    agnostic: np.ndarray
    logs: List[LogEntry] = field(default_factory=list)


def _to_normalized_tensor(image: Image.Image, height: int, width: int) -> np.ndarray:
    resized = image.convert("RGB").resize((width, height))
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[np.newaxis]


def _build_cloth_mask(
    image: Image.Image, height: int, width: int, logs: List[LogEntry]
) -> np.ndarray:
    resized = image.resize((width, height))

    # Check if image has genuine alpha transparency
    has_real_alpha = False
    if resized.mode == "RGBA":
        alpha_arr = np.array(resized)[:, :, 3]
        if alpha_arr.min() < 250:
            has_real_alpha = True

    if has_real_alpha:
        rgba_arr = np.array(resized.convert("RGBA"), dtype=np.float32) / 255.0
        alpha = rgba_arr[..., 3]
        mask = (alpha > 0.1).astype(np.float32)
        coverage = float(mask.mean()) * 100
        logs.append(LogEntry("cloth_mask", "info", f"Used alpha channel. Mask covers {coverage:.1f}% of image."))
    else:
        # GrabCut for opaque images (JPEG or PNG without transparency)
        rgb = np.array(resized.convert("RGB"), dtype=np.uint8)
        margin_x = max(5, width // 20)
        margin_y = max(5, height // 20)
        rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)

        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(rgb, grabcut_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1.0, 0.0
            ).astype(np.float32)
            coverage = float(mask.mean()) * 100
            logs.append(LogEntry("cloth_mask", "info", f"Used GrabCut segmentation. Mask covers {coverage:.1f}% of image."))
        except cv2.error as e:
            # Fallback: simple luminance threshold (non-white = foreground)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            mask = (gray < 0.92).astype(np.float32)
            coverage = float(mask.mean()) * 100
            logs.append(LogEntry("cloth_mask", "warn", f"GrabCut failed ({e}), used luminance threshold. Mask covers {coverage:.1f}%."))

    # Morphological cleanup: close small holes, remove small noise, soften edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    # Soften jagged edges: small blur + re-binarize
    mask = cv2.GaussianBlur(mask, (3, 3), 0.8)
    mask = (mask > 0.5).astype(np.float32)

    # Warn if mask looks wrong
    coverage_val = float(mask.mean()) * 100
    if coverage_val < 5:
        logs.append(LogEntry("cloth_mask", "warn", f"Mask coverage very low ({coverage_val:.1f}%). Cloth may not be detected."))
    elif coverage_val > 95:
        logs.append(LogEntry("cloth_mask", "warn", f"Mask coverage very high ({coverage_val:.1f}%). Background may not be removed."))

    return mask[np.newaxis, np.newaxis]


def _build_default_parse_agnostic(height: int, width: int, classes: int = 13) -> np.ndarray:
    parse_agnostic = np.zeros((1, classes, height, width), dtype=np.float32)
    parse_agnostic[:, 0, :, :] = 1.0
    return parse_agnostic


def _extract_pose_and_mask(
    person_rgb: np.ndarray, logs: List[LogEntry]
) -> Tuple[Optional[object], np.ndarray]:
    try:
        import mediapipe as mp
    except Exception:
        logs.append(LogEntry("pose_detection", "error", "MediaPipe not installed. Using fallback heuristics."))
        return None, np.ones(person_rgb.shape[:2], dtype=np.float32)

    # Resolve model path: check env, config default, then common local paths
    from config import settings
    model_path = os.environ.get("POSE_MODEL_PATH", getattr(settings, "POSE_MODEL_PATH", "models/pose_landmarker_heavy.task"))

    if not os.path.isfile(model_path):
        logs.append(LogEntry("pose_detection", "error", f"Pose model not found at {model_path}. Using fallback heuristics."))
        return None, np.ones(person_rgb.shape[:2], dtype=np.float32)

    try:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            output_segmentation_masks=True,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=person_rgb)
        result = landmarker.detect(mp_image)
        landmarker.close()
    except Exception as e:
        logs.append(LogEntry("pose_detection", "error", f"PoseLandmarker failed: {e}. Using fallback heuristics."))
        return None, np.ones(person_rgb.shape[:2], dtype=np.float32)

    # Extract segmentation mask
    seg_mask = np.ones(person_rgb.shape[:2], dtype=np.float32)
    if result.segmentation_masks and len(result.segmentation_masks) > 0:
        raw_mask = result.segmentation_masks[0].numpy_view()
        seg_mask = np.clip(raw_mask.astype(np.float32), 0.0, 1.0)
        # New API returns (H, W, 1) — squeeze to (H, W)
        if seg_mask.ndim == 3:
            seg_mask = seg_mask.squeeze(axis=-1)
        seg_coverage = float((seg_mask > 0.4).mean()) * 100
        logs.append(LogEntry("pose_detection", "info", f"Segmentation mask covers {seg_coverage:.1f}% of image."))
    else:
        logs.append(LogEntry("pose_detection", "warn", "No segmentation mask returned by MediaPipe."))

    # Extract landmarks (new API: result.pose_landmarks is list of list of NormalizedLandmark)
    landmarks = None
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]  # list of NormalizedLandmark with .x, .y, .visibility
        visible = sum(1 for lm in landmarks if getattr(lm, "visibility", 0) > 0.2)
        logs.append(LogEntry("pose_detection", "info", f"Detected {visible}/33 landmarks with visibility > 0.2."))
    else:
        logs.append(LogEntry("pose_detection", "warn", "No pose landmarks detected. Will use heuristic fallbacks."))

    return landmarks, seg_mask


def _landmark_px(
    landmarks: Optional[object],
    idx: int,
    width: int,
    height: int,
    min_visibility: float = 0.2,
) -> Optional[Tuple[int, int]]:
    if landmarks is None:
        return None
    lm = landmarks[idx]
    if getattr(lm, "visibility", 1.0) < min_visibility:
        return None
    x = int(np.clip(lm.x * width, 0, width - 1))
    y = int(np.clip(lm.y * height, 0, height - 1))
    return (x, y)


def _draw_limb(mask: np.ndarray, p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]], thickness: int) -> None:
    if p1 is None or p2 is None:
        return
    cv2.line(mask, p1, p2, color=1, thickness=thickness)
    cv2.circle(mask, p1, thickness // 2, color=1, thickness=-1)
    cv2.circle(mask, p2, thickness // 2, color=1, thickness=-1)


def _get_person_bbox(
    person_mask: np.ndarray, height: int, width: int
) -> Optional[Tuple[int, int, int, int]]:
    """Return (x_min, y_min, x_max, y_max) from the person segmentation mask."""
    fg_rows = np.where(person_mask.max(axis=1) > 0.4)[0]
    fg_cols = np.where(person_mask.max(axis=0) > 0.4)[0]
    if len(fg_rows) == 0 or len(fg_cols) == 0:
        return None
    return int(fg_cols[0]), int(fg_rows[0]), int(fg_cols[-1]), int(fg_rows[-1])


def _compute_r(landmarks: object, width: int, height: int) -> int:
    """Compute radius parameter from shoulder distance, matching original HR-VITON."""
    l_sh = _landmark_px(landmarks, 11, width, height)
    r_sh = _landmark_px(landmarks, 12, width, height)
    if l_sh is not None and r_sh is not None:
        length_a = np.linalg.norm(np.array(l_sh) - np.array(r_sh))
        return max(5, int(length_a / 16) + 1)
    return max(5, width // 60)


def _fill_limb_region(
    mask: np.ndarray,
    p1: Optional[Tuple[int, int]],
    p2: Optional[Tuple[int, int]],
    thickness: int,
) -> None:
    """Draw a thick filled limb region between two points."""
    if p1 is None or p2 is None:
        return
    cv2.line(mask, p1, p2, color=1, thickness=thickness)
    cv2.circle(mask, p1, thickness // 2, color=1, thickness=-1)
    cv2.circle(mask, p2, thickness // 2, color=1, thickness=-1)


# ---- SCHP (Self-Correction Human Parsing) ONNX inference ----
_schp_session = None
_schp_load_attempted = False


def _run_schp(
    person_rgb: np.ndarray, height: int, width: int, logs: List[LogEntry]
) -> Optional[np.ndarray]:
    """Run SCHP ONNX model to get pixel-precise CIHP labels (0-19).

    Input: person_rgb [H, W, 3] uint8 RGB
    Output: [H, W] uint8 label map (0-19), or None if model unavailable.
    """
    global _schp_session, _schp_load_attempted
    import onnxruntime as ort
    from config import settings

    if not _schp_load_attempted:
        _schp_load_attempted = True
        schp_path = getattr(settings, "SCHP_MODEL_PATH", "models/schp_lip.onnx")
        if os.path.exists(schp_path):
            _schp_session = ort.InferenceSession(schp_path, providers=['CPUExecutionProvider'])
            logs.append(LogEntry("schp", "info", f"Loaded SCHP model: {schp_path}"))
        else:
            logs.append(LogEntry("schp", "warn",
                                 f"SCHP model not found at {schp_path}. Falling back to landmark proxy."))

    if _schp_session is None:
        return None

    try:
        input_h, input_w = 473, 473

        # Preprocess: resize, BGR-order normalize (matches SCHP training)
        # SCHP was trained with cv2 (BGR), then ToTensor (keeps BGR channel order),
        # then Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        # Since our input is RGB, convert to BGR first.
        img_bgr = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img_f = img_resized.astype(np.float32) / 255.0
        # BGR-order mean/std (matching SCHP pretrained_settings)
        mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
        img_f = (img_f - mean) / std
        inp = img_f.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # [1, 3, 473, 473]

        # Run model
        output = _schp_session.run(None, {"input": inp})
        logits = output[0]  # [1, 20, 119, 119] (1/4 resolution)

        # Upsample logits to input resolution, then argmax
        # Use bilinear for logits (matches reference: F.interpolate + argmax)
        logits_up = np.zeros((20, input_h, input_w), dtype=np.float32)
        for c in range(20):
            logits_up[c] = cv2.resize(logits[0, c], (input_w, input_h),
                                      interpolation=cv2.INTER_LINEAR)
        labels_473 = np.argmax(logits_up, axis=0).astype(np.uint8)  # [473, 473]

        # Resize to target resolution (NEAREST preserves discrete labels)
        labels = cv2.resize(labels_473, (width, height), interpolation=cv2.INTER_NEAREST)

        # Log label distribution
        unique, counts = np.unique(labels, return_counts=True)
        total = labels.size
        dist_str = ", ".join(f"{u}:{c*100/total:.0f}%" for u, c in zip(unique, counts) if c*100/total > 1)
        logs.append(LogEntry("schp", "info", f"SCHP labels: {dist_str}"))

        return labels

    except Exception as e:
        logs.append(LogEntry("schp", "error", f"SCHP inference failed: {e}. Falling back to landmarks."))
        return None


# Detectron2 DensePose color palette (sampled from VITON-HD ground truth)
_DP_HEAD = (248, 250, 14)        # bright yellow
_DP_NECK = (145, 191, 115)       # olive green
_DP_TORSO = (20, 80, 193)        # solid blue
_DP_UPPER_ARM_L = (145, 191, 115)  # olive green (inner arm)
_DP_LOWER_ARM_L = (250, 228, 29)   # golden yellow
_DP_UPPER_ARM_R = (170, 189, 104)  # olive (slightly different)
_DP_LOWER_ARM_R = (251, 207, 46)   # golden-orange
_DP_UPPER_LEG_L = (22, 172, 184)   # cyan-teal
_DP_LOWER_LEG_L = (6, 166, 197)    # teal-blue
_DP_UPPER_LEG_R = (22, 172, 184)   # cyan-teal
_DP_LOWER_LEG_R = (7, 109, 221)    # blue


def _build_densepose_proxy(
    landmarks: Optional[object],
    person_mask: np.ndarray,
    height: int,
    width: int,
    logs: List[LogEntry],
) -> np.ndarray:
    """Build surface-filled DensePose proxy matching Detectron2 color palette.

    Creates filled body-part regions (not skeleton lines) using landmarks
    and person_mask, colored to match the ground truth DensePose images
    the tocg model was trained on.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    used_landmarks = False
    fg = (person_mask > 0.4).astype(np.uint8)

    if landmarks is not None:
        r = _compute_r(landmarks, width, height)
        idx = {
            "l_sh": 11, "r_sh": 12, "l_el": 13, "r_el": 14,
            "l_wr": 15, "r_wr": 16, "l_hip": 23, "r_hip": 24,
            "l_kn": 25, "r_kn": 26, "l_an": 27, "r_an": 28,
            "nose": 0, "l_ear": 7, "r_ear": 8,
        }
        pts: Dict[str, Optional[Tuple[int, int]]] = {
            k: _landmark_px(landmarks, v, width, height) for k, v in idx.items()
        }

        # Build individual body-part masks, then composite with priority
        # Lower priority painted first, higher priority overwrites

        # --- Legs (lowest priority for upper body model, but still needed) ---
        limb_th = max(20, r * 12)
        for (p1k, p2k, color) in [
            ("l_hip", "l_kn", _DP_UPPER_LEG_L),
            ("l_kn", "l_an", _DP_LOWER_LEG_L),
            ("r_hip", "r_kn", _DP_UPPER_LEG_R),
            ("r_kn", "r_an", _DP_LOWER_LEG_R),
        ]:
            p1, p2 = pts[p1k], pts[p2k]
            if p1 is not None and p2 is not None:
                leg_mask = np.zeros((height, width), dtype=np.uint8)
                _fill_limb_region(leg_mask, p1, p2, limb_th)
                # Intersect with person mask for natural body shape
                leg_mask = leg_mask & fg
                canvas[leg_mask > 0] = color

        # --- Torso (large filled region) ---
        torso_poly = [pts["l_sh"], pts["r_sh"], pts["r_hip"], pts["l_hip"]]
        if all(p is not None for p in torso_poly):
            torso_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(torso_mask, [np.array(torso_poly, dtype=np.int32)], 1)
            # Expand torso to fill the body silhouette between shoulders and hips
            sh_y = min(pts["l_sh"][1], pts["r_sh"][1])
            hip_y = max(pts["l_hip"][1], pts["r_hip"][1])
            torso_region = np.zeros((height, width), dtype=np.uint8)
            torso_region[sh_y:hip_y, :] = 1
            torso_mask = torso_mask | (torso_region & fg)
            # Wide edges
            cv2.line(torso_mask, pts["l_sh"], pts["l_hip"], 1, thickness=r * 8)
            cv2.line(torso_mask, pts["r_sh"], pts["r_hip"], 1, thickness=r * 8)
            cv2.line(torso_mask, pts["l_hip"], pts["r_hip"], 1, thickness=r * 10)
            torso_mask = torso_mask & fg
            canvas[torso_mask > 0] = _DP_TORSO
            used_landmarks = True

        # --- Neck (between chin and shoulders) ---
        if pts["nose"] is not None and pts["l_sh"] is not None and pts["r_sh"] is not None:
            neck_x = (pts["l_sh"][0] + pts["r_sh"][0]) // 2
            neck_y = (pts["l_sh"][1] + pts["r_sh"][1]) // 2
            chin_y = pts["nose"][1] + r * 2
            neck_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(neck_mask,
                          (neck_x - r * 5, chin_y),
                          (neck_x + r * 5, neck_y),
                          1, thickness=-1)
            neck_mask = neck_mask & fg
            canvas[neck_mask > 0] = _DP_NECK

        # --- Arms (higher priority than torso — painted on top) ---
        arm_th = max(20, r * 12)
        for (p1k, p2k, color) in [
            ("l_sh", "l_el", _DP_UPPER_ARM_L),
            ("l_el", "l_wr", _DP_LOWER_ARM_L),
            ("r_sh", "r_el", _DP_UPPER_ARM_R),
            ("r_el", "r_wr", _DP_LOWER_ARM_R),
        ]:
            p1, p2 = pts[p1k], pts[p2k]
            if p1 is not None and p2 is not None:
                arm_mask = np.zeros((height, width), dtype=np.uint8)
                _fill_limb_region(arm_mask, p1, p2, arm_th)
                arm_mask = arm_mask & fg
                canvas[arm_mask > 0] = color

        # Shoulder transition regions (ellipses)
        for sh_key, color in [("l_sh", _DP_UPPER_ARM_L), ("r_sh", _DP_UPPER_ARM_R)]:
            if pts[sh_key] is not None:
                sh_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.ellipse(sh_mask, pts[sh_key], (r * 5, r * 6), 0, 0, 360, 1, thickness=-1)
                sh_mask = sh_mask & fg
                canvas[sh_mask > 0] = color

        # --- Head (highest priority) ---
        if pts["nose"] is not None:
            head_mask = np.zeros((height, width), dtype=np.uint8)
            head_r = max(14, r * 5)
            cv2.circle(head_mask, pts["nose"], head_r, 1, thickness=-1)
            # Extend head upward within person mask
            eye_y = pts["nose"][1] - r * 3
            head_region = np.zeros((height, width), dtype=np.uint8)
            if pts["l_ear"] is not None and pts["r_ear"] is not None:
                ear_l = min(pts["l_ear"][0], pts["r_ear"][0]) - r * 2
                ear_r = max(pts["l_ear"][0], pts["r_ear"][0]) + r * 2
                head_region[:pts["nose"][1] + r, max(0, ear_l):min(width, ear_r)] = 1
            else:
                cx = pts["nose"][0]
                head_region[:pts["nose"][1] + r, max(0, cx - head_r):min(width, cx + head_r)] = 1
            head_mask = (head_mask | head_region) & fg
            canvas[head_mask > 0] = _DP_HEAD

    # Fallback: fill person silhouette with torso color
    if not used_landmarks:
        bbox = _get_person_bbox(person_mask, height, width)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            ph = y_max - y_min
            # Head
            head_y2 = y_min + int(ph * 0.12)
            canvas[y_min:head_y2, x_min:x_max][fg[y_min:head_y2, x_min:x_max] > 0] = _DP_HEAD
            # Torso
            torso_y2 = y_min + int(ph * 0.50)
            canvas[head_y2:torso_y2, x_min:x_max][fg[head_y2:torso_y2, x_min:x_max] > 0] = _DP_TORSO
            # Legs
            canvas[torso_y2:y_max, x_min:x_max][fg[torso_y2:y_max, x_min:x_max] > 0] = _DP_UPPER_LEG_L
            logs.append(LogEntry("densepose", "warn", "Used heuristic body fill (no landmarks)."))
        else:
            logs.append(LogEntry("densepose", "warn", "No person detected for densepose proxy."))
    else:
        # No blur — ground truth DensePose has sharp, distinct color boundaries
        has_color = (canvas.sum(axis=2) > 0).astype(np.uint8)
        coverage = float(has_color.mean()) * 100
        logs.append(LogEntry("densepose", "info",
                             f"Built surface-filled densepose proxy (sharp colors). Coverage: {coverage:.1f}%%."))

    # Zero out background
    canvas = canvas * fg[..., None]
    dense = canvas.astype(np.float32) / 255.0
    dense = dense * 2.0 - 1.0
    return dense.transpose(2, 0, 1)[np.newaxis]


def _build_densepose_from_cihp(
    cihp_labels: np.ndarray,
    landmarks: Optional[object],
    height: int,
    width: int,
    logs: List[LogEntry],
) -> np.ndarray:
    """Build DensePose proxy using pixel-precise SCHP labels + landmark joint splits.

    Uses SCHP for precise body-part boundaries (the hard part) and MediaPipe
    landmarks to split upper/lower arm and upper/lower leg at joints (the easy part).

    Returns: [1, 3, H, W] float32 in [-1, 1]
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Simple mapping: CIHP label → DensePose color (no upper/lower split)
    # Head: face(13) + hair(2) + hat(1)
    head_mask = (cihp_labels == 13) | (cihp_labels == 2) | (cihp_labels == 1)
    canvas[head_mask] = _DP_HEAD

    # Neck: label 10
    canvas[cihp_labels == 10] = _DP_NECK

    # Torso: upper-clothes(5) + dress(6) + coat(7)
    torso_mask = (cihp_labels == 5) | (cihp_labels == 6) | (cihp_labels == 7)
    canvas[torso_mask] = _DP_TORSO

    # Arms: split upper/lower at elbow using landmarks
    pts = {}
    if landmarks is not None:
        for name, mp_idx in [("l_el", 13), ("r_el", 14), ("l_kn", 25), ("r_kn", 26)]:
            pts[name] = _landmark_px(landmarks, mp_idx, width, height)

    # Left arm (label 14)
    left_arm = cihp_labels == 14
    if left_arm.any():
        if pts.get("l_el") is not None:
            elbow_y = pts["l_el"][1]
            canvas[left_arm & (np.arange(height)[:, None] <= elbow_y)] = _DP_UPPER_ARM_L
            canvas[left_arm & (np.arange(height)[:, None] > elbow_y)] = _DP_LOWER_ARM_L
        else:
            canvas[left_arm] = _DP_UPPER_ARM_L

    # Right arm (label 15)
    right_arm = cihp_labels == 15
    if right_arm.any():
        if pts.get("r_el") is not None:
            elbow_y = pts["r_el"][1]
            canvas[right_arm & (np.arange(height)[:, None] <= elbow_y)] = _DP_UPPER_ARM_R
            canvas[right_arm & (np.arange(height)[:, None] > elbow_y)] = _DP_LOWER_ARM_R
        else:
            canvas[right_arm] = _DP_UPPER_ARM_R

    # Left leg (label 16)
    left_leg = cihp_labels == 16
    if left_leg.any():
        if pts.get("l_kn") is not None:
            knee_y = pts["l_kn"][1]
            canvas[left_leg & (np.arange(height)[:, None] <= knee_y)] = _DP_UPPER_LEG_L
            canvas[left_leg & (np.arange(height)[:, None] > knee_y)] = _DP_LOWER_LEG_L
        else:
            canvas[left_leg] = _DP_UPPER_LEG_L

    # Right leg (label 17)
    right_leg = cihp_labels == 17
    if right_leg.any():
        if pts.get("r_kn") is not None:
            knee_y = pts["r_kn"][1]
            canvas[right_leg & (np.arange(height)[:, None] <= knee_y)] = _DP_UPPER_LEG_R
            canvas[right_leg & (np.arange(height)[:, None] > knee_y)] = _DP_LOWER_LEG_R
        else:
            canvas[right_leg] = _DP_UPPER_LEG_R

    # Shoes: left(18) → lower leg color, right(19) → lower leg color
    canvas[cihp_labels == 18] = _DP_LOWER_LEG_L
    canvas[cihp_labels == 19] = _DP_LOWER_LEG_R

    coverage = float((canvas.sum(axis=2) > 0).mean()) * 100
    logs.append(LogEntry("densepose", "info",
                         f"Built SCHP-enhanced densepose (pixel-precise boundaries). Coverage: {coverage:.1f}%%."))

    dense = canvas.astype(np.float32) / 255.0
    dense = dense * 2.0 - 1.0
    return dense.transpose(2, 0, 1)[np.newaxis]


def _cihp_to_13ch(cihp_labels: np.ndarray, classes: int = 13) -> np.ndarray:
    """Convert CIHP label map (0-19) to 13-channel one-hot, matching HR-VITON."""
    one_hot = np.zeros((classes, *cihp_labels.shape), dtype=np.float32)
    for cihp_label, channel in CIHP_TO_13CH.items():
        one_hot[channel] += (cihp_labels == cihp_label).astype(np.float32)
    return one_hot


def _build_cihp_labels(
    landmarks: Optional[object],
    person_mask: np.ndarray,
    height: int,
    width: int,
    logs: List[LogEntry],
) -> np.ndarray:
    """Build full CIHP label map (0-19) from landmarks and person mask.

    Returns a (height, width) uint8 array with CIHP labels.
    Used by both _build_parse_agnostic() and _build_agnostic_person().
    """
    cihp = np.zeros((height, width), dtype=np.uint8)
    fg = person_mask > 0.4
    used_landmarks = False

    if landmarks is not None:
        r = _compute_r(landmarks, width, height)
        idx = {
            "l_sh": 11, "r_sh": 12, "l_el": 13, "r_el": 14,
            "l_wr": 15, "r_wr": 16, "l_hip": 23, "r_hip": 24,
            "l_kn": 25, "r_kn": 26, "l_an": 27, "r_an": 28,
            "nose": 0, "l_ear": 7, "r_ear": 8,
            "l_eye_o": 3, "r_eye_o": 6,
        }
        pts: Dict[str, Optional[Tuple[int, int]]] = {
            k: _landmark_px(landmarks, v, width, height) for k, v in idx.items()
        }

        # --- Face (CIHP label 13) ---
        if pts["nose"] is not None:
            face_r = max(14, r * 4)
            cv2.circle(cihp, pts["nose"], face_r, 13, thickness=-1)
            for ear_key in ["l_ear", "r_ear"]:
                if pts[ear_key] is not None:
                    cv2.circle(cihp, pts[ear_key], face_r // 2, 13, thickness=-1)
            # Fill between ears if both visible
            if pts["l_ear"] is not None and pts["r_ear"] is not None:
                face_poly = [pts["l_ear"], pts["r_ear"],
                             (pts["r_ear"][0], pts["nose"][1] + face_r // 2),
                             (pts["l_ear"][0], pts["nose"][1] + face_r // 2)]
                cv2.fillPoly(cihp, [np.array(face_poly, dtype=np.int32)], 13)

        # --- Hair (CIHP label 2) — region above face within person mask ---
        if pts["nose"] is not None:
            eye_y = pts["nose"][1] - r * 3  # approximate top-of-face level
            hair_region = np.zeros((height, width), dtype=bool)
            hair_region[:max(0, eye_y), :] = True
            # Also include area around ears above nose
            if pts["l_ear"] is not None and pts["r_ear"] is not None:
                ear_left_x = min(pts["l_ear"][0], pts["r_ear"][0]) - r * 2
                ear_right_x = max(pts["l_ear"][0], pts["r_ear"][0]) + r * 2
                hair_region[:pts["nose"][1], ear_left_x:ear_right_x] = True
            cihp[hair_region & fg & (cihp == 0)] = 2

        # --- Lower body / pants (CIHP label 9) ---
        if pts["l_hip"] is not None and pts["r_hip"] is not None:
            hip_y = max(pts["l_hip"][1], pts["r_hip"][1])
            if pts["l_an"] is not None and pts["r_an"] is not None:
                ankle_y = max(pts["l_an"][1], pts["r_an"][1])
                lower = [pts["l_hip"], pts["r_hip"], pts["r_an"], pts["l_an"]]
                cv2.fillPoly(cihp, [np.array(lower, dtype=np.int32)], 9)
                cv2.line(cihp, pts["l_hip"], pts["l_an"], 9, thickness=r * 4)
                cv2.line(cihp, pts["r_hip"], pts["r_an"], 9, thickness=r * 4)
            else:
                fg_rows = np.where(person_mask.max(axis=1) > 0.4)[0]
                ankle_y = int(fg_rows[-1]) if len(fg_rows) > 0 else height - 1
            lower_region = np.zeros((height, width), dtype=bool)
            lower_region[hip_y:ankle_y, :] = True
            cihp[lower_region & fg & (cihp == 0)] = 9
            used_landmarks = True

        # --- Shoes (CIHP labels 18=left, 19=right) ---
        if pts["l_an"] is not None:
            cv2.circle(cihp, pts["l_an"], max(8, r * 2), 18, thickness=-1)
        if pts["r_an"] is not None:
            cv2.circle(cihp, pts["r_an"], max(8, r * 2), 19, thickness=-1)

        # --- Upper-clothes / torso (CIHP label 5) ---
        torso = [pts["l_sh"], pts["r_sh"], pts["r_hip"], pts["l_hip"]]
        if all(p is not None for p in torso):
            cv2.fillPoly(cihp, [np.array(torso, dtype=np.int32)], 5)
            cv2.line(cihp, pts["l_sh"], pts["l_hip"], 5, thickness=r * 6)
            cv2.line(cihp, pts["r_sh"], pts["r_hip"], 5, thickness=r * 6)
            used_landmarks = True

        # --- Neck (CIHP label 10) ---
        if pts["nose"] is not None and pts["l_sh"] is not None and pts["r_sh"] is not None:
            neck_x = (pts["l_sh"][0] + pts["r_sh"][0]) // 2
            neck_y = (pts["l_sh"][1] + pts["r_sh"][1]) // 2
            chin_y = pts["nose"][1] + r * 2
            cv2.rectangle(cihp,
                          (neck_x - r * 4, chin_y),
                          (neck_x + r * 4, neck_y),
                          10, thickness=-1)

        # --- Arms (CIHP 14=left, 15=right) ---
        arm_th = max(40, r * 10)
        for arm_label, joints in [
            (14, [("l_sh", "l_el"), ("l_el", "l_wr")]),
            (15, [("r_sh", "r_el"), ("r_el", "r_wr")]),
        ]:
            for p1k, p2k in joints:
                p1, p2 = pts[p1k], pts[p2k]
                if p1 is not None and p2 is not None:
                    cv2.line(cihp, p1, p2, arm_label, thickness=arm_th)
                    cv2.circle(cihp, p2, r * 5, arm_label, thickness=-1)

        # Remaining foreground between shoulders and hips → upper-clothes
        # Only assign within torso band, NOT everywhere (avoids over-graying arms/legs)
        if all(p is not None for p in [pts["l_sh"], pts["r_sh"], pts["l_hip"], pts["r_hip"]]):
            sh_y = min(pts["l_sh"][1], pts["r_sh"][1])
            hip_y = max(pts["l_hip"][1], pts["r_hip"][1])
            torso_band = np.zeros((height, width), dtype=bool)
            torso_band[sh_y:hip_y, :] = True
            cihp[(cihp == 0) & fg & torso_band] = 5

    # Fallback without landmarks
    if not used_landmarks:
        bbox = _get_person_bbox(person_mask, height, width)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            pw, ph = x_max - x_min, y_max - y_min
            cx = (x_min + x_max) // 2
            hair_y2 = y_min + int(ph * 0.10)
            hair_x1 = cx - int(pw * 0.20)
            hair_x2 = cx + int(pw * 0.20)
            cihp[y_min:hair_y2, hair_x1:hair_x2][fg[y_min:hair_y2, hair_x1:hair_x2]] = 2
            face_cy = y_min + int(ph * 0.13)
            face_r = int(pw * 0.12)
            face_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(face_mask, (cx, face_cy), face_r, 1, -1)
            cihp[(face_mask > 0) & fg & (cihp == 0)] = 13
            torso_y1 = y_min + int(ph * 0.18)
            torso_y2 = y_min + int(ph * 0.50)
            cihp[torso_y1:torso_y2, x_min:x_max][fg[torso_y1:torso_y2, x_min:x_max] & (cihp[torso_y1:torso_y2, x_min:x_max] == 0)] = 5
            lower_y2 = y_min + int(ph * 0.92)
            cihp[torso_y2:lower_y2, x_min:x_max][fg[torso_y2:lower_y2, x_min:x_max] & (cihp[torso_y2:lower_y2, x_min:x_max] == 0)] = 9
            cihp[(cihp == 0) & fg] = 5
            logs.append(LogEntry("parse_agnostic", "warn", "Used heuristic body regions (no landmarks)."))
        else:
            cihp[fg] = 5
            logs.append(LogEntry("parse_agnostic", "warn", "No person detected; set all foreground to torso."))

    cihp[~fg] = 0
    return cihp


def _build_parse_agnostic(
    landmarks: Optional[object],
    person_mask: np.ndarray,
    height: int,
    width: int,
    logs: List[LogEntry],
    classes: int = 13,
    cihp_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build parse-agnostic map using CIHP label convention.

    Creates a CIHP label map (or reuses provided one), then zeros out upper-body
    clothing regions to match ground-truth parse-agnostic format.
    Converts to 13-channel one-hot via CIHP_TO_13CH mapping.
    """
    fg = person_mask > 0.4

    if cihp_labels is None:
        cihp = _build_cihp_labels(landmarks, person_mask, height, width, logs)
    else:
        cihp = cihp_labels

    # Make agnostic — zero out upper body clothing regions
    agnostic = cihp.copy()
    for remove_label in [5, 6, 7, 10, 14, 15]:  # upper-clothes, dress, coat, neck, arms
        agnostic[agnostic == remove_label] = 0

    preserved_pct = float((agnostic > 0).mean()) * 100
    removed_pct = float(((cihp > 0) & (agnostic == 0) & fg).mean()) * 100
    logs.append(LogEntry("parse_agnostic", "info",
                         f"CIHP parse built. Preserved {preserved_pct:.1f}% (hair+face+lower). "
                         f"Removed {removed_pct:.1f}% (torso+arms+neck)."))

    # Convert to 13-channel one-hot
    one_hot = _cihp_to_13ch(agnostic, classes)
    return one_hot[np.newaxis]


def _build_agnostic_person(
    person_rgb: np.ndarray,
    person_mask: np.ndarray,
    logs: List[LogEntry],
    cihp_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build agnostic person image matching original HR-VITON get_agnostic().

    Starts with original person image (preserving background, face, hair, legs),
    then sets ONLY clothing-labeled pixels to gray using CIHP labels.
    This matches the reference: agnostic = im.copy() then draw gray over clothing.
    """
    height, width, _ = person_rgb.shape
    person = person_rgb.astype(np.float32) / 255.0
    gray_val = 0.5

    # Start with original person image (background + all regions preserved)
    agnostic = person.copy()

    # Labels to REMOVE (set to gray): upper-clothes, dress, coat, neck, left-arm, right-arm
    _CLOTHING_LABELS = [5, 6, 7, 10, 14, 15]

    if cihp_labels is not None:
        clothing_mask = np.zeros((height, width), dtype=bool)
        for label in _CLOTHING_LABELS:
            clothing_mask |= (cihp_labels == label)

        # Sharp binary mask (matching reference: ImageDraw with no blur)
        # Constrain to person silhouette
        clothing_mask = clothing_mask & (person_mask > 0.35)

        # Hard replace: set clothing pixels to gray (no feathering)
        agnostic[clothing_mask] = gray_val

        grayed_pct = float(clothing_mask.mean()) * 100
        logs.append(LogEntry("agnostic", "info",
                             f"Sharp mask approach: grayed {grayed_pct:.1f}% (clothing+arms+neck). "
                             f"Background, face, hair, lower body preserved from original."))
    else:
        # Fallback without CIHP labels: gray out torso heuristically
        bbox = _get_person_bbox(person_mask, height, width)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            pw, ph = x_max - x_min, y_max - y_min
            torso_y1 = max(0, y_min + int(ph * 0.15))
            torso_y2 = min(height, y_min + int(ph * 0.55))
            clothing_region = np.zeros((height, width), dtype=bool)
            clothing_region[torso_y1:torso_y2, :] = True
            apply_mask = clothing_region & (person_mask > 0.35)
            agnostic[apply_mask] = gray_val
            logs.append(LogEntry("agnostic", "warn", "Used heuristic torso rectangle (no CIHP labels)."))
        else:
            logs.append(LogEntry("agnostic", "warn", "No person detected. Agnostic unchanged."))

    agnostic = agnostic * 2.0 - 1.0
    return agnostic.transpose(2, 0, 1)[np.newaxis]


def preprocess_person_and_cloth(person_img: Image.Image, cloth_img: Image.Image, height: int, width: int) -> PreprocessedInputs:
    logs: List[LogEntry] = []

    person_resized = person_img.convert("RGB").resize((width, height))
    cloth_resized = cloth_img.convert("RGBA").resize((width, height))

    person_rgb = np.array(person_resized, dtype=np.uint8)
    landmarks, person_mask = _extract_pose_and_mask(person_rgb, logs)

    cloth_tensor = _to_normalized_tensor(cloth_resized, height, width)
    cloth_mask = _build_cloth_mask(cloth_resized, height, width, logs)

    # Try real SCHP model first for pixel-precise CIHP labels
    cihp_labels = _run_schp(person_rgb, height, width, logs)
    use_schp = cihp_labels is not None

    if not use_schp:
        # Fallback to landmark-based proxy
        cihp_labels = _build_cihp_labels(landmarks, person_mask, height, width, logs)

    # DensePose: use SCHP labels for better boundaries if available
    if use_schp:
        densepose = _build_densepose_from_cihp(cihp_labels, landmarks, height, width, logs)
    else:
        densepose = _build_densepose_proxy(landmarks, person_mask, height, width, logs)
    parse_agnostic = _build_parse_agnostic(landmarks, person_mask, height, width, logs,
                                           cihp_labels=cihp_labels)
    agnostic = _build_agnostic_person(person_rgb, person_mask, logs,
                                      cihp_labels=cihp_labels)

    return PreprocessedInputs(
        cloth=cloth_tensor,
        cloth_mask=cloth_mask,
        parse_agnostic=parse_agnostic,
        densepose=densepose,
        agnostic=agnostic,
        logs=logs,
    )
