"""
HR-VITON ONNX inference pipeline.
Two-stage: ConditionGenerator (tocg) -> post-processing -> SPADEGenerator.
"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

from config import settings
from preprocessing import (
    load_and_normalize_image, load_mask, load_parse_map,
    load_densepose, downsample, upsample
)
from postprocessing import (
    gaussian_blur_segmap, segmap_to_parse7, apply_clothmask_composition,
    remove_overlap, upsample_flow_and_warp, softmax, match_cloth_color
)


class HRVITONInference:
    """End-to-end HR-VITON inference using ONNX Runtime."""

    def __init__(self, tocg_path: str = None, gen_path: str = None):
        tocg_path = tocg_path or settings.TOCG_MODEL_PATH
        gen_path = gen_path or settings.GEN_MODEL_PATH

        providers = ['CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # 0 = use all available CPU cores
        opts.inter_op_num_threads = 0

        self.tocg_session = None
        self.gen_session = None

        if os.path.exists(tocg_path):
            self.tocg_session = ort.InferenceSession(tocg_path, sess_options=opts, providers=providers)
            print(f"Loaded tocg model: {tocg_path}")

        if os.path.exists(gen_path):
            self.gen_session = ort.InferenceSession(gen_path, sess_options=opts, providers=providers)
            print(f"Loaded generator model: {gen_path}")

    @property
    def is_loaded(self) -> bool:
        return self.tocg_session is not None and self.gen_session is not None

    def run_from_preprocessed(
        self,
        cloth: np.ndarray,        # [1, 3, fine_H, fine_W] [-1,1]
        cloth_mask: np.ndarray,    # [1, 1, fine_H, fine_W] {0,1}
        parse_agnostic: np.ndarray,  # [1, 13, fine_H, fine_W] one-hot
        densepose: np.ndarray,     # [1, 3, fine_H, fine_W] [-1,1]
        agnostic: np.ndarray,      # [1, 3, fine_H, fine_W] [-1,1]
        occlusion: bool = True,
    ) -> np.ndarray:
        """
        Run full two-stage pipeline on pre-processed inputs.

        Returns:
            output_image: [H, W, 3] uint8 RGB image
        """
        fine_h = settings.FINE_HEIGHT
        fine_w = settings.FINE_WIDTH
        low_h = settings.LOW_HEIGHT
        low_w = settings.LOW_WIDTH

        # ---- Stage 1: Downsample and run ConditionGenerator ----
        cloth_down = downsample(cloth, low_h, low_w, 'bilinear')
        cloth_mask_down = downsample(cloth_mask, low_h, low_w, 'nearest')
        parse_agnostic_down = downsample(parse_agnostic, low_h, low_w, 'nearest')
        densepose_down = downsample(densepose, low_h, low_w, 'bilinear')

        input1 = np.concatenate([cloth_down, cloth_mask_down], axis=1)   # [1, 4, 256, 192]
        input2 = np.concatenate([parse_agnostic_down, densepose_down], axis=1)  # [1, 16, 256, 192]

        flow, fake_segmap, warped_cloth_lo, warped_cm_lo = self.tocg_session.run(
            None, {"input1": input1, "input2": input2}
        )

        # ---- Inter-stage post-processing ----
        # Cloth mask composition
        fake_segmap = apply_clothmask_composition(fake_segmap, warped_cm_lo, mode='warp_grad')

        # Upsample segmap to fine resolution and apply Gaussian blur
        fake_segmap_fine = upsample(fake_segmap, fine_h, fine_w, 'bilinear')
        fake_segmap_blurred = gaussian_blur_segmap(fake_segmap_fine)

        # Convert 13-ch segmap to 7-ch parse
        parse7 = segmap_to_parse7(fake_segmap_blurred, fine_h, fine_w)

        # Warp cloth at full resolution using upsampled flow
        warped_cloth, warped_clothmask = upsample_flow_and_warp(
            flow, cloth, cloth_mask, fine_h, fine_w, low_h, low_w
        )

        # Occlusion handling
        if occlusion:
            seg_softmax = softmax(fake_segmap_blurred, axis=1)
            warped_clothmask = remove_overlap(seg_softmax, warped_clothmask)
            warped_cloth = warped_cloth * warped_clothmask + np.ones_like(warped_cloth) * (1 - warped_clothmask)

        # ---- Stage 2: Run SPADEGenerator ----
        gen_input = np.concatenate([agnostic, densepose, warped_cloth], axis=1)  # [1, 9, H, W]

        output = self.gen_session.run(None, {"input": gen_input, "segmap": parse7})
        output_tensor = output[0]  # [1, 3, H, W] in [-1, 1]

        # Convert to uint8 image (no post-processing — matches reference)
        img = (output_tensor[0].transpose(1, 2, 0) + 1) / 2 * 255  # CHW->HWC, [-1,1]->[0,255]
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def run_from_files(
        self,
        cloth_path: str,
        cloth_mask_path: str,
        parse_agnostic_path: str,
        densepose_path: str,
        agnostic_path: str,
        occlusion: bool = True,
    ) -> np.ndarray:
        """
        Run pipeline from file paths (VITON-HD formatted pre-processed data).

        Returns:
            output_image: [H, W, 3] uint8 RGB image
        """
        fine_h = settings.FINE_HEIGHT
        fine_w = settings.FINE_WIDTH

        cloth = load_and_normalize_image(cloth_path, fine_h, fine_w, 'neg1_1')
        cloth_mask = load_mask(cloth_mask_path, fine_h, fine_w)
        parse_agnostic = load_parse_map(parse_agnostic_path, fine_h, fine_w, 13)
        densepose = load_densepose(densepose_path, fine_h, fine_w)
        agnostic = load_and_normalize_image(agnostic_path, fine_h, fine_w, 'neg1_1')

        return self.run_from_preprocessed(
            cloth, cloth_mask, parse_agnostic, densepose, agnostic, occlusion
        )


# Singleton instance
_inference: HRVITONInference = None


def get_inference() -> HRVITONInference:
    global _inference
    if _inference is None:
        _inference = HRVITONInference()
    return _inference
