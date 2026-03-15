"""
Post-processing between Stage 1 (tocg) and Stage 2 (generator).
All operations use NumPy/OpenCV — no PyTorch or kornia needed.
"""
import numpy as np
import cv2


# Label remapping: 13-class parse → 7-class parse
# Matches the mapping in test_generator.py
LABEL_MAP = {
    0: [0],                       # background
    1: [2, 4, 7, 8, 9, 10, 11],  # paste (body parts to keep)
    2: [3],                       # upper (clothing region)
    3: [1],                       # hair
    4: [5],                       # left_arm
    5: [6],                       # right_arm
    6: [12],                      # noise
}


def gaussian_blur_segmap(segmap: np.ndarray, kernel_size: int = 15, sigma: float = 3.0) -> np.ndarray:
    """
    Apply Gaussian blur to the segmentation map.
    Replaces kornia/torchgeometry GaussianBlur((15,15),(3,3)).

    Args:
        segmap: [B, 13, H, W] float32 array (at fine resolution after upsampling)
    Returns:
        blurred: [B, 13, H, W] float32 array
    """
    B, C, H, W = segmap.shape
    blurred = np.zeros_like(segmap)
    for b in range(B):
        for c in range(C):
            blurred[b, c] = cv2.GaussianBlur(segmap[b, c], (kernel_size, kernel_size), sigma)
    return blurred


def segmap_to_parse7(segmap_blurred: np.ndarray, fine_height: int, fine_width: int) -> np.ndarray:
    """
    Convert 13-channel segmap to 7-channel parse map.

    Args:
        segmap_blurred: [B, 13, H, W] float32 (blurred, at fine resolution)
        fine_height, fine_width: target resolution
    Returns:
        parse7: [B, 7, fine_height, fine_width] float32
    """
    B = segmap_blurred.shape[0]

    # argmax to get class labels
    fake_parse = segmap_blurred.argmax(axis=1)  # [B, H, W]

    # One-hot encode to 13 channels
    old_parse = np.zeros((B, 13, fine_height, fine_width), dtype=np.float32)
    for b in range(B):
        for c in range(13):
            old_parse[b, c] = (fake_parse[b] == c).astype(np.float32)

    # Remap to 7 channels
    parse7 = np.zeros((B, 7, fine_height, fine_width), dtype=np.float32)
    for new_label, old_labels in LABEL_MAP.items():
        for old_label in old_labels:
            parse7[:, new_label] += old_parse[:, old_label]

    return parse7


def apply_clothmask_composition(fake_segmap: np.ndarray, warped_clothmask: np.ndarray,
                                  mode: str = 'warp_grad') -> np.ndarray:
    """
    Apply cloth mask composition to the segmentation map.
    Matches test_generator.py logic.
    """
    if mode == 'no_composition':
        return fake_segmap

    cloth_mask = np.ones_like(fake_segmap)
    if mode == 'warp_grad':
        cloth_mask[:, 3:4, :, :] = warped_clothmask
    elif mode == 'detach':
        warped_cm_onehot = (warped_clothmask > 0.5).astype(np.float32)
        cloth_mask[:, 3:4, :, :] = warped_cm_onehot

    return fake_segmap * cloth_mask


def remove_overlap(seg_softmax: np.ndarray, warped_cm: np.ndarray) -> np.ndarray:
    """
    Remove cloth mask overlap with non-clothing body regions.
    Matches test_generator.py remove_overlap().

    Args:
        seg_softmax: [B, 13, H, W] softmax of blurred segmap at fine res
        warped_cm: [B, 1, H, W] warped cloth mask at fine res
    """
    # Subtract regions: labels 1,2 (paste parts) and 5+ (arms, noise)
    overlap = np.concatenate([seg_softmax[:, 1:3, :, :], seg_softmax[:, 5:, :, :]], axis=1)
    overlap_sum = overlap.sum(axis=1, keepdims=True)
    warped_cm = warped_cm - overlap_sum * warped_cm
    return warped_cm


def upsample_flow_and_warp(flow: np.ndarray, cloth: np.ndarray, cloth_mask: np.ndarray,
                            fine_height: int, fine_width: int,
                            low_height: int = 256, low_width: int = 192) -> tuple:
    """
    Upsample optical flow from low res to full res and warp cloth + mask.
    Replaces F.grid_sample with cv2.remap.

    Args:
        flow: [B, low_H, low_W, 2] optical flow from tocg (at ~128x96 or last scale)
        cloth: [B, 3, fine_H, fine_W] full-resolution cloth image [-1,1]
        cloth_mask: [B, 1, fine_H, fine_W] full-resolution cloth mask
        fine_height, fine_width: target resolution (1024, 768)
    Returns:
        warped_cloth: [B, 3, fine_H, fine_W]
        warped_clothmask: [B, 1, fine_H, fine_W]
    """
    B = flow.shape[0]
    warped_cloths = []
    warped_masks = []

    # Flow normalization constants (from test_generator.py: 96 and 128)
    # These come from low_width/2=96 and low_height/2=128
    flow_norm_w = (low_width / 2 - 1.0) / 2.0
    flow_norm_h = (low_height / 2 - 1.0) / 2.0

    for b in range(B):
        # Upsample flow: [low_H, low_W, 2] -> [fine_H, fine_W, 2]
        # Matches reference: F.interpolate(flow, size=(iH, iW), mode='bilinear')
        flow_b = flow[b]  # [low_H, low_W, 2]
        flow_x = cv2.resize(flow_b[:, :, 0], (fine_width, fine_height), interpolation=cv2.INTER_LINEAR)
        flow_y = cv2.resize(flow_b[:, :, 1], (fine_width, fine_height), interpolation=cv2.INTER_LINEAR)

        # Normalize flow (matches reference: / ((96-1)/2), / ((128-1)/2))
        flow_x_norm = flow_x / flow_norm_w
        flow_y_norm = flow_y / flow_norm_h

        # Create sampling grid: base grid [-1,1] + flow
        grid_x = np.linspace(-1.0, 1.0, fine_width).reshape(1, -1).repeat(fine_height, axis=0)
        grid_y = np.linspace(-1.0, 1.0, fine_height).reshape(-1, 1).repeat(fine_width, axis=1)

        sample_x = grid_x + flow_x_norm
        sample_y = grid_y + flow_y_norm

        # Convert from [-1,1] to pixel coordinates for cv2.remap
        map_x = ((sample_x + 1) / 2 * (fine_width - 1)).astype(np.float32)
        map_y = ((sample_y + 1) / 2 * (fine_height - 1)).astype(np.float32)

        # Warp cloth (3 channels) — bilinear to match F.grid_sample default
        cloth_b = cloth[b].transpose(1, 2, 0)  # [H, W, 3]
        warped = cv2.remap(cloth_b, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_cloths.append(warped.transpose(2, 0, 1))  # [3, H, W]

        # Warp mask (1 channel) — keep linear for clean edges
        mask_b = cloth_mask[b, 0]  # [H, W]
        warped_m = cv2.remap(mask_b, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_masks.append(warped_m[np.newaxis])  # [1, H, W]

    return np.stack(warped_cloths), np.stack(warped_masks)


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def match_cloth_color(
    output_img: np.ndarray,
    reference_cloth: np.ndarray,
    cloth_mask: np.ndarray,
    warped_clothmask: np.ndarray,
    strength: float = 0.7,
) -> np.ndarray:
    """Match clothing region color in output to reference garment via histogram matching.

    Args:
        output_img: [H, W, 3] uint8 — generator output image
        reference_cloth: [1, 3, H, W] float32 [-1,1] — original cloth tensor
        cloth_mask: [1, 1, H, W] float32 {0,1} — original flat cloth mask
        warped_clothmask: [1, 1, H, W] float32 — warped cloth mask (clothing region in output)
        strength: blend factor (0=no correction, 1=full correction)
    Returns:
        corrected: [H, W, 3] uint8
    """
    # Convert reference cloth to uint8
    ref_img = ((reference_cloth[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0)
    ref_img = np.clip(ref_img, 0, 255).astype(np.uint8)

    # Get masks as 2D boolean
    ref_mask = cloth_mask[0, 0] > 0.5  # [H, W]
    out_mask = warped_clothmask[0, 0] > 0.3  # [H, W] slightly lower threshold for soft edges

    # Skip if insufficient pixels for stable histogram
    ref_count = int(ref_mask.sum())
    out_count = int(out_mask.sum())
    if ref_count < 100 or out_count < 100:
        return output_img

    # Extract pixels in clothing regions
    ref_pixels = ref_img[ref_mask]  # [N, 3]
    out_pixels = output_img[out_mask]  # [M, 3]

    # Build per-channel CDF lookup table
    corrected = output_img.copy()
    for c in range(3):
        ref_hist, _ = np.histogram(ref_pixels[:, c], bins=256, range=(0, 256))
        out_hist, _ = np.histogram(out_pixels[:, c], bins=256, range=(0, 256))

        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1] + 1e-8
        out_cdf = np.cumsum(out_hist).astype(np.float64)
        out_cdf /= out_cdf[-1] + 1e-8

        # Map output CDF → reference CDF
        lut = np.interp(out_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
        corrected[..., c] = lut[output_img[..., c]]

    # Feather the mask for smooth blending at boundaries
    blend_mask = warped_clothmask[0, 0].astype(np.float32)
    blend_mask = cv2.GaussianBlur(blend_mask, (7, 7), 2.0)
    blend_mask = (blend_mask * strength)[..., np.newaxis]  # [H, W, 1]

    # Blend: corrected in clothing region, original elsewhere
    result = (corrected.astype(np.float32) * blend_mask +
              output_img.astype(np.float32) * (1.0 - blend_mask))
    return np.clip(result, 0, 255).astype(np.uint8)
