"""
Image preprocessing utilities for the HR-VITON inference pipeline.
Handles loading and normalizing images to the expected tensor format.
"""
import numpy as np
from PIL import Image
import cv2


def load_and_normalize_image(path: str, height: int, width: int, normalize_range: str = 'neg1_1') -> np.ndarray:
    """
    Load an image and normalize to model input format.

    Args:
        path: Path to image file
        height, width: Target dimensions
        normalize_range: 'neg1_1' for [-1,1] or '0_1' for [0,1]
    Returns:
        [1, 3, H, W] float32 array
    """
    img = Image.open(path).convert('RGB').resize((width, height))
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]

    if normalize_range == 'neg1_1':
        arr = arr * 2.0 - 1.0  # [-1, 1]

    # HWC -> CHW -> BCHW
    return arr.transpose(2, 0, 1)[np.newaxis]


def load_mask(path: str, height: int, width: int) -> np.ndarray:
    """
    Load a binary mask image.

    Returns:
        [1, 1, H, W] float32 array with values in {0, 1}
    """
    img = Image.open(path).convert('L').resize((width, height))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return arr[np.newaxis, np.newaxis]


def load_parse_map(path: str, height: int, width: int, num_classes: int = 13) -> np.ndarray:
    """
    Load a semantic parse map and convert to one-hot encoding.

    Args:
        path: Path to parse map image (single channel, pixel values = class labels)
        height, width: Target dimensions
        num_classes: Number of semantic classes
    Returns:
        [1, num_classes, H, W] float32 one-hot array
    """
    img = Image.open(path).convert('L').resize((width, height), Image.NEAREST)
    arr = np.array(img, dtype=np.int64)

    one_hot = np.zeros((num_classes, height, width), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c] = (arr == c).astype(np.float32)

    return one_hot[np.newaxis]


def load_densepose(path: str, height: int, width: int) -> np.ndarray:
    """
    Load a DensePose visualization image.

    Returns:
        [1, 3, H, W] float32 array normalized to [-1, 1]
    """
    return load_and_normalize_image(path, height, width, 'neg1_1')


def downsample(arr: np.ndarray, height: int, width: int, mode: str = 'bilinear') -> np.ndarray:
    """
    Downsample a BCHW array to target size.

    Args:
        arr: [B, C, H, W] float32
        height, width: Target dimensions
        mode: 'bilinear' or 'nearest'
    """
    B, C, _, _ = arr.shape
    interp = cv2.INTER_LINEAR if mode == 'bilinear' else cv2.INTER_NEAREST
    result = np.zeros((B, C, height, width), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            result[b, c] = cv2.resize(arr[b, c], (width, height), interpolation=interp)
    return result


def upsample(arr: np.ndarray, height: int, width: int, mode: str = 'bilinear') -> np.ndarray:
    """Alias for downsample (same implementation, just semantically different)."""
    return downsample(arr, height, width, mode)
