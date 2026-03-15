"""
Export SCHP (Self-Correction Human Parsing) model to ONNX.

Prerequisites:
  1. Clone SCHP repo:  git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git ../SCHP
  2. Download LIP checkpoint (20 classes) from HuggingFace:
     curl -L -o checkpoints/exp-schp-201908261155-lip.pth \
       "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth"

Usage:
  python export_schp.py
  # Produces: onnx_models/schp_lip.onnx
"""

import sys
import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---- Step 1: Mock the CUDA-dependent modules.functions BEFORE importing SCHP ----
# The real modules/functions.py tries to JIT-compile CUDA extensions at import time.
# We replace it with a pure-Python mock that provides the same constants and
# dummy callables (we only need eval-mode forward, handled via ABN.forward).

SCHP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SCHP')
sys.path.insert(0, SCHP_DIR)

# Create mock functions module
mock_functions = types.ModuleType('modules.functions')
mock_functions.ACT_RELU = "relu"
mock_functions.ACT_LEAKY_RELU = "leaky_relu"
mock_functions.ACT_ELU = "elu"
mock_functions.ACT_NONE = "none"

def _dummy_abn(*args, **kwargs):
    raise RuntimeError("inplace_abn should not be called — use ABN.forward instead")

mock_functions.inplace_abn = _dummy_abn
mock_functions.inplace_abn_sync = _dummy_abn

# Inject the mock before any SCHP imports
sys.modules['modules.functions'] = mock_functions

# ---- Step 2: Now import SCHP modules (they'll use our mock) ----
from modules.bn import ABN, InPlaceABN, InPlaceABNSync

# Patch InPlaceABN/Sync to use standard PyTorch ops (ABN.forward)
# In eval mode: batch_norm + optional activation — no CUDA needed
InPlaceABNSync.forward = ABN.forward
InPlaceABN.forward = ABN.forward
print("Patched InPlaceABN/Sync to use standard PyTorch batch_norm")


class SCHPWrapper(nn.Module):
    """Wrapper that returns only the fusion parsing result (not edge outputs)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Original model returns: [[parsing_result, fusion_result], [edge_result]]
        outputs = self.model(x)
        fusion_result = outputs[0][-1]  # [B, 20, H, W] logits
        return fusion_result


def main():
    import networks

    # Settings matching LIP dataset (20 CIHP classes, 473x473)
    num_classes = 20
    input_h, input_w = 473, 473

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'checkpoints', 'exp-schp-201908261155-lip.pth')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'onnx_models', 'schp_lip.onnx')

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Download with:")
        print('  curl -L -o checkpoints/exp-schp-201908261155-lip.pth \\')
        print('    "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth"')
        sys.exit(1)

    # Load model (pretrained=None skips ImageNet weight loading)
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    # Load checkpoint (strip 'module.' prefix from DataParallel state dict)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Wrap to return only fusion result
    wrapper = SCHPWrapper(model)
    wrapper.eval()

    # Test forward pass
    dummy_input = torch.randn(1, 3, input_h, input_w)
    with torch.no_grad():
        out = wrapper(dummy_input)
        print(f"Forward pass OK — output shape: {out.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=16,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
        },
    )
    print(f"Exported ONNX model to: {output_path}")

    # Verify ONNX model matches PyTorch output
    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {'input': dummy_input.numpy()})[0]
    pytorch_out = out.numpy()

    max_diff = np.max(np.abs(onnx_out - pytorch_out))
    print(f"Max difference PyTorch vs ONNX: {max_diff:.6f}")
    if max_diff < 0.01:
        print("ONNX export verified successfully!")
    else:
        print("WARNING: Large difference between PyTorch and ONNX outputs")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {file_size_mb:.1f} MB")


if __name__ == '__main__':
    main()
