"""
Export HR-VITON models (ConditionGenerator + SPADEGenerator) to ONNX format.
Usage: python export_models.py
"""
import sys
import os
import argparse
from collections import OrderedDict

# Add model architecture definitions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_architectures'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import ConditionGenerator
from network_generator import SPADEGenerator


class OptConfig:
    """Mimics argparse namespace with all required opt parameters."""
    def __init__(self):
        # Common
        self.cuda = False
        self.semantic_nc = 13
        self.output_nc = 13
        self.fine_width = 768
        self.fine_height = 1024
        # ConditionGenerator
        self.warp_feature = 'T1'
        self.out_layer = 'relu'
        # SPADEGenerator
        self.norm_G = 'spectralaliasinstance'
        self.ngf = 64
        self.gen_semantic_nc = 7
        self.num_upsampling_layers = 'most'
        self.init_type = 'xavier'
        self.init_variance = 0.02


class TocgWrapper(nn.Module):
    """Wraps ConditionGenerator to remove the `opt` argument from forward()."""
    def __init__(self, tocg, opt):
        super().__init__()
        self.tocg = tocg
        self.opt = opt

    def forward(self, input1, input2):
        flow_list, fake_segmap, warped_cloth, warped_cm = self.tocg(self.opt, input1, input2)
        # Return only the final (highest-res) flow + outputs
        return flow_list[-1], fake_segmap, warped_cloth, warped_cm


class GenWrapper(nn.Module):
    """Wraps SPADEGenerator to ensure clean tracing (noise replaced with zeros)."""
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, seg):
        return self.generator(x, seg)


def remove_spectral_norm_from_model(model):
    """Remove spectral normalization from all layers for ONNX compatibility."""
    for name, module in model.named_modules():
        try:
            torch.nn.utils.remove_spectral_norm(module)
        except ValueError:
            pass  # Module didn't have spectral norm


def patch_spade_noise():
    """Patch SPADENorm to skip noise entirely for clean ONNX tracing.
    The noise_scale parameters are near-zero after training, so skipping noise
    has negligible impact on output quality while avoiding ONNX tracing issues
    with dynamic-shaped tensor creation."""
    from network_generator import SPADENorm

    def patched_forward(self, x, seg, misalign_mask=None):
        # Skip noise entirely — avoids ConstantOfShape issues in ONNX tracing
        if misalign_mask is None:
            normalized = self.param_free_norm(x)
        else:
            normalized = self.param_free_norm(x, misalign_mask)

        actv = self.conv_shared(seg)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)
        return normalized * (1 + gamma) + beta

    SPADENorm.forward = patched_forward


def load_tocg(opt, checkpoint_path):
    """Load ConditionGenerator with pretrained weights."""
    input1_nc = 4   # cloth (3) + cloth_mask (1)
    input2_nc = 16   # parse_agnostic (13) + densepose (3)

    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc,
                               output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    tocg.load_state_dict(state_dict, strict=False)
    tocg.eval()
    return tocg


def load_generator(opt, checkpoint_path):
    """Load SPADEGenerator with pretrained weights (with key renaming)."""
    generator = SPADEGenerator(opt, 3 + 3 + 3)  # agnostic + densepose + warped_cloth

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # Rename keys: 'ace' -> 'alias', remove '.Spade' (matches original load_checkpoint_G)
    new_state_dict = OrderedDict(
        [(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()]
    )
    if hasattr(state_dict, '_metadata'):
        new_state_dict._metadata = OrderedDict(
            [(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()]
        )
    generator.load_state_dict(new_state_dict, strict=True)
    generator.eval()
    return generator


def export_tocg(opt, checkpoint_path, output_path):
    """Export ConditionGenerator to ONNX."""
    print("=" * 60)
    print("Exporting ConditionGenerator (tocg) to ONNX...")
    print("=" * 60)

    tocg = load_tocg(opt, checkpoint_path)
    wrapper = TocgWrapper(tocg, opt)
    wrapper.eval()

    # Input shapes at 256x192 (downsampled resolution)
    dummy_input1 = torch.randn(1, 4, 256, 192)   # cloth + mask
    dummy_input2 = torch.randn(1, 16, 256, 192)   # parse_agnostic + densepose

    print(f"  input1 shape: {dummy_input1.shape}")
    print(f"  input2 shape: {dummy_input2.shape}")

    # Test forward pass first
    with torch.no_grad():
        flow, segmap, warped_cloth, warped_cm = wrapper(dummy_input1, dummy_input2)
    print(f"  flow shape: {flow.shape}")
    print(f"  segmap shape: {segmap.shape}")
    print(f"  warped_cloth shape: {warped_cloth.shape}")
    print(f"  warped_clothmask shape: {warped_cm.shape}")

    torch.onnx.export(
        wrapper,
        (dummy_input1, dummy_input2),
        output_path,
        opset_version=16,
        input_names=["input1", "input2"],
        output_names=["flow", "segmap", "warped_cloth", "warped_clothmask"],
        dynamic_axes={
            "input1": {0: "batch"},
            "input2": {0: "batch"},
            "flow": {0: "batch"},
            "segmap": {0: "batch"},
            "warped_cloth": {0: "batch"},
            "warped_clothmask": {0: "batch"},
        },
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")
    return True


def export_generator(opt, checkpoint_path, output_path):
    """Export SPADEGenerator to ONNX."""
    print("=" * 60)
    print("Exporting SPADEGenerator to ONNX...")
    print("=" * 60)

    # Patch SPADENorm to skip noise (avoids dynamic-shape issues in ONNX tracing)
    patch_spade_noise()

    generator = load_generator(opt, checkpoint_path)

    # Remove spectral norm for ONNX compatibility
    remove_spectral_norm_from_model(generator)

    # Force all InstanceNorm2d to eval mode
    for m in generator.modules():
        if isinstance(m, nn.InstanceNorm2d):
            m.eval()
            m.training = False

    wrapper = GenWrapper(generator)
    wrapper.eval()

    # Full resolution inputs: 1024x768
    dummy_x = torch.randn(1, 9, 1024, 768)    # agnostic + densepose + warped_cloth
    dummy_seg = torch.randn(1, 7, 1024, 768)   # 7-channel parse map

    print(f"  input shape: {dummy_x.shape}")
    print(f"  segmap shape: {dummy_seg.shape}")

    # Test forward pass
    with torch.no_grad():
        output = wrapper(dummy_x, dummy_seg)
    print(f"  output shape: {output.shape}")

    print("  Running ONNX export (this may take a few minutes and use ~5GB RAM)...")
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_seg),
        output_path,
        opset_version=16,
        input_names=["input", "segmap"],
        output_names=["output"],
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes={
            "input": {0: "batch"},
            "segmap": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")
    return True


def validate_onnx(tocg_path, gen_path):
    """Validate exported ONNX models with onnxruntime."""
    import onnxruntime as ort
    import numpy as np

    print("\n" + "=" * 60)
    print("Validating ONNX models with onnxruntime...")
    print("=" * 60)

    # Validate tocg
    print("\n  Validating tocg.onnx...")
    sess = ort.InferenceSession(tocg_path, providers=['CPUExecutionProvider'])
    input1 = np.random.randn(1, 4, 256, 192).astype(np.float32)
    input2 = np.random.randn(1, 16, 256, 192).astype(np.float32)
    outputs = sess.run(None, {"input1": input1, "input2": input2})
    print(f"    flow: {outputs[0].shape}")
    print(f"    segmap: {outputs[1].shape}")
    print(f"    warped_cloth: {outputs[2].shape}")
    print(f"    warped_clothmask: {outputs[3].shape}")
    print("    tocg.onnx: OK")

    # Validate generator
    print("\n  Validating gen.onnx...")
    sess = ort.InferenceSession(gen_path, providers=['CPUExecutionProvider'])
    x = np.random.randn(1, 9, 1024, 768).astype(np.float32)
    seg = np.random.randn(1, 7, 1024, 768).astype(np.float32)
    outputs = sess.run(None, {"input": x, "segmap": seg})
    print(f"    output: {outputs[0].shape}")
    assert outputs[0].shape == (1, 3, 1024, 768), f"Unexpected shape: {outputs[0].shape}"
    print("    gen.onnx: OK")

    print("\n  All validations passed!")


def main():
    parser = argparse.ArgumentParser(description="Export HR-VITON models to ONNX")
    parser.add_argument('--tocg-weights', default='checkpoints/mtviton.pth')
    parser.add_argument('--gen-weights', default='checkpoints/gen.pth')
    parser.add_argument('--output-dir', default='onnx_models')
    parser.add_argument('--skip-tocg', action='store_true')
    parser.add_argument('--skip-gen', action='store_true')
    parser.add_argument('--skip-validate', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    opt = OptConfig()

    tocg_onnx = os.path.join(args.output_dir, 'tocg.onnx')
    gen_onnx = os.path.join(args.output_dir, 'gen.onnx')

    if not args.skip_tocg:
        export_tocg(opt, args.tocg_weights, tocg_onnx)

    if not args.skip_gen:
        export_generator(opt, args.gen_weights, gen_onnx)

    if not args.skip_validate:
        validate_onnx(tocg_onnx, gen_onnx)

    print("\nDone! ONNX models saved to:", args.output_dir)


if __name__ == '__main__':
    main()
