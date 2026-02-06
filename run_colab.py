#!/usr/bin/env python3
"""
Colab Error Bridge - Enhanced runner with automatic error reporting
This script runs on Colab and saves errors in a format easy to retrieve locally.
"""

import sys
import traceback
import os
from datetime import datetime
from pathlib import Path

def save_error_for_download(step, error, tb):
    """Save error in formatted file ready for download"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    error_content = f"""COLAB ERROR REPORT
{'='*70}
Timestamp: {timestamp}
Failed Step: {step}
Error Type: {type(error).__name__}
Error Message: {str(error)}

FULL TRACEBACK:
{'-'*70}
{tb}
{'-'*70}

SYSTEM INFO:
{'-'*70}
"""

    # Add system info
    import torch
    error_content += f"Python: {sys.version}\n"
    error_content += f"PyTorch: {torch.__version__}\n"
    error_content += f"CUDA: {torch.cuda.is_available()}\n"
    if torch.cuda.is_available():
        error_content += f"CUDA version: {torch.version.cuda}\n"
        error_content += f"GPU: {torch.cuda.get_device_name(0)}\n"
        error_content += f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"

    error_content += f"{'='*70}\n"

    # Save to file
    error_file = "colab_error.txt"
    with open(error_file, 'w') as f:
        f.write(error_content)

    print("\n" + "!"*70)
    print("❌ ERROR OCCURRED")
    print("!"*70)
    print(error_content)
    print("!"*70)
    print(f"📁 Error saved to: {error_file}")
    print("!"*70)

    # Auto-download if in Colab
    try:
        from google.colab import files
        print("\n⬇️  DOWNLOADING ERROR LOG...")
        files.download(error_file)
        print("✅ Downloaded! Check your Downloads folder.")
        print("\n📋 NEXT STEPS:")
        print("   1. Find 'colab_error.txt' in your Downloads")
        print("   2. Tell Claude: 'read colab error'")
        print("   3. I'll analyze and fix the issue")
    except ImportError:
        # Not in Colab
        print(f"\n📋 Error saved to: {error_file}")
        print("Copy this file to share with Claude")

    return error_file

def log_progress(msg):
    """Print progress message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

def run_pipeline():
    """Execute the full pipeline with error handling"""

    print("\n" + "="*70)
    print("FOUNDATIONSTEREO DEPTH ESTIMATION")
    print("="*70 + "\n")

    try:
        # ===== STEP 1: Imports =====
        log_progress("Loading libraries...")
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        import time

        from projectaria_tools.core import data_provider
        from projectaria_tools.core.image import InterpolationMethod

        sys.path.insert(0, str(Path.cwd() / 'FoundationStereo'))
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from stereo_utils import (
            create_scanline_rectified_cameras,
            fisheye_to_linear_calib,
            rectify_stereo_pair,
            compute_stereo_baseline,
            disparity_to_depth,
            get_rectified_camera_transform,
        )

        log_progress("✅ Libraries loaded")

        # ===== STEP 2: Configuration =====
        log_progress("Setting up configuration...")

        VRS_FILE = "./dataset/cook_0/video.vrs"
        MODEL_CKPT = "./FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"

        if not Path(VRS_FILE).exists():
            raise FileNotFoundError(f"VRS not found: {VRS_FILE}")
        if not Path(MODEL_CKPT).exists():
            raise FileNotFoundError(f"Model not found: {MODEL_CKPT}")

        CONFIG = {
            'frame_index': 100,
            'output_width': 512,
            'output_height': 512,
            'focal_scale': 1.25,
            'valid_iters': 32,
            'min_disparity': 1.0,
            'max_depth': 20.0
        }

        log_progress("✅ Configuration set")

        # ===== STEP 3: Load VRS =====
        log_progress("Loading VRS file...")

        provider = data_provider.create_vrs_data_provider(VRS_FILE)
        device_calib = provider.get_device_calibration()

        left_stream = provider.get_stream_id_from_label("slam-front-left")
        right_stream = provider.get_stream_id_from_label("slam-front-right")

        log_progress(f"✅ VRS loaded ({provider.get_num_data(left_stream)} frames)")

        # ===== STEP 4: Get calibration =====
        log_progress("Extracting calibration...")

        left_calib = device_calib.get_camera_calib("slam-front-left")
        right_calib = device_calib.get_camera_calib("slam-front-right")

        T_left = left_calib.get_transform_device_camera().inverse()
        T_right = right_calib.get_transform_device_camera().inverse()

        baseline = compute_stereo_baseline(T_left, T_right)

        log_progress(f"✅ Baseline: {baseline*1000:.1f} mm")

        # ===== STEP 5: Load frame =====
        log_progress(f"Loading frame {CONFIG['frame_index']}...")

        left_data, _ = provider.get_image_data_by_index(left_stream, CONFIG['frame_index'])
        right_data, _ = provider.get_image_data_by_index(right_stream, CONFIG['frame_index'])

        left_img = left_data.to_numpy_array()
        right_img = right_data.to_numpy_array()

        log_progress(f"✅ Frame loaded: {left_img.shape}")

        # ===== STEP 6: Rectification =====
        log_progress("Rectifying stereo pair...")

        left_linear = fisheye_to_linear_calib(
            left_calib, CONFIG['focal_scale'],
            CONFIG['output_width'], CONFIG['output_height']
        )
        right_linear = fisheye_to_linear_calib(
            right_calib, CONFIG['focal_scale'],
            CONFIG['output_width'], CONFIG['output_height']
        )

        Rl, Rr = create_scanline_rectified_cameras(T_left, T_right)

        left_rect, right_rect = rectify_stereo_pair(
            left_img, right_img,
            left_calib, right_calib,
            left_linear, right_linear,
            Rl, Rr,
            InterpolationMethod.BILINEAR
        )

        log_progress("✅ Rectification complete")

        # ===== STEP 7: Load model =====
        log_progress("Loading FoundationStereo model...")

        cfg = OmegaConf.load(Path(MODEL_CKPT).parent / 'cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        cfg.valid_iters = CONFIG['valid_iters']

        model = FoundationStereo(cfg)
        ckpt = torch.load(MODEL_CKPT, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        model = model.cuda().eval()

        log_progress(f"✅ Model loaded (epoch {ckpt['epoch']})")

        # ===== STEP 8: Inference =====
        log_progress("Running stereo matching...")

        # Prepare input
        if len(left_rect.shape) == 2:
            left_rgb = np.stack([left_rect] * 3, axis=-1)
            right_rgb = np.stack([right_rect] * 3, axis=-1)
        else:
            left_rgb = left_rect
            right_rgb = right_rect

        left_t = torch.from_numpy(left_rgb).float().cuda().permute(2,0,1).unsqueeze(0)
        right_t = torch.from_numpy(right_rgb).float().cuda().permute(2,0,1).unsqueeze(0)

        padder = InputPadder(left_t.shape, divis_by=32)
        left_p, right_p = padder.pad(left_t, right_t)

        # Inference
        t0 = time.time()
        with torch.cuda.amp.autocast(True):
            with torch.no_grad():
                disp = model.forward(left_p, right_p, iters=cfg.valid_iters, test_mode=True)

        elapsed = (time.time() - t0) * 1000

        disp = padder.unpad(disp.float())
        disparity = disp.cpu().numpy().squeeze()

        log_progress(f"✅ Inference: {elapsed:.1f}ms, range=[{disparity.min():.1f}, {disparity.max():.1f}]")

        # ===== STEP 9: Convert to depth =====
        log_progress("Converting to metric depth...")

        focal = left_linear.get_projection_params()[0]
        depth = disparity_to_depth(
            disparity, baseline, focal,
            CONFIG['min_disparity'], CONFIG['max_depth']
        )

        valid = depth[depth > 0]
        log_progress(f"✅ Depth: {len(valid)/depth.size*100:.1f}% valid, mean={valid.mean():.2f}m")

        # ===== STEP 10: Save results =====
        log_progress("Saving outputs...")

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        np.save(out_dir / "disparity.npy", disparity)
        np.save(out_dir / "depth.npy", depth)

        # Visualization
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].imshow(left_rect, cmap='gray')
        ax[0].set_title('Rectified')
        ax[0].axis('off')
        ax[1].imshow(disparity, cmap='magma')
        ax[1].set_title('Disparity')
        ax[1].axis('off')
        ax[2].imshow(depth, cmap='turbo', vmin=0, vmax=10)
        ax[2].set_title('Depth (m)')
        ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / "result.png", dpi=120)

        log_progress(f"✅ Saved to {out_dir}/")

        # ===== SUCCESS =====
        print("\n" + "="*70)
        print("✅ SUCCESS - Pipeline completed")
        print("="*70)
        print(f"Output: {out_dir}/")
        print("  • disparity.npy")
        print("  • depth.npy")
        print("  • result.png")
        print("="*70 + "\n")

        return True

    except Exception as e:
        # Extract step name from traceback
        tb_str = traceback.format_exc()
        step = "Unknown"
        for line in tb_str.split('\n'):
            if 'STEP' in line and '=====' in line:
                step = line.strip('#= ')
                break

        save_error_for_download(step, e, tb_str)
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
