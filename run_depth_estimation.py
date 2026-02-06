#!/usr/bin/env python3
"""
FoundationStereo Depth Estimation - Automated Runner
Run this script on Colab to execute the full pipeline with structured error logging.
"""

import sys
import traceback
import json
from datetime import datetime
from pathlib import Path

# Error log file
ERROR_LOG = "error_log.txt"
SUCCESS_LOG = "run_log.txt"

def log_error(step, error, tb):
    """Log error in a structured format for easy copy-paste to Claude"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    error_report = f"""
{'='*70}
ERROR REPORT - Copy everything below to Claude
{'='*70}
Timestamp: {timestamp}
Step Failed: {step}
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
    error_report += f"Python: {sys.version}\n"
    error_report += f"PyTorch: {torch.__version__}\n"
    error_report += f"CUDA available: {torch.cuda.is_available()}\n"
    if torch.cuda.is_available():
        error_report += f"CUDA version: {torch.version.cuda}\n"
        error_report += f"GPU: {torch.cuda.get_device_name(0)}\n"
        error_report += f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"

    error_report += f"{'='*70}\n"

    # Write to file
    with open(ERROR_LOG, 'w') as f:
        f.write(error_report)

    print(error_report)
    print(f"\n📄 Full error saved to: {ERROR_LOG}")
    print("👆 Copy the content above and paste to Claude\n")

def log_progress(step, status="started"):
    """Log progress"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = "🔄" if status == "started" else "✅"
    print(f"{emoji} [{timestamp}] {step}")

    with open(SUCCESS_LOG, 'a') as f:
        f.write(f"[{timestamp}] {step} - {status}\n")

def run_pipeline():
    """Run the complete depth estimation pipeline"""

    print("\n" + "="*70)
    print("FOUNDATIONSTEREO DEPTH ESTIMATION - AUTOMATED RUN")
    print("="*70 + "\n")

    # ==================== STEP 1: Setup ====================
    try:
        log_progress("Step 1: Environment setup")

        import os
        import time
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        import torch
        from projectaria_tools.core import data_provider, calibration
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId
        from projectaria_tools.core.sophus import SE3, SO3
        from projectaria_tools.core.image import InterpolationMethod
        import rerun as rr

        log_progress("Step 1: Environment setup", "completed")

    except Exception as e:
        log_error("Step 1: Environment setup", e, traceback.format_exc())
        return False

    # ==================== STEP 2: Load FoundationStereo ====================
    try:
        log_progress("Step 2: Loading FoundationStereo")

        # Find FoundationStereo
        FOUNDATION_STEREO_PATH = Path.cwd() / 'FoundationStereo'
        if not FOUNDATION_STEREO_PATH.exists():
            raise FileNotFoundError(f"FoundationStereo not found at {FOUNDATION_STEREO_PATH}")

        sys.path.insert(0, str(FOUNDATION_STEREO_PATH))

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

        log_progress("Step 2: Loading FoundationStereo", "completed")

    except Exception as e:
        log_error("Step 2: Loading FoundationStereo", e, traceback.format_exc())
        return False

    # ==================== STEP 3: Configuration ====================
    try:
        log_progress("Step 3: Configuration")

        VRS_FILE_PATH = "./dataset/cook_0/video.vrs"
        FOUNDATION_STEREO_CKPT = "./FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"

        # Verify files exist
        if not Path(VRS_FILE_PATH).exists():
            raise FileNotFoundError(f"VRS file not found: {VRS_FILE_PATH}")
        if not Path(FOUNDATION_STEREO_CKPT).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {FOUNDATION_STEREO_CKPT}")

        FRAME_INDEX = 100
        OUTPUT_WIDTH = 512
        OUTPUT_HEIGHT = 512
        FOCAL_SCALE = 1.25
        VALID_ITERS = 32
        MIN_DISPARITY = 1.0
        MAX_DEPTH = 20.0

        print(f"  VRS: {VRS_FILE_PATH}")
        print(f"  Model: {FOUNDATION_STEREO_CKPT}")
        print(f"  Output: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")

        log_progress("Step 3: Configuration", "completed")

    except Exception as e:
        log_error("Step 3: Configuration", e, traceback.format_exc())
        return False

    # ==================== STEP 4: Load VRS ====================
    try:
        log_progress("Step 4: Loading VRS data")

        vrs_data_provider = data_provider.create_vrs_data_provider(VRS_FILE_PATH)
        assert vrs_data_provider is not None, "Failed to load VRS file"

        device_calib = vrs_data_provider.get_device_calibration()

        left_stream_id = vrs_data_provider.get_stream_id_from_label("slam-front-left")
        right_stream_id = vrs_data_provider.get_stream_id_from_label("slam-front-right")

        assert left_stream_id is not None, "Could not find slam-front-left stream"
        assert right_stream_id is not None, "Could not find slam-front-right stream"

        left_num_frames = vrs_data_provider.get_num_data(left_stream_id)
        right_num_frames = vrs_data_provider.get_num_data(right_stream_id)

        print(f"  Left: {left_num_frames} frames")
        print(f"  Right: {right_num_frames} frames")

        log_progress("Step 4: Loading VRS data", "completed")

    except Exception as e:
        log_error("Step 4: Loading VRS data", e, traceback.format_exc())
        return False

    # ==================== STEP 5: Get Calibration ====================
    try:
        log_progress("Step 5: Camera calibration")

        left_calib = device_calib.get_camera_calib("slam-front-left")
        right_calib = device_calib.get_camera_calib("slam-front-right")

        T_left_cam_device = left_calib.get_transform_device_camera().inverse()
        T_right_cam_device = right_calib.get_transform_device_camera().inverse()

        baseline = compute_stereo_baseline(T_left_cam_device, T_right_cam_device)

        print(f"  Baseline: {baseline*1000:.1f} mm")

        log_progress("Step 5: Camera calibration", "completed")

    except Exception as e:
        log_error("Step 5: Camera calibration", e, traceback.format_exc())
        return False

    # ==================== STEP 6: Load Frame ====================
    try:
        log_progress(f"Step 6: Loading frame {FRAME_INDEX}")

        left_data, left_record = vrs_data_provider.get_image_data_by_index(left_stream_id, FRAME_INDEX)
        right_data, right_record = vrs_data_provider.get_image_data_by_index(right_stream_id, FRAME_INDEX)

        left_image = left_data.to_numpy_array()
        right_image = right_data.to_numpy_array()
        timestamp_ns = left_record.capture_timestamp_ns

        print(f"  Shape: {left_image.shape}")

        log_progress(f"Step 6: Loading frame {FRAME_INDEX}", "completed")

    except Exception as e:
        log_error(f"Step 6: Loading frame {FRAME_INDEX}", e, traceback.format_exc())
        return False

    # ==================== STEP 7: Rectification ====================
    try:
        log_progress("Step 7: Stereo rectification")

        left_linear = fisheye_to_linear_calib(
            left_calib, focal_scale=FOCAL_SCALE,
            output_width=OUTPUT_WIDTH, output_height=OUTPUT_HEIGHT
        )
        right_linear = fisheye_to_linear_calib(
            right_calib, focal_scale=FOCAL_SCALE,
            output_width=OUTPUT_WIDTH, output_height=OUTPUT_HEIGHT
        )

        Rl_n, Rr_n = create_scanline_rectified_cameras(T_left_cam_device, T_right_cam_device)

        start_time = time.time()
        left_rectified, right_rectified = rectify_stereo_pair(
            left_image, right_image, left_calib, right_calib,
            left_linear, right_linear, Rl_n, Rr_n,
            interpolation=InterpolationMethod.BILINEAR
        )
        rectify_time = (time.time() - start_time) * 1000

        print(f"  Completed in {rectify_time:.1f} ms")

        log_progress("Step 7: Stereo rectification", "completed")

    except Exception as e:
        log_error("Step 7: Stereo rectification", e, traceback.format_exc())
        return False

    # ==================== STEP 8: Load Model ====================
    try:
        log_progress("Step 8: Loading FoundationStereo model")

        cfg_path = Path(FOUNDATION_STEREO_CKPT).parent / 'cfg.yaml'
        cfg = OmegaConf.load(cfg_path)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        cfg.valid_iters = VALID_ITERS

        model = FoundationStereo(cfg)
        ckpt = torch.load(FOUNDATION_STEREO_CKPT, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        model = model.cuda().eval()

        print(f"  Model loaded (epoch {ckpt['epoch']})")

        log_progress("Step 8: Loading FoundationStereo model", "completed")

    except Exception as e:
        log_error("Step 8: Loading FoundationStereo model", e, traceback.format_exc())
        return False

    # ==================== STEP 9: Inference ====================
    try:
        log_progress("Step 9: Running inference")

        # Prepare input
        if len(left_rectified.shape) == 2:
            left_rgb = np.stack([left_rectified] * 3, axis=-1)
            right_rgb = np.stack([right_rectified] * 3, axis=-1)
        else:
            left_rgb = left_rectified
            right_rgb = right_rectified

        left_tensor = torch.from_numpy(left_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)
        right_tensor = torch.from_numpy(right_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)

        padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)

        # Run model
        start_time = time.time()
        with torch.cuda.amp.autocast(True):
            with torch.no_grad():
                disparity = model.forward(left_padded, right_padded, iters=cfg.valid_iters, test_mode=True)

        inference_time = (time.time() - start_time) * 1000
        disparity = padder.unpad(disparity.float())
        disparity_map = disparity.cpu().numpy().squeeze()

        print(f"  Inference: {inference_time:.1f} ms")
        print(f"  Disparity range: [{disparity_map.min():.2f}, {disparity_map.max():.2f}] px")

        log_progress("Step 9: Running inference", "completed")

    except Exception as e:
        log_error("Step 9: Running inference", e, traceback.format_exc())
        return False

    # ==================== STEP 10: Depth Conversion ====================
    try:
        log_progress("Step 10: Converting to depth")

        focal_length = left_linear.get_projection_params()[0]
        depth_map = disparity_to_depth(
            disparity_map, baseline=baseline, focal_length=focal_length,
            min_disparity=MIN_DISPARITY, max_depth=MAX_DEPTH
        )

        valid_depth = depth_map[depth_map > 0]
        print(f"  Valid pixels: {len(valid_depth)/depth_map.size*100:.1f}%")
        print(f"  Depth range: [{valid_depth.min():.2f}, {valid_depth.max():.2f}] m")
        print(f"  Mean depth: {valid_depth.mean():.2f} m")

        log_progress("Step 10: Converting to depth", "completed")

    except Exception as e:
        log_error("Step 10: Converting to depth", e, traceback.format_exc())
        return False

    # ==================== STEP 11: Save Results ====================
    try:
        log_progress("Step 11: Saving results")

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Save disparity
        np.save(output_dir / "disparity.npy", disparity_map)

        # Save depth
        np.save(output_dir / "depth.npy", depth_map)

        # Save visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(left_rectified, cmap='gray')
        axes[0].set_title('Left Rectified')
        axes[0].axis('off')
        axes[1].imshow(disparity_map, cmap='magma')
        axes[1].set_title('Disparity')
        axes[1].axis('off')
        axes[2].imshow(depth_map, cmap='turbo', vmin=0, vmax=10)
        axes[2].set_title('Depth (m)')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "results.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved to: {output_dir}/")

        log_progress("Step 11: Saving results", "completed")

    except Exception as e:
        log_error("Step 11: Saving results", e, traceback.format_exc())
        return False

    # ==================== SUCCESS ====================
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Results saved to: {output_dir}/")
    print(f"  - disparity.npy")
    print(f"  - depth.npy")
    print(f"  - results.png")
    print("="*70 + "\n")

    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
