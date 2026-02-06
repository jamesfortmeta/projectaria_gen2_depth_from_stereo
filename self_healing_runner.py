#!/usr/bin/env python3
"""
Self-Healing Colab Runner
Runs in a loop, auto-commits errors to GitHub, waits for Claude to fix, auto-retries.
User starts this once and walks away.
"""

import sys
import os
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
MAX_RETRIES = 10
POLL_INTERVAL = 30  # seconds between checking for fixes
ERROR_BRANCH = "colab-errors"
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')

def setup_git():
    """Configure git with token for pushing"""
    if not GITHUB_TOKEN:
        print("⚠️  GITHUB_TOKEN not set. Running without auto-commit.")
        return False

    # Configure git
    subprocess.run(['git', 'config', 'user.email', 'colab@auto.run'], capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'Colab Auto'], capture_output=True)

    # Set up authenticated remote
    repo_url = f"https://{GITHUB_TOKEN}@github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo.git"
    subprocess.run(['git', 'remote', 'set-url', 'origin', repo_url], capture_output=True)

    return True

def commit_error(error_msg, attempt):
    """Commit error to GitHub for Claude to see"""
    try:
        # Write error to file
        error_file = "COLAB_ERROR.txt"
        with open(error_file, 'w') as f:
            f.write(f"Attempt: {attempt}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*70 + "\n")
            f.write(error_msg)

        # Commit and push
        subprocess.run(['git', 'checkout', '-B', ERROR_BRANCH], capture_output=True, check=True)
        subprocess.run(['git', 'add', error_file], capture_output=True, check=True)
        subprocess.run(['git', 'commit', '-m', f'Colab error (attempt {attempt})'],
                      capture_output=True, check=True)
        subprocess.run(['git', 'push', '-f', 'origin', ERROR_BRANCH],
                      capture_output=True, check=True)

        print(f"✅ Error committed to GitHub branch: {ERROR_BRANCH}")
        print(f"🤖 Claude will see this and fix it automatically")
        return True
    except Exception as e:
        print(f"⚠️  Could not commit error: {e}")
        return False

def wait_for_fix():
    """Poll main branch for new commits (Claude's fixes)"""
    print(f"\n⏳ Waiting for Claude to fix the error...")
    print(f"   Checking every {POLL_INTERVAL}s for updates...")

    # Get current commit hash
    result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True)
    start_hash = result.stdout.strip()

    checks = 0
    while checks < 60:  # Max 30 minutes
        time.sleep(POLL_INTERVAL)
        checks += 1

        # Fetch latest
        subprocess.run(['git', 'fetch', 'origin', 'main'],
                      capture_output=True)

        # Check if main branch updated
        result = subprocess.run(['git', 'rev-parse', 'origin/main'],
                              capture_output=True, text=True, check=True)
        new_hash = result.stdout.strip()

        if new_hash != start_hash:
            print(f"\n✅ New commit detected! Claude pushed a fix.")

            # Switch back to main and pull
            subprocess.run(['git', 'checkout', 'main'], capture_output=True, check=True)
            subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True, check=True)

            return True

        print(f"   Check {checks}: No updates yet...", end='\r')

    print("\n⏰ Timeout waiting for fix (30 min)")
    return False

def run_pipeline():
    """Run the actual depth estimation pipeline"""
    print("\n" + "="*70)
    print("RUNNING PIPELINE")
    print("="*70 + "\n")

    # Import and run the actual pipeline
    sys.path.insert(0, str(Path.cwd() / 'FoundationStereo'))

    try:
        # All the pipeline code here
        import torch
        import numpy as np
        from pathlib import Path
        import time

        from projectaria_tools.core import data_provider
        from projectaria_tools.core.image import InterpolationMethod
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from stereo_utils import (
            create_scanline_rectified_cameras,
            fisheye_to_linear_calib,
            rectify_stereo_pair,
            compute_stereo_baseline,
            disparity_to_depth,
        )

        print("✓ Libraries loaded")

        # Config
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

        print("✓ Configuration")

        # Load VRS
        provider = data_provider.create_vrs_data_provider(VRS_FILE)
        device_calib = provider.get_device_calibration()
        left_stream = provider.get_stream_id_from_label("slam-front-left")
        right_stream = provider.get_stream_id_from_label("slam-front-right")

        print(f"✓ VRS loaded ({provider.get_num_data(left_stream)} frames)")

        # Calibration
        left_calib = device_calib.get_camera_calib("slam-front-left")
        right_calib = device_calib.get_camera_calib("slam-front-right")
        T_left = left_calib.get_transform_device_camera().inverse()
        T_right = right_calib.get_transform_device_camera().inverse()
        baseline = compute_stereo_baseline(T_left, T_right)

        print(f"✓ Baseline: {baseline*1000:.1f} mm")

        # Load frame
        left_data, _ = provider.get_image_data_by_index(left_stream, CONFIG['frame_index'])
        right_data, _ = provider.get_image_data_by_index(right_stream, CONFIG['frame_index'])
        left_img = left_data.to_numpy_array()
        right_img = right_data.to_numpy_array()

        print(f"✓ Frame {CONFIG['frame_index']}: {left_img.shape}")

        # Rectification
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

        print("✓ Rectification")

        # Load model
        cfg = OmegaConf.load(Path(MODEL_CKPT).parent / 'cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        cfg.valid_iters = CONFIG['valid_iters']

        model = FoundationStereo(cfg)
        ckpt = torch.load(MODEL_CKPT, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        model = model.cuda().eval()

        print(f"✓ Model loaded (epoch {ckpt['epoch']})")

        # Inference
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

        with torch.cuda.amp.autocast(True):
            with torch.no_grad():
                disp = model.forward(left_p, right_p, iters=cfg.valid_iters, test_mode=True)

        disp = padder.unpad(disp.float())
        disparity = disp.cpu().numpy().squeeze()

        print(f"✓ Inference: range=[{disparity.min():.1f}, {disparity.max():.1f}]")

        # Depth conversion
        focal = left_linear.get_projection_params()[0]
        depth = disparity_to_depth(
            disparity, baseline, focal,
            CONFIG['min_disparity'], CONFIG['max_depth']
        )

        valid = depth[depth > 0]
        print(f"✓ Depth: {len(valid)/depth.size*100:.1f}% valid, mean={valid.mean():.2f}m")

        # Save
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "disparity.npy", disparity)
        np.save(out_dir / "depth.npy", depth)

        print(f"✓ Saved to {out_dir}/")

        return True, None

    except Exception as e:
        error_msg = f"""
Error Type: {type(e).__name__}
Error: {str(e)}

System Info:
- Python: {sys.version.split()[0]}
- PyTorch: {torch.__version__}
- CUDA: {torch.cuda.is_available()}
"""
        if torch.cuda.is_available():
            error_msg += f"- GPU: {torch.cuda.get_device_name(0)}\n"
            error_msg += f"- VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"

        error_msg += f"\nTraceback:\n{traceback.format_exc()}"

        return False, error_msg

def main():
    """Main self-healing loop"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "SELF-HEALING COLAB RUNNER" + " "*28 + "║")
    print("╚" + "="*68 + "╝\n")

    # Setup
    has_git = setup_git()
    if not has_git:
        print("⚠️  Running without GitHub integration.")
        print("   Set GITHUB_TOKEN in Colab secrets for auto-healing.")
        print("   Continuing in single-run mode...\n")

    # Run loop
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*70}")
        print(f"ATTEMPT {attempt}/{MAX_RETRIES}")
        print('='*70)

        success, error_msg = run_pipeline()

        if success:
            print("\n" + "╔" + "="*68 + "╗")
            print("║" + " "*25 + "SUCCESS!" + " "*36 + "║")
            print("╚" + "="*68 + "╝\n")
            print("Results saved to: outputs/")
            return 0

        # Handle error
        print(f"\n❌ Error on attempt {attempt}")

        if has_git and attempt < MAX_RETRIES:
            # Commit error for Claude to see
            if commit_error(error_msg, attempt):
                # Wait for Claude to fix
                if wait_for_fix():
                    print(f"\n🔄 Retrying with fixed code...")
                    continue
                else:
                    print("\n⏰ Timeout - stopping")
                    break
            else:
                print("\n⚠️  Could not commit error - stopping")
                break
        else:
            # No git or last attempt - print error
            print("\n" + "="*70)
            print("ERROR DETAILS:")
            print("="*70)
            print(error_msg)
            print("="*70)
            break

    print("\n❌ Failed after", attempt, "attempts")
    return 1

if __name__ == "__main__":
    sys.exit(main())
