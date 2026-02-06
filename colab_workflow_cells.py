# ═══════════════════════════════════════════════════════════════
# COLAB WORKFLOW UTILITIES
# Copy these cells to the top of your Colab notebook
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# CELL 1: Clone Repo (run once at start)
# ───────────────────────────────────────────────────────────────
"""
!git clone https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo
%cd projectaria_gen2_depth_from_stereo
"""

# ───────────────────────────────────────────────────────────────
# CELL 2: Pull Latest Changes (run when code is updated)
# ───────────────────────────────────────────────────────────────
"""
import subprocess
import sys

def pull_latest():
    '''Pull latest code changes from GitHub'''
    try:
        result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Code updated successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error pulling changes:")
        print(e.stderr)
        return False

# Uncomment to auto-pull on cell run
# pull_latest()
"""

# ───────────────────────────────────────────────────────────────
# CELL 3: System Info (helps Claude debug remotely)
# ───────────────────────────────────────────────────────────────
"""
import torch
import sys
import subprocess

def print_system_info():
    '''Print system information for debugging'''
    print("=" * 60)
    print("SYSTEM INFORMATION (copy to Claude if errors occur)")
    print("=" * 60)

    # Python version
    print(f"Python: {sys.version}")

    # PyTorch & CUDA
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Git status
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            text=True
        ).strip()
        print(f"Git commit: {git_hash}")
    except:
        print("Git commit: unknown")

    print("=" * 60)

print_system_info()
"""

# ───────────────────────────────────────────────────────────────
# CELL 4: Error Reporter (wrap error-prone cells with this)
# ───────────────────────────────────────────────────────────────
"""
import traceback
import sys
from contextlib import contextmanager

@contextmanager
def error_reporter(step_name):
    '''
    Wrap code blocks to capture detailed error info

    Usage:
        with error_reporter("Loading model"):
            model = FoundationStereo(cfg)
            model.load_state_dict(...)
    '''
    print(f"🔄 {step_name}...")
    try:
        yield
        print(f"✅ {step_name} completed")
    except Exception as e:
        print("")
        print("=" * 60)
        print(f"❌ ERROR IN: {step_name}")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("")
        print("FULL TRACEBACK (copy everything below to Claude):")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        print(f"Step that failed: {step_name}")
        print("=" * 60)
        raise  # Re-raise to stop execution

# Example usage (remove the triple quotes to use):
'''
with error_reporter("Loading FoundationStereo model"):
    cfg = OmegaConf.load(cfg_path)
    model = FoundationStereo(cfg)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()
'''
"""

# ───────────────────────────────────────────────────────────────
# CELL 5: Upload Data Helper
# ───────────────────────────────────────────────────────────────
"""
from google.colab import drive
import os

def mount_drive_and_copy():
    '''Mount Google Drive and copy model/data files'''

    # Mount Drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')

    # Define paths (UPDATE THESE to match your Drive folder)
    DRIVE_BASE = '/content/drive/MyDrive/aria_gen2_data'

    files_to_copy = {
        'Model weights': {
            'src': f'{DRIVE_BASE}/23-51-11',
            'dst': './FoundationStereo/pretrained_models/23-51-11'
        },
        'VRS file': {
            'src': f'{DRIVE_BASE}/cook_0/video.vrs',
            'dst': './dataset/cook_0/video.vrs'
        },
        'Semi-dense points': {
            'src': f'{DRIVE_BASE}/cook_0/mps',
            'dst': './dataset/cook_0/mps'
        }
    }

    # Copy files
    for name, paths in files_to_copy.items():
        print(f"\\nCopying {name}...")
        os.makedirs(os.path.dirname(paths['dst']), exist_ok=True)

        if os.path.isdir(paths['src']):
            !cp -r "{paths['src']}" "{paths['dst']}"
        else:
            !cp "{paths['src']}" "{paths['dst']}"

        if os.path.exists(paths['dst']):
            print(f"✅ {name} copied successfully")
        else:
            print(f"❌ {name} copy failed")

    print("\\n" + "="*60)
    print("Files ready. You can now run the notebook.")
    print("="*60)

# Uncomment to run:
# mount_drive_and_copy()
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║  COLAB WORKFLOW CELLS CREATED                                ║
╚══════════════════════════════════════════════════════════════╝

The cells above have been saved to: colab_workflow_cells.py

To use in Colab:
1. Copy each cell's content to a new cell at the top of the notebook
2. Run them in order as needed

Cells:
  1. Clone repo (run once)
  2. Pull latest changes (run when code updated)
  3. Print system info (helps Claude debug)
  4. Error reporter wrapper (use around error-prone code)
  5. Upload data from Google Drive
""")
