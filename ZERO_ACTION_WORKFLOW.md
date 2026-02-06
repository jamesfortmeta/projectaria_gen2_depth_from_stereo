# Zero-Manual-Action Workflow

## Your Total Manual Actions: 2

1. **One-time setup** (2 min): Add GitHub token to Colab
2. **Start** (1 click): Run self-healing script in Colab

Then walk away. Come back to results.

---

## One-Time Setup (2 minutes)

### Step 1: Create GitHub Token

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: `Colab Auto-Runner`
4. Expiration: 90 days
5. Scopes: Check **`repo`** (all sub-options)
6. Click "Generate token"
7. **Copy the token** (starts with `ghp_...`)

### Step 2: Add Token to Colab

1. Open Colab: https://colab.research.google.com
2. Upload `depth_from_stereo.ipynb`
3. Click 🔑 (Key icon) in left sidebar → "Secrets"
4. Click "+ Add new secret"
5. Name: `GITHUB_TOKEN`
6. Value: Paste your token
7. Toggle "Notebook access" ON

**That's it. Never do this again.**

---

## Running (Once per session)

### In Colab - Add Setup Cell:

```python
# SETUP - Run once
!git clone https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo
%cd projectaria_gen2_depth_from_stereo

!pip install -q projectaria-tools[all]==2.0.0 omegaconf flash-attn

from google.colab import drive, userdata
import shutil
from pathlib import Path

drive.mount('/content/drive')

# Get GitHub token from secrets
import os
os.environ['GITHUB_TOKEN'] = userdata.get('GITHUB_TOKEN')

# Copy data
Path('./FoundationStereo/pretrained_models').mkdir(parents=True, exist_ok=True)
Path('./dataset').mkdir(exist_ok=True)

shutil.copytree('/content/drive/MyDrive/aria_gen2_data/23-51-11',
                './FoundationStereo/pretrained_models/23-51-11', dirs_exist_ok=True)
shutil.copytree('/content/drive/MyDrive/aria_gen2_data/cook_0',
                './dataset/cook_0', dirs_exist_ok=True)

print("✅ Setup complete")
```

### Run Self-Healing Script:

```python
# START SELF-HEALING LOOP - Run and walk away
!python self_healing_runner.py
```

**Now walk away.** Colab will:
- Run the pipeline
- If error → commit to GitHub
- Wait for Claude's fix
- Auto-pull and retry
- Repeat until success

---

## On Claude's Side (Automated)

While you're away, I run locally:

```bash
./watch_for_errors.sh
```

This script:
- Polls GitHub every 15s
- Detects new errors from Colab
- Shows me the error
- I analyze and fix
- I commit and push
- Colab auto-detects and retries

---

## The Full Flow

```
[You start Colab] → Walk away
       ↓
[Colab runs] → Error
       ↓
[Commits to GitHub] ← I'm watching
       ↓
[I see error] → Analyze → Fix → Push
       ↓
[Colab detects fix] → Pulls → Retries
       ↓
[Still error?] → Loop continues
       ↓
[Success!] → Saves results
       ↓
[You come back] → Download results
```

---

## Timeline Example

| Time | What Happens | Who Acts |
|------|--------------|----------|
| 0:00 | You start script | You |
| 0:00 | Script runs, hits error | Colab |
| 0:01 | Error committed to GitHub | Colab |
| 0:01 | Colab waits for fix | Colab |
| 0:01 | I detect error | Claude (me) |
| 0:02 | I analyze and fix | Claude |
| 0:03 | I push fix to GitHub | Claude |
| 0:03 | Colab detects update | Colab |
| 0:04 | Colab pulls and retries | Colab |
| 0:05 | New error appears | Colab |
| 0:06 | Commits error #2 | Colab |
| ... | Loop continues | Automated |
| 0:25 | Success! | Colab |
| ∞ | You come back anytime | You |

**Your screen time: 0 minutes**

---

## What You See When You Return

If successful:
```
✅ SUCCESS!
Results saved to: outputs/
  - disparity.npy
  - depth.npy
```

If failed (after 10 attempts):
```
❌ Failed after 10 attempts
[Shows last error]
```

---

## Manual Actions Required

| Action | When | Frequency |
|--------|------|-----------|
| Create GitHub token | Setup | Once per 90 days |
| Add token to Colab | Setup | Once per 90 days |
| Start script | Each run | Once |
| **Total while running** | - | **ZERO** |

---

## Comparison to Old Workflow

| Workflow | Manual Actions Per Error |
|----------|-------------------------|
| Old (copy-paste) | 1 copy + 1 paste + 1 git pull = 3 |
| **New (automated)** | **0** |

With 5 errors to fix:
- Old: 15 manual actions
- **New: 0 manual actions**

---

## Ready?

1. Get GitHub token (2 min)
2. Add to Colab secrets (1 min)
3. Run setup cell + self-healing script
4. **Walk away**
5. Come back to results

That's it.
