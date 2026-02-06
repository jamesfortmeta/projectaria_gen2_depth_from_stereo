#!/bin/bash
# Claude Auto-Fixer - Watches for Colab errors and auto-fixes them
# Run this locally while Colab self-healing runner is running

REPO_DIR="/Users/jamesfort/Documents/Projects/2026_Feb5/projectaria_gen2_depth_from_stereo"
ERROR_BRANCH="colab-errors"
CHECK_INTERVAL=15  # seconds

cd "$REPO_DIR" || exit 1

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          CLAUDE AUTO-FIXER - Watching for Colab errors              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Monitoring GitHub for errors from Colab..."
echo "Press Ctrl-C to stop"
echo ""

last_error_hash=""

while true; do
    # Fetch latest
    git fetch origin "$ERROR_BRANCH" 2>/dev/null >/dev/null

    # Check if error branch exists and has new commits
    current_hash=$(git rev-parse "origin/$ERROR_BRANCH" 2>/dev/null || echo "")

    if [ -n "$current_hash" ] && [ "$current_hash" != "$last_error_hash" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚨 NEW ERROR DETECTED"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Checkout error branch
        git checkout "$ERROR_BRANCH" --quiet 2>/dev/null

        # Show error
        if [ -f "COLAB_ERROR.txt" ]; then
            cat "COLAB_ERROR.txt"
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "⏸️  PAUSED FOR CLAUDE TO ANALYZE AND FIX"
            echo ""
            echo "Claude should now:"
            echo "  1. Read the error above"
            echo "  2. Identify the issue"
            echo "  3. Fix the code"
            echo "  4. Commit and push to main"
            echo ""
            echo "After Claude pushes the fix, Colab will automatically:"
            echo "  - Detect the new commit"
            echo "  - Pull the fixed code"
            echo "  - Retry the pipeline"
            echo ""
            echo "Press Enter when fix is pushed to continue monitoring..."
            read

            # Switch back to main
            git checkout main --quiet
        fi

        last_error_hash="$current_hash"
    fi

    # Show heartbeat
    echo -ne "⏳ Checking for errors... (last check: $(date +%H:%M:%S))\r"
    sleep $CHECK_INTERVAL
done
