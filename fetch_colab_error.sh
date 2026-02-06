#!/bin/bash
# Colab Error Fetcher
# Run this locally (with Claude) to automatically retrieve the latest error from Colab

set -e

REPO_DIR="/Users/jamesfort/Documents/Projects/2026_Feb5/projectaria_gen2_depth_from_stereo"
ERROR_BRANCH="colab-errors"

cd "$REPO_DIR"

echo "🔄 Fetching latest Colab error..."
echo ""

# Fetch latest from remote
git fetch origin "$ERROR_BRANCH" 2>/dev/null || {
    echo "❌ No errors branch found yet"
    echo "   (Colab hasn't reported any errors)"
    exit 1
}

# Check out error branch
git checkout "$ERROR_BRANCH" 2>/dev/null || git checkout -b "$ERROR_BRANCH" "origin/$ERROR_BRANCH"
git pull origin "$ERROR_BRANCH" --quiet

# Display the error
if [ -f "colab_error_log.txt" ]; then
    echo "📄 Latest Colab Error:"
    echo "═══════════════════════════════════════════════════════════════════════"
    cat colab_error_log.txt
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    echo "✅ Error retrieved successfully"
    echo ""

    # Switch back to main
    git checkout main --quiet

    exit 0
else
    echo "❌ No error log found"
    git checkout main --quiet
    exit 1
fi
