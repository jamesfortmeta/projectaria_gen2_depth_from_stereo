#!/bin/bash
# Read Colab Error - Fetch error from git or local file
# Usage: Called by Claude when user says "read colab error"

REPO_DIR="/Users/jamesfort/Documents/Projects/2026_Feb5/projectaria_gen2_depth_from_stereo"
ERROR_FILE="colab_error.txt"

cd "$REPO_DIR" || exit 1

echo "🔄 Checking for Colab error..."
echo ""

# Try to pull latest (in case error was pushed to git)
git pull origin main --quiet 2>/dev/null

# Check if error file exists in repo
if [ -f "$ERROR_FILE" ]; then
    echo "📄 Found error in repo: $ERROR_FILE"
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    cat "$ERROR_FILE"
    echo "══════════════════════════════════════════════════════════════════════"
    echo ""
    echo "✅ Error loaded. Analyzing..."
    exit 0
else
    echo "❌ No error file found in repo"
    echo ""
    echo "The error should have auto-downloaded to your Downloads."
    echo "Please run this command:"
    echo ""
    echo "  cp ~/Downloads/colab_error.txt $REPO_DIR/"
    echo ""
    echo "Then say 'read colab error' again."
    exit 1
fi
