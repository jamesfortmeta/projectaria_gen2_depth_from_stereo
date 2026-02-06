#!/bin/bash
# Read Colab Error - Local utility for Claude to fetch and display Colab errors
# Usage: Called by Claude when user says "read colab error"

DOWNLOADS="$HOME/Downloads"
ERROR_FILE="colab_error.txt"

echo "🔍 Looking for Colab error in Downloads..."
echo ""

# Find the most recent colab_error.txt in Downloads
if [ -f "$DOWNLOADS/$ERROR_FILE" ]; then
    echo "📄 Found: $DOWNLOADS/$ERROR_FILE"
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    cat "$DOWNLOADS/$ERROR_FILE"
    echo "══════════════════════════════════════════════════════════════════════"
    echo ""
    echo "✅ Error loaded. I'll analyze this and fix the code."
    exit 0
else
    echo "❌ No error file found at: $DOWNLOADS/$ERROR_FILE"
    echo ""
    echo "Did the error auto-download? Check:"
    ls -lt "$DOWNLOADS" | grep "colab_error" | head -5
    exit 1
fi
