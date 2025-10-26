#!/bin/bash
# Reset script for Inaugural Address Analysis
# Removes all analysis results but preserves the original text files

echo "=========================================="
echo "RESETTING INAUGURAL ADDRESS ANALYSIS"
echo "=========================================="
echo ""

# Current directory should be the inagural_addresses directory
CURRENT_DIR=$(pwd)

echo "Current directory: $CURRENT_DIR"
echo ""

# Remove analysis output files
echo "Removing analysis output files..."
rm -f inaugural_similarity_heatmap.png
rm -f inaugural_tsne_clusters.png
rm -f inaugural_analysis_results.csv
rm -f inaugural_top_similar_pairs.csv

echo "✓ Analysis output files removed"
echo ""

# Show what text files we're preserving
TXT_COUNT=$(find txt -name "*.txt" 2>/dev/null | wc -l | xargs)
echo "Preserving $TXT_COUNT text files in txt/ directory"

# Show what CSV metadata exists
if [ -f "inaugural_addresses.csv" ]; then
    echo "Preserving inaugural_addresses.csv (metadata)"
else
    echo "No inaugural_addresses.csv found"
fi

echo ""
echo "=========================================="
echo "RESET COMPLETE"
echo "=========================================="
echo ""
echo "Preserved:"
echo "  • $TXT_COUNT text files in txt/ directory"
echo "  • inaugural_addresses.csv (if it exists)"
echo ""
echo "Removed:"
echo "  • All PNG visualization files"
echo "  • All analysis result CSV files"
echo ""
echo "To run analysis again:"
echo "  python analyze_inaugural_addresses.py"
echo ""

