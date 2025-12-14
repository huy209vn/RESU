#!/bin/bash
# Quick verification that RESU works

echo "========================================="
echo "RESU Quick Verification"
echo "========================================="
echo ""

# Run verification script
python scripts/verify_resu.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ RESU verification PASSED"
    echo "========================================="
    echo ""
    echo "Your RESU implementation is working correctly!"
    echo "Weights were resurrected and performance recovered."
    echo ""
    echo "Next steps:"
    echo "  1. Run full test suite: pytest"
    echo "  2. Run benchmarks: python benchmarks/bench_throughput.py"
    echo "  3. Start experiments for your paper"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ RESU verification FAILED"
    echo "========================================="
    echo ""
    echo "Please check the error messages above."
    echo ""
fi

exit $exit_code
