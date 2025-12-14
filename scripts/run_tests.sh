#!/bin/bash
# Run RESU tests with various configurations

set -e

echo "========================================="
echo "RESU Test Suite"
echo "========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run tests
run_test() {
    local name=$1
    local args=$2
    echo ""
    echo -e "${YELLOW}Running: $name${NC}"
    echo "========================================="
    if pytest $args; then
        echo -e "${GREEN}✓ $name passed${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
        exit 1
    fi
}

# Quick tests (no slow, no integration)
run_test "Quick unit tests" "-m 'not slow and not integration'"

# All unit tests
run_test "All unit tests" "-m 'not integration'"

# Integration tests
if [ "${RUN_INTEGRATION:-0}" = "1" ]; then
    run_test "Integration tests" "-m 'integration'"
fi

# All tests
if [ "${RUN_ALL:-0}" = "1" ]; then
    run_test "All tests" ""
fi

# Coverage
if [ "${RUN_COVERAGE:-0}" = "1" ]; then
    echo ""
    echo -e "${YELLOW}Running with coverage${NC}"
    pytest --cov=resu --cov-report=html --cov-report=term
    echo -e "${GREEN}Coverage report generated in htmlcov/${NC}"
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}All selected tests passed!${NC}"
echo -e "${GREEN}=========================================${NC}"
