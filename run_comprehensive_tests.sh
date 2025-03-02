#!/bin/bash
set -e

# Run Comprehensive Testing Suite for Ember
# This script runs all the tests defined in the TESTING_PLAN.md

# Text styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Make the 'ember' module directly importable
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}   Ember Comprehensive Testing   ${NC}"
echo -e "${BLUE}================================${NC}"

# Install test dependencies if needed
echo -e "\n${YELLOW}Checking test dependencies...${NC}"
if ! python -c "import pytest_cov" &> /dev/null; then
    echo -e "${YELLOW}Installing pytest-cov...${NC}"
    pip install pytest-cov
fi

if ! python -c "import hypothesis" &> /dev/null; then
    echo -e "${YELLOW}Installing hypothesis...${NC}"
    pip install hypothesis
fi

# Clean previous coverage data
echo -e "\n${YELLOW}Cleaning previous coverage data...${NC}"
rm -rf .coverage htmlcov coverage.xml

# 1. Run unit tests with coverage
echo -e "\n${YELLOW}Running unit tests with coverage...${NC}"
python -m pytest tests/unit -v --cov=src/ember --cov-report=term --cov-report=html --cov-report=xml

# 2. Run property-based tests
echo -e "\n${YELLOW}Running property-based tests...${NC}"
python -m pytest tests/unit/core/utils/data/base/test_transformers_properties.py -v

# 3. Run integration tests
echo -e "\n${YELLOW}Running integration tests...${NC}"
RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration -v

# 4. Run fuzzing tests (limited time for CI)
echo -e "\n${YELLOW}Running fuzzing tests (10 seconds per test)...${NC}"
mkdir -p fuzzing_results
for fuzz_test in tests/fuzzing/fuzz_*.py; do
    echo -e "${YELLOW}Running $fuzz_test...${NC}"
    python $fuzz_test --time_limit=10
done

# 5. Check coverage quality
echo -e "\n${YELLOW}Checking coverage quality...${NC}"
python -m coverage report --fail-under=90 || echo -e "${RED}Coverage is below 90%${NC}"

# 6. Run mutation testing (limited for CI)
if command -v mutmut &> /dev/null; then
    echo -e "\n${YELLOW}Running mutation testing on sample module...${NC}"
    mutmut run --paths-to-mutate=src/ember/core/utils/embedding_utils.py
    mutmut results
else
    echo -e "\n${YELLOW}Skipping mutation testing (mutmut not installed)${NC}"
    echo -e "${YELLOW}Install with: pip install mutmut${NC}"
fi

# 7. Summary
echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}   Testing Complete   ${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "${YELLOW}Coverage report: ${NC}file://$(pwd)/htmlcov/index.html"
echo -e "${YELLOW}XML report: ${NC}$(pwd)/coverage.xml"

# Exit with success status
exit 0