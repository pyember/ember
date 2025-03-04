#!/bin/bash
# Enhanced test runner script for Ember

# Terminal colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                 ${YELLOW}Ember Test Runner${BLUE}                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"

# Using proper Python packaging now - no need to manually set PYTHONPATH

# Check Python environment
echo -e "${YELLOW}Python environment:${NC}"
python --version
echo ""

# Parse arguments for test types
RUN_UNIT=true
RUN_INTEGRATION=false
RUN_PERFORMANCE=false
COVERAGE=true
VERBOSE=false

for arg in "$@"; do
  case $arg in
    --all)
      RUN_INTEGRATION=true
      RUN_PERFORMANCE=true
      ;;
    --integration)
      RUN_INTEGRATION=true
      ;;
    --performance)
      RUN_PERFORMANCE=true
      ;;
    --no-cov)
      COVERAGE=false
      ;;
    --verbose|-v)
      VERBOSE=true
      ;;
    --help|-h)
      echo -e "${GREEN}Ember Test Runner Usage:${NC}"
      echo -e "  ${YELLOW}./run_tests.sh [options] [pytest args]${NC}"
      echo -e ""
      echo -e "${GREEN}Options:${NC}"
      echo -e "  ${YELLOW}--all${NC}           Run all tests (unit, integration, performance)"
      echo -e "  ${YELLOW}--integration${NC}   Run integration tests"
      echo -e "  ${YELLOW}--performance${NC}   Run performance tests"
      echo -e "  ${YELLOW}--no-cov${NC}        Disable coverage reporting"
      echo -e "  ${YELLOW}--verbose, -v${NC}   Enable verbose output"
      echo -e "  ${YELLOW}--help, -h${NC}      Show this help message"
      echo -e ""
      echo -e "Any additional arguments are passed directly to pytest."
      exit 0
      ;;
  esac
done

# Prepare pytest arguments
PYTEST_ARGS=""

# Add coverage options if enabled
if [ "$COVERAGE" = true ]; then
  PYTEST_ARGS="$PYTEST_ARGS --cov=src/ember --cov-report=term"
  
  # Create coverage directory if it doesn't exist
  mkdir -p coverage
  PYTEST_ARGS="$PYTEST_ARGS --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml"
fi

# Add verbosity if requested
if [ "$VERBOSE" = true ]; then
  PYTEST_ARGS="$PYTEST_ARGS -v"
fi

# Run the tests
echo -e "${GREEN}Running unit tests...${NC}"
if ! python -m pytest tests/unit $PYTEST_ARGS; then
  echo -e "${RED}Unit tests failed!${NC}"
  exit 1
fi

# Run integration tests if requested
if [ "$RUN_INTEGRATION" = true ]; then
  echo -e "\n${GREEN}Running integration tests...${NC}"
  if ! python -m pytest tests/integration $PYTEST_ARGS; then
    echo -e "${RED}Integration tests failed!${NC}"
    exit 1
  fi
fi

# Run performance tests if requested
if [ "$RUN_PERFORMANCE" = true ]; then
  echo -e "\n${GREEN}Running performance tests...${NC}"
  if ! python -m pytest tests/integration/performance $PYTEST_ARGS; then
    echo -e "${RED}Performance tests failed!${NC}"
    exit 1
  fi
fi

echo -e "\n${GREEN}All tests passed successfully!${NC}"