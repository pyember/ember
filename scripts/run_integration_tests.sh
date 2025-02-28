#!/bin/bash

# Script to run integration tests with the proper environment variables

# Set default options
RUN_MOCKED=true
RUN_REAL_API=false
VERBOSE=false
TARGET="tests/integration"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --with-api)
      RUN_REAL_API=true
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --target=*)
      TARGET="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--with-api] [--verbose|-v] [--target=path/to/tests]"
      exit 1
      ;;
  esac
done

# Setup environment variables
export RUN_INTEGRATION_TESTS=1

if [ "$RUN_REAL_API" = true ]; then
  export ALLOW_EXTERNAL_API_CALLS=1
  echo "🌐 Running tests with REAL external API calls enabled"
else
  echo "🔄 Running tests with mocked external dependencies"
fi

# Set verbosity
PYTEST_ARGS=""
if [ "$VERBOSE" = true ]; then
  PYTEST_ARGS="-v"
fi

# Run the tests
echo "🧪 Running integration tests: $TARGET"
python -m pytest $TARGET $PYTEST_ARGS

# Clean up environment variables
unset RUN_INTEGRATION_TESTS
unset ALLOW_EXTERNAL_API_CALLS 