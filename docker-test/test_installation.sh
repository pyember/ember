#!/bin/bash
# Script to test the Ember installation in a Docker container

set -e  # Exit on error

# Create needed directories
mkdir -p docker-test

# Check if Dockerfile exists
if [ ! -f "docker-test/Dockerfile" ]; then
    echo "Error: Dockerfile not found in docker-test directory"
    exit 1
fi

# Build the Docker image
echo "Building Docker image for testing..."
docker build -t ember-test -f docker-test/Dockerfile .

# Run the minimal example
echo -e "\n\n==== Running Minimal Example ===="
docker run --rm ember-test src/ember/examples/basic/minimal_example.py

# Run the minimal operator example
echo -e "\n\n==== Running Minimal Operator Example ===="
docker run --rm ember-test src/ember/examples/basic/minimal_operator_example.py

# Test import functionality
echo -e "\n\n==== Testing Import Functionality ===="
docker run --rm ember-test -c "import ember; print(f'Ember version: {ember.__version__}')"

echo -e "\n\n==== Installation Test Complete ===="
echo "All tests passed successfully!"