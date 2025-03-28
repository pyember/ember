#!/bin/bash

echo "Running all tests except those requiring API keys..."
uv run pytest tests --run-all-tests -v

echo ""
echo "To run performance tests: ./run_all_tests.sh perf"
echo "To run API tests: ./run_all_tests.sh api"
echo "To run all test types: ./run_all_tests.sh all"

if [[ "$1" == "perf" || "$1" == "all" ]]; then
    echo ""
    echo "Running performance tests..."
    uv run pytest tests --run-perf-tests -v
fi

if [[ "$1" == "api" || "$1" == "all" ]]; then
    echo ""
    echo "Running API tests..."
    echo "Note: These tests require proper API keys to be set in environment variables."
    uv run pytest tests --run-api-tests -v
fi
