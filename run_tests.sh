#!/bin/bash

# Make the 'ember' module directly importable
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

# Run the tests
pytest tests "$@" 