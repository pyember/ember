# Docker-based Installation Testing for Ember

This directory contains tools for testing the Ember installation process in a consistent environment using Docker. This approach is particularly useful when:

1. You don't have Python 3.11+ installed locally
2. You want to test in a clean environment without affecting your system
3. You need to verify that the installation process works across different systems

## Prerequisites

- Docker installed and running on your system
- Basic familiarity with Docker and shell commands

## Files

- `Dockerfile`: Defines a Python 3.11 environment that installs Ember and its dependencies
- `test_installation.sh`: Script that builds the Docker image and runs the test examples

## Usage

From the root directory of the Ember project, run:

```bash
./docker-test/test_installation.sh
```

This script will:
1. Build a Docker image with Python 3.11 and all required dependencies
2. Run the minimal example to verify basic functionality
3. Run the minimal operator example to verify operator functionality
4. Test that the Ember package can be imported correctly

## Customizing Tests

You can modify the `test_installation.sh` script to run different examples or tests. For instance, to test a different example:

```bash
docker run --rm ember-test src/ember/examples/path/to/your_example.py
```

## Troubleshooting

If you encounter issues:

1. **Docker build failures**:
   Check that Docker is running and you have sufficient permissions.

2. **Package installation issues**:
   The Docker build logs will show detailed information about any package installation failures.

3. **Example runtime errors**:
   These indicate that while the installation succeeded, there may be issues with the code or dependencies.

## Alternative Approaches

If you prefer not to use Docker, see the `TESTING_INSTALLATION.md` file in the project root for other methods to test installation.