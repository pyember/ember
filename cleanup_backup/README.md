# Ember Codebase Cleanup

This directory contains files that were removed from the main codebase during a cleanup effort.

## Files Backed Up

1. Development utility scripts:
   - `check_imports.py`: Script to check Python imports
   - `fix_python_path.py`: Script to create a symbolic link for imports
   - `fix_imports.py`: Script to fix import paths
   - `fix_newlines.py`: Script to ensure proper file endings
   - `fix_unused_imports.py`: Script to remove unused imports

2. Non-standard test files:
   - `test_operator_return.py`: Test file outside the test directory
   - `modified_test_non.py`: Modified test file outside the test directory

3. Other misc files:
   - `__init__.py`: Empty file in the project root
   - `datasets_example.ipynb`: Outdated notebook
   - `log.txt`: Log file

## Reason for Removal

These files were removed to:
1. Simplify the repository structure
2. Remove one-off utility scripts that have served their purpose
3. Consolidate test files into proper test directories
4. Remove unnecessary files from version control

## Note on Python Import Methodology

Previously, a symlink (`ember -> src/ember`) was created in the root directory to facilitate imports during development. This approach has been replaced with proper Python packaging practices:

1. The project now uses `pyproject.toml` with proper package configuration:
   ```toml
   [tool.poetry]
   packages = [
       { include = "ember", from = "src" },
   ]
   ```

2. For development, use editable installs:
   ```bash
   # Using poetry
   poetry install
   
   # Or using pip
   pip install -e .
   ```

3. The Python path is properly configured in `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   pythonpath = [
       "src",
       "tests"
   ]
   ```

This approach follows Python best practices and eliminates the need for symlinks or path manipulation.

If any functionality from these files is still needed, they can be moved back or their functionality can be incorporated into more appropriate locations in the codebase.