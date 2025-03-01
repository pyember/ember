#!/usr/bin/env python3
"""
Create a Python link so tests can import ember.

This script creates a symbolic link from src/ember to the root directory
to ensure imports work correctly. This is a simple solution for Python
package import problems.

For a proper solution in production, use `pip install -e .`.
"""

import os
import sys
import shutil
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.absolute()
SRC_DIR = ROOT_DIR / "src"
EMBER_SRC = SRC_DIR / "ember"
EMBER_LINK = ROOT_DIR / "ember"

def main():
    """Create a symbolic link from src/ember to root/ember."""
    # Safety check - is this the right directory?
    if not (ROOT_DIR / "pyproject.toml").exists():
        print("Error: This script should be run from the project root directory.")
        return 1
    
    # Check if ember directory already exists at root
    if EMBER_LINK.exists():
        if os.path.islink(EMBER_LINK):
            # If it's already a symlink, we're good
            print(f"Symbolic link already exists: {EMBER_LINK}")
            return 0
        else:
            # If it's a real directory, back it up
            backup_dir = ROOT_DIR / "ember.bak"
            print(f"Moving existing ember directory to {backup_dir}")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(EMBER_LINK, backup_dir)
    
    # Check if source directory exists
    if not EMBER_SRC.exists():
        print(f"Error: Source directory not found: {EMBER_SRC}")
        return 1
    
    # Create the symbolic link
    print(f"Creating symbolic link: {EMBER_SRC} -> {EMBER_LINK}")
    os.symlink(EMBER_SRC, EMBER_LINK, target_is_directory=True)
    print("Done! Python imports should now work correctly.")
    
    # Instructions
    print(f"\nNow you can run tests with:")
    print(f"  python -m pytest tests/unit/core/test_non.py -v")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())