#!/usr/bin/env python
"""
Simple script to add src directory to Python's path using a .pth file.

This enables imports like 'from ember.core import ...' instead of 
'from ember.core import ...' without needing to change import statements.
"""

import os
import sys
import site
from pathlib import Path


def create_path_file():
    """Create a .pth file to add src directory to Python's path."""
    # Get the current site-packages directory
    site_packages = site.getsitepackages()[0]

    # Create the path to src directory
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / "src"

    # Create the .pth file in site-packages
    pth_file = Path(site_packages) / "ember-src.pth"

    # Write the src path to the .pth file
    with open(pth_file, "w") as f:
        f.write(str(src_path))

    print(f"Created {pth_file} with path: {src_path}")
    print("Now imports like 'from ember.core import ...' should work properly.")


if __name__ == "__main__":
    create_path_file()
