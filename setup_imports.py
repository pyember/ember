"""
Helper script for setting up proper import structure in Ember

This creates a .pth file in the virtual environment site-packages directory
to add the src/ directory to Python's path, allowing imports without the 'src.' prefix.
"""

import sys
import site
import os
from pathlib import Path


def setup_imports():
    """Add the src directory to Python's path via a .pth file."""
    # Get the current virtual environment site-packages directory
    site_packages = site.getsitepackages()[0]

    # Create the .pth file to add src directory to Python's path
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / "src"

    # Create the .pth file
    pth_file = Path(site_packages) / "ember-src.pth"

    with open(pth_file, "w") as f:
        f.write(str(src_path))

    print(f"Created {pth_file} to add {src_path} to Python's path")
    print(
        "Now you can use 'from ember.core import ...' instead of 'from ember.core import ...'"
    )


if __name__ == "__main__":
    setup_imports()
