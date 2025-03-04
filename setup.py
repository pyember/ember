"""
Ember - Compositional Framework for AI Systems

This setup.py file is provided for pip install compatibility.
For the full package configuration, please use Poetry.
"""

from setuptools import setup

# This file is intentionally minimal as poetry is the primary build system
# This enables pip install -e . for development
if __name__ == "__main__":
    setup(
        name="ember-ai",
        use_scm_version=True,
        description="Compositional framework for building and orchestrating Compound AI Systems and Networks of Networks (NONs).",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="Jared Quincy Davis",
        author_email="jared@mlfoundry.com",
        url="https://github.com/pyember/ember",
        package_dir={"": "src"},
        packages=["ember"],
        python_requires=">=3.11",
        # Only include the bare minimum to make pip install -e . work
        install_requires=[
            "poetry>=1.5.0",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
