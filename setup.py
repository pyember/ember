from setuptools import setup, find_namespace_packages

setup(
    name="ember",
    version="0.1.0",
    description="Compound AI Systems framework for Network of Network (NON) construction.",
    author="Jared Quincy Davis",
    author_email="jaredq@cs.stanford.edu",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "pandas>=1.0.0,<2.2.0",
        "numpy>=1.21.0,<1.27.0",
    ],
)
