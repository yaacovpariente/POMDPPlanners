from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="POMDPPlanners",
    version="0.1.0",
    author="Yaacov Pariente",
    author_email="yaacovpar@gmail.com",
    description="A Python package for POMDP planning algorithms and environments",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yaacovpariente/POMDPPlanners",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        "dev": read_requirements('requirements-dev.txt'),
        "docs": [
            "sphinx",
            "sphinx-autodoc-typehints",
            "sphinx-rtd-theme",
        ],
    },
    keywords="pomdp planning reinforcement-learning monte-carlo-tree-search artificial-intelligence",
    project_urls={
        "Bug Reports": "https://github.com/yaacovpariente/POMDPPlanners/issues",
        "Source": "https://github.com/yaacovpariente/POMDPPlanners",
        "Documentation": "https://yaacovpariente.github.io/POMDPPlanners/",
    },
) 