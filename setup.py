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
    # long_description=open("README.md").read(),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        "dev": [
            "pylint",
            "black",
            "pre-commit",
        ],
    },
) 