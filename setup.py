from setuptools import setup, find_packages

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
    install_requires=[
        "numpy",
        "matplotlib",
        "mlflow>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pylint",
            "black",
            "pre-commit",
        ],
    },
) 