from setuptools import setup, find_packages

setup(
    name="POMDPPlanners",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "mlflow>=2.8.0",
    ],
) 