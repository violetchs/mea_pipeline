from setuptools import find_packages, setup

setup(
    name='new_maxwell_experiment',
    version="0.1.0",
    packages=find_packages(include=["python", "python.*"]),
    install_requires=["pyyaml>=6.0", "numpy>=1.24", "h5py>=3.0"],
    python_requires=">=3.10",
)
