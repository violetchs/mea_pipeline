from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"
LONG_DESCRIPTION = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="mea-pipeline",
    version="0.3.3",
    description="MEA (Microelectrode Array) data processing pipeline for neural activity analysis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="MEA Pipeline contributors",
    python_requires=">=3.10",
    packages=find_packages(where="src", exclude=("tests", "tests.*")),
    py_modules=["pipeline"],
    package_dir={"": "src"},
    include_package_data=False,
    install_requires=[
        "numpy>=2.1.0",
        "scipy>=1.14.1",
        "pandas>=2.2.3",
        "matplotlib>=3.9.2",
        "scikit-learn>=1.5.2",
        "PySide6>=6.8.0",
        "h5py>=3.10.0",
        "Pillow>=11.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mea-pipeline=gui.app:main",
            "mea-pipeline-gui=gui.app:main",
        ],
        "gui_scripts": [
            "mea-pipeline-app=gui.app:main",
        ],
    },
    data_files=[
        ("share/mea_pipeline/config", ["config/channel_maps.json"]),
        ("share/mea_pipeline/maxwell_hdf5_plugin", ["tools/maxwell_hdf5_plugin/compression.dll"]),
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "pytest-cov>=5.0.0",
            "jupyter>=1.1.1",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
