# Publishing and Fast Updates

This project can be distributed as a Python package. The recommended public
release path is GitHub Releases first, with optional PyPI publishing later.

## Release Model

Use semantic versions in `setup.py`, for example:

```python
version="0.1.1"
```

Create a matching Git tag:

```powershell
git tag v0.1.1
git push origin v0.1.1
```

The GitHub Actions workflow `.github/workflows/release.yml` builds:

- `dist/mea_pipeline-<version>-py3-none-any.whl`
- `dist/mea_pipeline-<version>.tar.gz`

For tag builds, both files are attached to the GitHub Release.

## User Installation

Install the latest code directly from GitHub:

```powershell
python -m pip install -U git+https://github.com/<owner>/<repo>.git
```

Install a specific release wheel:

```powershell
python -m pip install -U https://github.com/<owner>/<repo>/releases/download/v0.1.1/mea_pipeline-0.1.1-py3-none-any.whl
```

Launch the GUI after installation:

```powershell
mea-pipeline
```

or:

```powershell
mea-pipeline-gui
```

## Local Package Build

Build locally before publishing:

```powershell
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Install the generated wheel locally:

```powershell
python -m pip install -U dist\mea_pipeline-0.1.1-py3-none-any.whl
```

## Optional PyPI Publishing

If the package name is available on PyPI, publish there so users can run:

```powershell
python -m pip install -U mea-pipeline
```

Recommended PyPI setup:

1. Create a PyPI project.
2. Configure PyPI trusted publishing for the GitHub repository.
3. Add a separate publish job using `pypa/gh-action-pypi-publish`.

Do not include local `data/`, `.github/`, tests, notebooks, or docs in the
package distribution. `MANIFEST.in` already excludes those folders from
source distributions.
