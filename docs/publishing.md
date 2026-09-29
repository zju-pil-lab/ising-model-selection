# Publishing this repository

Suggested repository name: `ising-model-selection`.

Suggested description:

> Replica theory and full-graph numerical reproduction of Figure 2 in Meng, Obuchi & Kabashima, NeurIPS 2021.

The release directory is ready to become the GitHub repository root. It includes the MIT license, English/Chinese README files, citation metadata, CI configuration and complete reference checkpoints. Do not upload a virtual environment, compiled `.so` files, local `outputs/`, or the release ZIP itself as source files.

## First push

Create an empty GitHub repository with your chosen owner/name, without adding a second README or license. Then, from this directory:

```bash
git init -b main
git add .
git commit -m "Release Figure 2 theory and simulation reproduction"
```

Add the remote URL supplied by GitHub, then push:

```bash
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

The workflow will run after the push. Its matrix covers Ubuntu/macOS and Python 3.11/3.13. Local checks are recorded separately; configured CI is not a claim that GitHub jobs have already run. Once CI passes, an optional first release tag is `v1.0.0`.

No remote URL is hard-coded into this project. After publishing, you may add your actual repository URL to the top level of `CITATION.cff` as `repository-code` and add a CI badge pointing to your own repository. Do not assign a fabricated DOI; a DOI can be added if one is later issued for an archived release.

## Build a source archive

```bash
python tools/package_release.py
```

This uses an allowlist, verifies reference checksums, and writes a ZIP plus file hashes to `dist/`. It excludes virtual environments, `.git`, generated native libraries and output directories. The GitHub repository itself should contain the unpacked source/data tree.

The previous private working directories and reproduction archives are not required to use this repository.
