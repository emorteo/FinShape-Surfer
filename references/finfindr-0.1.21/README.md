# FinFindR 0.1.21 Reference Environment

This directory provides an isolated, reproducible harness for generating golden
results from selected FinFindR core algorithms. It does not integrate FinFindR
into FinShape Surfer and does not modify the frozen manual workflow.

## Upstream identity

- Repository: `https://github.com/haimeh/finFindR.git`
- Commit: `5d7babd208af9642e440ef9143ec679c1154ff91`
- Declared package version: `0.1.21`
- Upstream license: GPL 2 or later

FinFindR does not publish a `0.1.21` Git tag. The pinned commit is the first
upstream commit whose `DESCRIPTION` declares version 0.1.21. Later commits on
the same branch retained that version number, so the commit hash is the actual
reproducibility boundary.

The upstream repository is downloaded into `.reference/upstream/`, which is
ignored by Git. No GPL-licensed FinFindR source is vendored into this MIT
repository.

## Validated scope

Golden cases use deterministic synthetic inputs and cover:

- path encoding and decoding;
- numeric distance comparison;
- C++ A* edge-path search;
- C++ non-maximum edge suppression;
- C++ annulus extraction;
- C++ glare filling; and
- bounded image resizing.

The following remain **BLOCKED / NOT TESTED in the original R runtime**:

- numerical parity of MXNet fin detection;
- numerical parity of MXNet learned trailing-edge segmentation;
- MXNet triplet embeddings and learned catalog matching; and
- the Shiny application.

FinFindR 0.1.21 calls MXNet at runtime but does not declare it in
`DESCRIPTION`. The legacy R MXNet stack is not available for the current
Apple-silicon R runtime. These capabilities must not be described as validated
until a separately pinned legacy runtime executes the original models.

FinShape Surfer now also contains a narrow Python/PyTorch compatibility layer
for the exact crop and six-class edge symbol graphs. Both checkpoints execute
deterministically and have adapter regression fixtures. That evidence does not
replace a comparison against the original R/MXNet runtime; see
`../../docs/FINFINDR_ASSISTED_VISION.md`.

## Build and run

Requirements:

- R 4.5.x;
- command-line Git;
- a C++ compiler; and
- CRAN access during restore.

On macOS, `imager` must be built from source with its required FFTW and TIFF
libraries. The environment used for the committed golden file uses Homebrew
`fftw`, `libtiff`, `libx11`, and `pkgconf`; the CImg display path is disabled,
so the administrator-level XQuartz installation is not required.

Install those user-space prerequisites once with:

```bash
brew install fftw libtiff libx11 pkgconf
```

From this directory:

```bash
Rscript scripts/bootstrap.R
Rscript scripts/verify_golden.R
```

To deliberately regenerate the baseline:

```bash
Rscript scripts/generate_golden.R
Rscript scripts/verify_golden.R
```

Do not regenerate a golden file merely to silence a failure. First identify
whether the upstream commit, R version, package lock, compiler, or algorithmic
output changed, and review the scientific and integration impact.

After the first successful bootstrap, the pinned upstream checkout is reused
without network access. Set `FINFINDR_REFRESH_UPSTREAM=1` only when deliberately
checking that the same commit is still retrievable.

## Build adapter

The reference loader does not change FinFindR calculations. It copies the three
C++ files into the ignored build cache and removes only two obsolete
`Rcpp::depends(imager)` annotations. Modern explicit include flags replace that
build metadata. Source hashes for the original upstream files are recorded in
the golden result.

## Data governance

All inputs are generated mathematically. No field photograph, animal catalog
identifier, location, EXIF record, or unpublished research data is used or
stored.
