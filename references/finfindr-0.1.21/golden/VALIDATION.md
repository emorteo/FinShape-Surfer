# Reference Validation

Validation date: 2026-09-28

## Environment

- Host architecture: Apple Silicon (`arm64`)
- R: 4.5.2
- FinFindR source: commit `5d7babd208af9642e440ef9143ec679c1154ff91`
- FinFindR declared version: 0.1.21
- Dependency state: `renv.lock`

## Results

The reference bootstrap was run from a clean temporary copy with empty R and
renv caches. It restored the lockfile, rebuilt `imager` 1.0.8 from source
against Homebrew libraries, fetched the exact upstream commit, compiled the
selected C++ routines, and verified the committed golden file successfully.

The golden verifier also passed twice in the development worktree without
regeneration. The second run reused the exact local upstream checkout and did
not fetch the network.

Before assisted-vision implementation, the unchanged FinShape Surfer Python
suite passed all 28 tests. After the additive implementation, the complete
suite passed with the external-model tests skipped when no model directory was
configured; the two checkpoint tests also passed when run against the pinned
ignored checkout.

## Commands

```bash
Rscript scripts/bootstrap.R
Rscript scripts/generate_golden.R
Rscript scripts/verify_golden.R
Rscript scripts/verify_golden.R
python -m pytest -q -p no:cacheprovider
```

## Scope boundary

The deterministic R/C++ core cases passed. The exact crop and six-class edge
checkpoints also execute through the new inference-only Python adapter and have
deterministic regression summaries. Their numerical parity against original
R/MXNet inference remains **UNVERIFIED** because FinFindR 0.1.21 relies on an
undeclared legacy R MXNet runtime that is unavailable for this R 4.5.2 Apple
Silicon environment. Triplet embedding, learned catalog matching, and Shiny
workflows remain **BLOCKED / NOT TESTED**. No private image is stored in the
golden fixture.
