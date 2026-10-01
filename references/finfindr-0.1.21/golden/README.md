# Golden Result

`core-golden.json` is the reviewed output of the deterministic synthetic cases
defined in `R/golden_cases.R`. It records the exact upstream commit, selected
source-file hashes, the R and package versions, explicit coverage boundaries,
and canonical hashes of larger numerical arrays.

The reference platform is Apple arm64 with R 4.5.2. A mismatch on another
platform is evidence to review, not permission to replace this file
automatically.

No field photograph, animal identifier, catalog record, or location data is
used in the fixture.

`ml-inference-golden.json` freezes numerical summaries from deterministic
gradient inputs passed through the exact crop and six-class edge checkpoints
using FinShape Surfer's inference-only compatibility layer. It is an adapter
regression fixture, not evidence of numerical parity with the unavailable
original R/MXNet runtime and not evidence of scientific accuracy.

`SHA256SUMS` binds the golden file to its reference metadata, lockfile, and
harness source. `VALIDATION.md` records the environment, commands, passing
scope, and blocked scope from the reference run.
