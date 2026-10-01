# Third-party notices

## FinFindR

The optional assisted-vision feature interoperates with model and symbol files
from [FinFindR](https://github.com/haimeh/finFindR), pinned to commit
`5d7babd208af9642e440ef9143ec679c1154ff91` (declared version 0.1.21).

FinFindR's `DESCRIPTION` declares `GPL (>= 2)`. Its source, checkpoints, and
other upstream files are not distributed in this MIT-licensed repository. The
preparation tool obtains them from their upstream repository into a local user
cache and verifies the exact checkpoint hashes before use.

Users and redistributors are responsible for reviewing and satisfying the
upstream license terms when obtaining or distributing FinFindR materials. This
notice is provenance information, not legal advice.

## PyTorch

The optional compatibility layer uses PyTorch for tensor inference. PyTorch is
installed separately through the `finfindr` optional dependency and is not
vendored in this repository. See the PyTorch distribution for its current
license and bundled third-party notices.

