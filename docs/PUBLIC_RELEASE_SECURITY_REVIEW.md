# Public Release Security Review

Review date: 2026-09-25  
Publication target: `https://github.com/emorteo/FinShape-Surfer`

## Scope

The review covered the current working tree, untracked release candidates,
the source repository's Git history and remote configuration, archived validation records,
common credential signatures, suspicious secret assignments, high-entropy text
tokens, filenames associated with credentials, and metadata embedded in the
three public repository images.

The `.git` object database and local test virtual environment were excluded
from the current-tree content scan. Git history was scanned separately.

## Results

### Credentials and secrets

No API keys, access tokens, passwords, private keys, authenticated remote URLs,
database connection strings, secret-bearing environment files, or credential
files were found in the working tree or commit diffs.

The high-entropy review produced SHA-256 digests and the intentionally embedded
PNG fallback in `FinShape_Surfer_GUI_1.1.3.py`. These were reviewed as expected
project data, not credentials.

### Local-machine paths

Four archived 1.2.2 validation records exposed a local macOS username and
virtual-environment path. Their public copies were sanitized using portable
`python` and temporary-path placeholders. The numerical results, timings, test
outcomes, production source, and scientific behavior were not changed.

The affected records are:

- `Code/Surfer_validation_1.2.2/README_release.md`
- `Code/Surfer_validation_1.2.2/TEST_COMPLETION.md`
- `Code/Surfer_validation_1.2.2/commands_executed.txt`
- `Code/Surfer_validation_1.2.2/results.json`

The baseline hash manifest and golden fixture were regenerated after the
redactions so that integrity tests continue to protect the public copies.

### Image metadata

The three public PNG files contain no EXIF records, GPS fields, textual metadata, email
addresses, or embedded local filesystem paths detectable by Pillow and binary
string inspection.

### Git history identity data

The source repository's existing commits contain personal author email
addresses. They are not being copied to the public repository. The public
candidate is created as a new repository with one root commit authored with
Eduardo Morteo's GitHub no-reply address, so the older identity metadata and
private-file history cannot be reached from the public commit graph.

### Private research data

The repository owner directed that catalog artifacts and identifiers remain
private. The rendered Photo-ID artifact, its filename, identifier,
case-specific metrics, and replay record were removed from the public candidate
before its first commit. Synthetic rendering artifacts remain for regression
evidence.

### Generated outputs

The experimental surface exporter deliberately records local provenance paths
and catalog identifiers in generated packages. This supports reproducibility
but may expose local usernames, directory structures, or restricted catalog
data when an output archive is shared. Users must inspect and sanitize generated
manifests before external distribution.

## Tooling limitations

No dedicated secret-scanning program was installed in the environment. The
review used repository-wide and history-wide signature searches plus a custom
high-entropy review. A dedicated scanner such as GitHub secret scanning should
also be enabled after publication; it complements rather than replaces this
manual review.

## Verification

- Current-tree hardcoded personal-path scan: no remaining matches.
- Current-tree credential-signature scan: no matches.
- Image metadata and binary-string review: no sensitive metadata detected.
- Baseline integrity manifest: 22 protected files, 0 mismatches.
- Repository regression suite: 28 passed.
- Archived 1.2.2 HQ suite: 8 passed from a temporary copy so the test did not
  overwrite frozen validation evidence.
- Release-specific edits introduce no new whitespace defects. A clean-root
  staged check still reports legacy trailing whitespace inside preserved
  baseline and prototype files; those files were not reformatted because that
  would invalidate the frozen hashes and broaden this publication-only change.

## Release assessment

**Source and credential status:** ready after the recorded path redactions and
successful regression verification.

**Research-data status:** ready; the private catalog artifact and identifier
were excluded before creation of the public repository history.

**Git author-email status:** ready for a clean public root commit using a GitHub
no-reply address; the source repository's earlier history is excluded.
