# Branch and Worktree Workflow

The `main` branch is the frozen, reviewable baseline. New features and fixes
must be developed on focused branches, preferably in separate Git worktrees.
Do not add another versioned application directory for each change.

## Before starting work

From the primary checkout:

```bash
git status --short --branch
git fetch origin
```

If the current branch is dirty, choose one safe action before switching:

1. Commit a coherent change with its tests.
2. Stash tracked and untracked work with a descriptive message:

   ```bash
   git stash push -u -m "wip: describe the interrupted work"
   ```

3. Stop and ask the repository owner which changes should be preserved when
   their ownership or intent is unclear.

Never discard or overwrite an unknown change to make a branch switch succeed.

## Create isolated work

Use one branch and one sibling worktree per task:

```bash
git worktree add -b feature/short-name ../FinShape-Surfer-short-name main
```

Use `fix/short-name` for defects and `docs/short-name` for documentation-only
work. A branch can be checked out in only one worktree at a time.

## Validate and integrate

Run the relevant automated tests inside the task worktree. Commit only source,
tests, documentation, small public fixtures, and reviewed golden results.
Dependency libraries, downloaded upstream repositories, private photographs,
catalog identifiers, and generated application output stay ignored.

After review, merge through a pull request or an explicit local merge. Remove
the completed worktree only after its branch is committed and recoverable:

```bash
git worktree remove ../FinShape-Surfer-short-name
git worktree prune
```

Delete the branch only after it has been merged and pushed.
