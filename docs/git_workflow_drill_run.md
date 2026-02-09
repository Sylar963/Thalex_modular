# Git Workflow: Drill and Run Version Tags

## Overview
This workflow is used to merge development changes from `MFT-dev` branch into `main` and create annotated version tags for release tracking.

## The Drill and Run Process

### Step 1: Check Current Git Status
```bash
git status && git log --oneline -10 && git diff main...MFT-dev --stat
```
**Purpose**: Verify current branch state, recent commits, and changes ahead of origin.

### Step 2: Push MFT-dev to Origin
```bash
git push origin MFT-dev
```
**Purpose**: Push all development commits to the remote repository.

### Step 3: Checkout Main and Fast-Forward Merge
```bash
git checkout main && git pull && git merge MFT-dev --ff-only
```
**Purpose**: Switch to main branch and merge MFT-dev using fast-forward only (no merge commits).

### Step 4: Create Version Tag
```bash
git tag -a v0.X.X-alpha -m "Version notes here"
```
**Purpose**: Create an annotated tag with release notes describing major changes.

### Step 5: Push Main and Tag to Origin
```bash
git push origin main && git push origin v0.X.X-alpha
```
**Purpose**: Push the merged main branch and the new version tag to remote.

## Key Principles

1. **Fast-Forward Only**: Ensures linear history on main branch
2. **Annotated Tags**: Tags contain messages for release documentation
3. **Branch Synchronization**: Both MFT-dev and main are kept in sync with origin
4. **Version Increments**: Minor version bumps (0.2.6 → 0.2.7 → 0.2.8) for incremental releases

## Tag Naming Convention
- `v0.X.Y-alpha` for alpha releases
- Increment Y for incremental changes
- Increment X for major feature releases

## Example Workflow Output
```
On branch MFT-dev
Your branch is ahead of 'origin/MFT-dev' by N commits.

[After push]
To https://github.com/Sylar963/Thalex_modular.git
   old_commit..new_commit  MFT-dev -> MFT-dev

[After merge]
Updating old_commit..new_commit
Fast-forward
   N files changed, X insertions(+), Y deletions(-)

[After tag push]
To https://github.com/Sylar963/Thalex_modular.git
   old_commit..new_commit  main -> main
To https://github.com/Sylar963/Thalex_modular.git
 * [new tag]         v0.X.Y-alpha -> v0.X.Y-alpha
```

## Verification Commands
```bash
# List all tags
git tag -l

# View tag details
git show v0.X.Y-alpha

# View changes between tags
git diff v0.X.Y-alpha...v0.X.Z-alpha --stat
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Tag already exists | `git tag -d v0.X.X-alpha` then recreate |
| Merge conflicts | Resolve manually, commit, then continue |
| Not on MFT-dev | `git checkout MFT-dev` first |
| Working tree not clean | Commit or stash changes first |
