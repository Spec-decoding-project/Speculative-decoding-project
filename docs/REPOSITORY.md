# Repository history and everyday Git use

This is an independent repository at `Frankstuff/Speculative-decoding-project`. Its clean starting point imports unmodified Python source from [`z-lab/dflash` commit `07ebd93db9f472af339b644bb70221ad8428328a`](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a), plus learning materials and project metadata cleanup. It begins a fresh Git history instead of importing DFlash's commits. Upstream was cloned into the cleanup workspace's gitignored `.context/` directory, and its files were copied without its `.git` history into this repository.

The earlier conditional-redrafting experiment at `2d683fcb2f005d088d949a5e47550e3a19d3c25b` is excluded from this baseline and preserved in the local recovery backup. All five `dflash/` source files are byte-identical to upstream. Documentation, maintainer/URLs, ignore rules, tour checks, and editor recommendations are repository-specific. The upstream PyPI publishing workflow is omitted.

The DFlash code remains derived from [z-lab/dflash](https://github.com/z-lab/dflash). Its [MIT license and copyright notice](../LICENSE) and [upstream documentation](UPSTREAM_README.md) are retained. Resetting Git history changes commit attribution and does not change who originally wrote that code. Package metadata identifies Frankstuff as this project's maintainer.

## Work directly in your repository

`main` is the default branch and the base for new workspaces. A pull request is not required to work on or push to your own repository. For an ordinary clone:

```bash
git clone https://github.com/Frankstuff/Speculative-decoding-project.git
cd Speculative-decoding-project
# Edit files, inspect the diff, and run the relevant checks.
git add path/to/changed-file
git commit -m "Describe the change"
git push origin main
```

Conductor uses a branch per workspace. From a workspace based on the clean `main`, after committing and checking changes, `git push origin HEAD:main` can publish directly to your repository's main branch. Use a normal push. If it is rejected because main advanced, fetch and integrate those changes first; do not force-push routine work. No upstream remote or upstream PR is needed.

## Existing workspaces after the reset

Workspaces created before this cleanup still have their old local branches. Merging or pushing those histories back into main would reintroduce the original commits. Start a new workspace from the clean `origin/main` for new work. Preserve any unfinished edits in an old workspace and port the relevant patch into a fresh workspace; do not blindly reset a worktree with uncommitted changes.

The remote `qwen-draft-model-changes` name is retained at the same clean starting commit for compatibility. Its previous history is backed up locally, rather than left as another historical branch on GitHub. Existing sibling worktrees are not reset by the cleanup.

## Local recovery record

The cleanup workspace's gitignored `.context/` directory contains `before-history-cleanup.bundle` with the original refs/history and `before-history-cleanup.json` with the original default branch, commit, and remote. It also contains a tar snapshot of the original code. Keep a separate copy of that directory if you plan to delete the cleanup workspace and want the backup long term. These backup files are not pushed.

To inspect the old version without changing the live repository, run the following from the cleanup workspace and choose an unused recovery destination:

```bash
git clone --branch qwen-draft-model-changes \
  .context/before-history-cleanup.bundle /path/to/unused-recovery-folder
```

GitHub calculates contributor displays from commit data, and its contributor graph focuses on the default branch. After a history rewrite, cached displays may take about 24 hours to refresh; GitHub documents contacting support if they remain incorrect afterward. The retained license attribution is separate from those displays. [GitHub contributor documentation](https://docs.github.com/en/repositories/viewing-activity-and-data-for-your-repository/viewing-a-projects-contributors).
