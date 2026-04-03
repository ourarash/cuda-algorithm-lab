---
name: repo-commit-helper
description: Prepare repo-aware Conventional Commit messages for cuda-algorithm-lab and optionally complete the commit workflow. If nothing is staged, ask whether to stage all current changes first.
---

# Repo Commit Helper

Use this skill when the user wants help preparing a commit message or making a git commit in this repository.

## Workflow

1. Check whether anything is already staged.

```bash
git diff --cached --name-only
```

2. If nothing is staged:
   - Check whether the working tree has any modified or untracked files.

```bash
git status --short
```

   - If the working tree is dirty, ask this exact question before continuing:
     `No files are staged. Should I stage all current changes with git add -A before I prepare the commit?`
   - Only run `git add -A` if the user says yes.
   - If the user says no, stop and let them choose what to stage manually.
   - If the working tree is clean, say there is nothing to commit and stop.

3. Analyze the staged change set deeply:
   - Read the staged diff non-interactively with:

```bash
git --no-pager diff --staged
```

   - Also inspect full staged file contents, not just hunks. Use staged blob reads such as:

```bash
git diff --cached --name-only
git show :path/to/file
```

4. Interpret changes in repo context:
   - This repo is an educational CUDA collection with kernels, algorithm
     progressions, and interactive HTML visualizers.
   - Common topics include reduction, scan, matrix multiplication, sparse
     operations, warp-level programming, and basic CUDA API demonstrations.
   - Meaningful impact usually involves algorithm correctness, kernel structure,
     memory-access patterns, comments/documentation, visualization logic, or
     educational flow.
   - Some updates may touch Makefiles, folder organization, repo-local plugin
     configuration, or README files.

5. Write a Conventional Commit message:
   - Focus on meaningful behavior, CUDA algorithm correctness, educational
     content, docs, visualizers, or repo organization improvements.
   - Ignore pure formatting, whitespace-only, or similarly trivial edits unless they
     support a meaningful change.
   - Prefer user-visible learning and interaction outcomes, such as corrected
     kernel logic, better memory behavior explanations, improved visualizer
     logic, added controls, clearer comments, or discoverability.

6. If the user asked to actually commit:
   - Draft the final commit message first.
   - Show the message to the user for confirmation unless they clearly asked you to
     proceed without review.
   - After confirmation, run a non-interactive `git commit` with the drafted message.

## Output Format For Commit Message Drafts

Return only a single plain-text commit message with these sections and no extra commentary.

1. Subject line
   - Format: `<type>: <description>`
   - Allowed types: `feat`, `fix`, `refactor`, `docs`, `style`, `test`, `chore`
   - Use imperative present tense.
   - Max 250 characters.
   - Reflect highest-level purpose.

2. Body (optional, 1-5 lines)
   - Explain why the change was made.
   - Wrap at about 80 characters.
   - Leave one blank line after the subject.

3. Changes section (required)
   - Title exactly: `Changes:`
   - Then bullets:
     `- <file_path>: <meaningful functional or structural change>`
   - Include only meaningful changes.
   - Group logically by module or topic when useful.

4. Footer (optional)
   - If needed, include:
     `BREAKING CHANGE: <description>`
   - Wrap at about 80 characters.
   - Separate from the previous section with one blank line.
