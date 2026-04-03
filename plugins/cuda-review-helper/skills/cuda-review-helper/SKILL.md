---
name: cuda-review-helper
description: Review CUDA kernels, progression folders, Makefiles, and interactive visualizations in cuda-algorithm-lab with a focus on correctness, readability, educational flow, and repo consistency.
---

# CUDA Review Helper

Use this skill when the user wants a review of CUDA code, a folder progression,
an HTML visualization, or overall repository quality in this repository.

## Review Priorities

1. CUDA correctness
   - Check indexing logic carefully.
   - Check boundary conditions and power-of-two assumptions.
   - Check synchronization placement such as `__syncthreads()`.
   - Check shared-memory usage, bank-conflict comments, and any mismatch
     between comments and implementation.
   - Check whether kernels appear internally consistent with their launch
     configuration.

2. Educational clarity
   - Prefer files that explain the algorithm at the top in plain language.
   - Look for progression quality: each numbered step should justify why it
     exists and how it improves on the previous step.
   - Check whether comments explain intent rather than restating syntax.
   - For visualization files, check that the UI actually teaches the same
     algorithm the CUDA file implements.

3. Repository consistency
   - Folder numbering should match filenames, README references, and Makefiles.
   - Root-level naming should avoid duplicate or superseded examples.
   - Local plugin metadata, repo URLs, and marketplace entries should match the
     current repository identity.

4. Submission readiness
   - Call out broken Makefile paths, stale references, temp files, unfinished
     code, or obvious compile-time issues.
   - Prefer actionable findings over broad summaries.

## Workflow

1. Read the target files and nearby README/Makefile context.
2. If reviewing code, compare comments with actual implementation behavior.
3. If reviewing a visualization, compare the visual explanation with the CUDA
   kernel it is supposed to teach.
4. Report findings ordered by severity.
5. If there are no findings, say so explicitly and mention any residual risks,
   such as "not compiled in this session."

## Output Style

- Findings first, with file references.
- Keep summaries brief.
- Focus on bugs, regressions, stale docs, misleading explanations, and missing
  educational context before style-only suggestions.
