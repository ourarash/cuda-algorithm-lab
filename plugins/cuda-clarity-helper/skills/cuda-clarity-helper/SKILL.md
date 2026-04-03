---
name: cuda-clarity-helper
description: Improve CUDA file readability in cuda-algorithm-lab by adding or refining comments, clarifying naming and structure, and ensuring each file begins with a clear intent statement plus a high-level algorithm summary.
---

# CUDA Clarity Helper

Use this skill when the user wants a CUDA file or folder made easier to read and
teach from.

## Primary Goals

1. Add or improve the top-of-file explanation
   - Start with what the file is trying to demonstrate.
   - Explain why this version exists in the progression.
   - Summarize the high-level algorithm in plain language before diving into
     code details.

2. Improve readability inside the file
   - Add comments that explain intent, data flow, memory behavior, and tricky
     indexing.
   - Avoid noisy comments that merely restate syntax.
   - Prefer short comment blocks before logically dense sections.

3. Preserve the teaching style of this repository
   - Keep comments accessible and direct.
   - Favor explanation of GPU concepts like coalescing, shared memory reuse,
     bank conflicts, warp behavior, scan/reduction trees, and work per thread.
   - When files are part of numbered progressions, make clear what improves over
     the previous step.

4. Keep the code honest
   - Do not let comments drift away from the real implementation.
   - If the code and comment disagree, fix the code or rewrite the comment.

## Workflow

1. Read the entire target file before editing.
2. Identify the file's teaching purpose and algorithmic role.
3. Add or rewrite the top-of-file summary first.
4. Add concise comments only where they improve comprehension.
5. If helpful, tighten naming or local structure while preserving behavior.
6. Keep edits consistent with nearby files in the same folder progression.

## Style Rules

- Prefer a file header that includes:
  - a title
  - intention
  - high-level algorithm
- Keep comments concrete and educational.
- Do not flood the file with line-by-line narration.
- Prefer comments like:
  - why this memory access pattern matters
  - why synchronization is required here
  - what each phase of the kernel is doing
  - how this differs from the earlier version

## Good Outcomes

- A new reader can understand the point of the file from the first 15-25 lines.
- Dense kernels become easier to follow without becoming cluttered.
- Numbered examples read like a coherent progression instead of isolated code dumps.
