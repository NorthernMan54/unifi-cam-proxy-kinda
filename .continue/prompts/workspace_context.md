---
description: Workspace Context & Path Mapping
priority: 100
---

**Project Context:**
This codebase runs inside a Docker container. Error logs and stack traces show container paths starting with `/app/`.

**Important Path Mapping Rule:**
- `/app/` → workspace root (`.`)

**Examples:**
- `/app/unifi/cams/managers/smart_motion_manager.py:66` → `unifi/cams/managers/smart_motion_manager.py`
- `/app/unifi/...` → `unifi/...`

**Instructions for the Agent:**
- Always translate container paths (`/app/...`) to local workspace paths when referencing files.
- When I paste error logs, automatically map the paths.
- When suggesting edits, reading files, or using `@file` references, use the **local workspace paths** (without `/app`).
- Never assume files are located under `/app/` on the host.

This rule applies to all interactions in this workspace.