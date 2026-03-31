---
name: improve
description: Multi-agent codebase improvement pipeline (project override).
user_invocable: true
---

# /improve -- Dagua Override

Follows the global pipeline at `~/.claude/skills/improve/SKILL.md` with these
project-specific quality gates:

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```
