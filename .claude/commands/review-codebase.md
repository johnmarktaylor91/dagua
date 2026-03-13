Review this codebase comprehensively. For each module:

1. Read through the code systematically
2. Look for: bugs, edge cases, missing error handling, type errors, dead code, inconsistencies, performance issues
3. Check that type hints and docstrings are present on all functions
4. Verify test coverage for key logic paths

After review:
- Update `.project-context/knowledge/gotchas.md` with any new findings
- Update `.project-context/architecture.md` if your understanding evolved
- Update `.project-context/todos.md` with actionable items

Present findings organized by severity (critical → low). For mechanical fixes, propose dispatching them to Codex. For design questions, discuss with me.

$ARGUMENTS
