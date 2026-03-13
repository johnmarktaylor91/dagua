The user is stepping away. This is downtime — use it well.

Run the following in sequence, dispatching via background tasks:

1. **Full test suite** (Tier 3): `ruff check . && mypy src/ --strict && pytest tests/ -v`
2. **Check for any pending items** in `.project-context/todos.md`
3. **Update knowledge files**: review recent changes and update gotchas.md, architecture.md if needed
4. **Clean old tasks**: `./scripts/clean-tasks.sh 7`

Dispatch the test suite immediately. Report results via ntfy when done.

If $ARGUMENTS contains additional tasks, run those too.

Do NOT wait for user input. Run everything autonomously. Results will be waiting when they return.
