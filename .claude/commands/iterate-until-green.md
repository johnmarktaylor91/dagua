Run the following command and iterate until it passes:

```
$ARGUMENTS
```

Rules:
1. Run the command
2. If it fails, analyze the ROOT CAUSE (not just the symptom)
3. Fix the issue — dispatch to Codex if it's implementation work
4. Re-run the command
5. If a different error appears, fix that too
6. Continue until fully passing or you've hit 5 consecutive failures on the same issue
7. Do NOT ask for permission between iterations — just keep going
8. Do NOT band-aid. If the same fix isn't working, step back and rethink the approach.
9. After success, report: what failed, what you fixed, how many iterations it took
10. After persistent failure, report: what you tried, what you think is actually wrong, proposed next steps

Update `.project-context/knowledge/gotchas.md` with anything you learned.
