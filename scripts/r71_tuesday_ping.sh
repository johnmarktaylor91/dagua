#!/usr/bin/env bash
# Fires Tue 2026-06-16 18:05 ET: reminds JMT to resume r71 if the session-scheduled
# resume didn't fire (belt and suspenders; removes its own crontab line after firing).
~/.claude/scripts/send-to-jmt.sh "r71 RESUME TIME (Tue 6pm ET, tokens back). If Claudio has not resumed on his own in the dagua session, say: resume r71 -- full procedure is in .project-context/research/sprint_rng_matching/r71_fidelity_completion_STATE.md (PAUSED block). Weekend compute summary: eval_output/fidelity_definitive/r71_weekend_summary.json"
crontab -l 2>/dev/null | grep -v r71_tuesday_ping | crontab -
