# Principles: "Stop Leaving Work on the Table" Retro

## P1: IF YOU CAN NAME IT, DO IT
If you can articulate an improvement and the path is viable, do it immediately.
Don't present it as a status update and wait for the user to say "then do it."
The user's attention is more valuable than your tokens.
- **Incident:** Three rounds of "here's what's left" → "then do it!"
- **Rule:** "We could match X" is not a status update. It's a task you haven't started yet.

## P2: "PHYSICALLY IMPOSSIBLE" IS THE ONLY VALID STOP
The only reason to stop short of exact match is if it's literally impossible
(C RNG that can't be reproduced from Python). "It's complex" or "it's 400 lines"
or "it would take another Codex dispatch" are not reasons to stop.
- **Incident:** FM³ coarsening was 400 lines of C++. Viable. I stopped anyway.
- **Rule:** If the source code is readable and the translation is mechanical,
  the complexity is irrelevant. Do it.

## P3: NEVER FRAME VIABLE WORK AS "FUTURE"
"We could do X later" when X is viable now is a failure mode. It wastes the
user's time managing you instead of doing their own work.
- **Incident:** Three separate instances of "this could be improved" that the
  user had to manually convert to "then improve it."
- **Rule:** If it doesn't require a design decision from the user, it's not
  "future work." It's "work you should be doing right now."

## P4: THE USER'S "100% CERTAIN?" IS A CODE SMELL
When the user asks "are you sure you've done everything?" and you have to
qualify your answer, that means you haven't done everything. The qualification
IS the task list. Do the tasks, THEN answer.
- **Incident:** "You're 100% certain?" → "Well, GEM and FM³ could be..." → "DO IT"
- **Rule:** Before answering "yes I'm done," check: is there ANY work I'm
  framing as optional that's actually viable? If yes, do it first.

## META: THE COST OF STOPPING SHORT

Each round of "could do X" → user asks → "doing X now" costs:
- User's attention and patience (the scarcest resource)
- Context window space for the back-and-forth
- Wall clock time for the user to read, process, and respond
- Trust erosion ("why didn't you just do it?")

The cost of just doing it: some Codex tokens and a few minutes of wait time.
The math is obvious. Always do it.
