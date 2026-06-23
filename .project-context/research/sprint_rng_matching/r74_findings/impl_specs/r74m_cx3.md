Read /tmp/r74m_cx_shared.md. AXIS CX3 -- the TOST/IUT equivalence BATTERY, the MARGINS, and rung-4-vs-3Q
logic. Findings -> /tmp/r74m_CX3.md. How is 3Q decided (which p-values, AND/IUT, BH correction)? What are
the exact equivalence margins for stress/crossings/np? Are they tighter than the REFERENCE's own
seed-to-seed variance (compute the reference within-engine variance and compare to the margin -- if margin
< ref self-variance, the test is mis-calibrated: the reference vs itself couldn't pass)? Is there a
positive control proving the battery CAN certify a known-equivalent pair (e.g. reference vs itself on a
disjoint seed split)? If absent, that's a critical validation gap -- the battery's sensitivity is
unverified. At defensible (variance-tied) margins, how many of the 574 would pass? Distinguish genuine
quality gaps (large scale-INVARIANT stress, e.g. sugiyama) from margin-artifact floor.
