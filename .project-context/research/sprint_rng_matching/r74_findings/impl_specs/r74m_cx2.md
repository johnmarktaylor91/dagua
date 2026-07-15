Read /tmp/r74m_cx_shared.md. AXIS CX2 -- the STRESS metric + SCALE/normalization/alignment. Findings ->
/tmp/r74m_CX2.md. There may be MORE THAN ONE stress function (a diagnostic one and a battery one). Read
both. Is the stress used INSIDE the 3Q equivalence battery scale-normalized (does it fit an optimal scale
alpha before the residual) or computed on RAW coordinates? Is there Procrustes/scale alignment first? If
the battery stress is un-scale-normalized, a pure dagua-vs-reference coordinate SCALE difference (recall
OGDF-scale issues) would inflate the battery stress difference with ZERO quality difference. VERIFY by
tracing specific rung-4 combos: compare the scale-INVARIANT stress (D vs R) to the BATTERY stress (D vs R)
-- if invariant says equivalent but battery says huge, it's a scale artifact. Quantify how many of 574
show this signature. Propose the exact fix (where to add alpha). Do not relax margins (laundering).
