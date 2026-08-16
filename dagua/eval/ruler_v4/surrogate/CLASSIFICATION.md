# 6.5 surrogate: 91-row differentiability classification

One row per scored sub-term in the frozen manifest (91 rows over 45
contracts). `class` is the closed-form disposition the P3FIX sweep
established (DISCREPANCIES entry 38's rule: contract-named smoothings
only, documented zero-gradient exemptions otherwise, invented softeners
banned):

- **naturally-smooth** -- the frozen closed form is continuous in
  positions off declared event manifolds (CC-1) with no named smoothing
  device: distances, angles, stresses over isotonic fits (the PAVA
  projection is piecewise-linear and a.e. differentiable), analytic
  areas/integrals with detached partitions.
- **contract-smoothed** -- the frozen closed form itself names the
  smoothing class with a pinned temperature (sigmoids, softplus/soft_pos
  hinges, LSE/softmin, smoothstep fades; the load-bearing citations are
  DISCREPANCIES entry 38's list).
- **irreducibly-discrete** -- the scored quantity is a hard count or
  rank structure the contract declines to smooth; the row carries a
  documented zero-gradient exemption with its citation, never an
  invented softener.

`channel` records where the row's position gradient actually flows in
THIS seam version (traced.py's channel inventory): `live` (gradient
reaches the position leaf on generic scenes; a.e.-flat plateaus are
noted), `detached-channel` (the closed form is smooth but its only
position path crosses a float seam this version -- honest exact-value
constant, goes live when that seam does), `input-owned` (the row reads
no drawn positions at all: style constants, temporal history, declared
weights), `diagnostic` (weight-0 gate rows, traced honestly, never
bound into l_total).

Verified by `tests/eval/ruler_v4/test_surrogate_manifest.py`
(completeness: every scored sub-term classified exactly once, enum
valid) and by the traced-facet tests per module (liveness pins).

**Scope: the `channel` column is conditional on the straight-route
model.** The frozen reconstruction contract fixes the whole drawing pool
to `bends: 0, curvature: 0, declared route style: straight`
(SCENE_RECONSTRUCTION_CONTRACT.md sec 9a), and `build_traced_scene`
re-links a route to the position leaf only when it is exactly the chord
of its endpoints -- so on the certified population 100% of routes are
live. Under a future profile that admits bends, non-chord routes become
input-owned constants SILENTLY, and measured on the acceptance fixture
that converts five now-live rows to constants (U11.v, U13.i,
U31.headline, U34.L_back, U34.L_mono -- the two U34 rows carry entry
38's flagship softplus citation). A bend-admitting profile must re-run
the liveness sweep and re-derive this column before trusting it.

| subterm | class | channel | citation / note |
|---|---|---|---|
| U01.headline | naturally-smooth | live | U01.md:15 isotonic stress; zero-residual strata take the exact-zero arm (e529778b guard) |
| U01b.local | naturally-smooth | live | U01 machinery on distance bands; NA where bands underpopulate |
| U01b.long | naturally-smooth | live | U01 machinery on the long band |
| U02.headline | irreducibly-discrete | diagnostic | U02.md:135 retraction keeps hard midranks; sec-15a tau_r (U02.md:146) promotion path NOT adopted; zero-gradient exemption |
| U03.r_1 | contract-smoothed | live | sigmoid credit tau=0.25 (U03.md:76-77); collapse-gate smoothstep eps_nb=0.1 (U03.md:90-91); kthvalue pick detached, picked element gathered live |
| U03.r_2 | contract-smoothed | live | same closed form at radius 2 |
| U03.r_4 | contract-smoothed | live | same closed form at radius 4; ineligible on small scenes |
| U04a.2u | contract-smoothed | detached-channel | U04a.md:199-205 closed form; polygon-union grid machinery is the float path this version |
| U04a.8u | contract-smoothed | detached-channel | same at 8u scale |
| U04b.part_1 | contract-smoothed | detached-channel | U04b.md:69-71; same float grid machinery |
| U04b.part_2 | contract-smoothed | detached-channel | same |
| U05.headline | naturally-smooth | live, diagnostic | U05.md:62-63 adopts U01 machinery verbatim; shared isotonic seam |
| U06.headline | contract-smoothed | live, diagnostic | spread gate U06.md:95-96 + LSE smoothed max at the global temperature U06.md:109 |
| U7.base | naturally-smooth | live | severity integrand over a detached crossing-event set; exact 0 constant on zero-event scenes (U07.md sec 7) |
| U7.tail | contract-smoothed | live | excess-severity hinge (d_e - s0)_+ per U07.md |
| U08.headline | contract-smoothed | live | confidence-faded LSE-max, tau=0.1 (U08.md) |
| U09.headline | naturally-smooth | live | MAD dispersion map U09.md:83-92 via the shared route_lengths seam; flat where MAD sits on an exact-zero deviation |
| U10.headline | naturally-smooth | live | zero-onset clearance-deficit band integral (U10.md sec 5a partition detached, integrand continuous); exact-0 constant out of band |
| U11.i | contract-smoothed | live | event-monotone sum + backtracking sigmoid (U11.md); anchored zero on straight chords |
| U11.ii | contract-smoothed | live | hz_tau=0.05 hinge (U11.md:179-181) + style sigmoids; dropped where no style is declared |
| U11.iii | contract-smoothed | live | softmin-LSE baseline, t_soft = 0.1*chord_e (U11.md:315-326, :204); Dijkstra decision detached, path length live |
| U11.iv | contract-smoothed | live | counterflow sigmoid (U11.md); flat where flow agrees |
| U11.v | naturally-smooth | live | Gaussian gap/tangent confusability (U11.md) |
| U12.headline | naturally-smooth | live | live secant angles + shared smooth confidence fade (U12.md) |
| U13.i | naturally-smooth | live | compact-support C1 kernel integral (U13.md); envelope partition detached, coefficients live |
| U13.ii | contract-smoothed | live | smooth-onset bundle terminal deficit via 10%-arc departure points (U13.md) |
| U14.headline | contract-smoothed | live, diagnostic | compact kernel + smoothstep achievement + logistic blend U14.md:68-93 |
| U15.i | contract-smoothed | detached-channel | relative-separation deficit + m_pair fades (U15.md); parallel non-chord routes are input-owned constants this version |
| U15.ii | naturally-smooth | live | U10 clearance integral through other nodes' boxes (U15.md) |
| U16.i | contract-smoothed | live | overlap fades (U16.md); label boxes constant-channel, live through node boxes and chord routes |
| U16.ii | contract-smoothed | live | ownership/anchoring fades (U16.md); same channels |
| U17.1 | naturally-smooth | live | clearance/overlap chain (U17.md) |
| U18.ll | contract-smoothed | live | label-label clearance achievement H (U17.md sec 6c via U18.md sec 6) |
| U18.ln | contract-smoothed | live | label-node clearance, same closed form |
| U18.le | contract-smoothed | live | label-route clearance; H(x>=1)=1 plateau has exactly zero derivative (honest flat beyond budget) |
| U19.headline | naturally-smooth | detached-channel | NA on every v4.0 profile; only position channel is the robust frame, and U19 is unexercised, kept detached |
| U20a.i | contract-smoothed | live, diagnostic | U20a.md sec 6 plateau construction ("sit-at-plateau is desired; no cliff"); live inside the f_min band |
| U20a.ii | contract-smoothed | live, diagnostic | same |
| U20a.iii | contract-smoothed | live, diagnostic | same |
| U20b.headline | contract-smoothed | live, diagnostic | median-shoulder logistic (U20b.md sec 6) |
| U21.d_sparse_n | contract-smoothed | live | soft_pos over log(A_fr/(4 A_ref)) (U21.md closed form); rides RobustFrame.area through the live frame seam; zero plateau on compact drawings |
| U21.d_overflow | contract-smoothed | detached-channel | smooth-onset excess^2/(excess+0.05) (U21.md); frames.overflow_defect is a float pipeline this version |
| U22.headline | contract-smoothed | detached-channel | soft_pos hinge U22.md:105-108; frames.robust_projection detaches positions this version |
| U23.headline | contract-smoothed | live | quintic smoothstep beta_bal=0.5 (U23.md:103-104); centroid channel live; zero-derivative knot at balance |
| U24.headline | contract-smoothed | live | soft_pos (U24.md:147); route-length chain live; kappa_ink=3 plateau flat |
| U25.headline | contract-smoothed | live | soft_pos over log radius ratios + smooth fades (U25.md); median/quantile picks detached, picked elements live; anchored zero on compact clusters |
| U26.i | contract-smoothed | live | separation margin smooth fade against the 0.15*sqrt(f) target (U26.md) |
| U26.ii | contract-smoothed | live | matched-stratum contrast, same fade family (U26.md); drops typed where no matched control stratum exists |
| U26.iii | contract-smoothed | live | community-faithfulness blend; smoothed-max component of the global 3.7 blend (U26.md:240) |
| U27.i | contract-smoothed | live | alpha-blended absolute/excess intrusion (exp onset + smooth fade, U27.md); alpha grid is input-owned |
| U27.ii | contract-smoothed | live | leave-one-out member-escape fade (U27.md); exact zero for interior members |
| U27.iii | contract-smoothed | live | route inside-fraction fade over 0.25 band (U27.md); saturated fades (fully inside/outside) have exactly zero derivative |
| U28.i | contract-smoothed | live | child-outside-parent depth fade (U28.md); exact zero under clean containment |
| U28.ii | contract-smoothed | live | parent coverage economy (adaptive-Simpson region areas live, U28.md) |
| U28.iii | contract-smoothed | live | sibling-overlap area fade (U28.md) |
| U29.headline | contract-smoothed | live | soft_pos over log isoperimetric quotient of the equal-disc union (analytic area/perimeter live, U29.md); exact zero inside the reference band |
| U30.i | contract-smoothed | live | derived-label containment (U30.md; smoothed-max component U30.md:227); label box input-owned, live through region geometry |
| U30.ii | contract-smoothed | live | declared-padding debt fade (U30.md); exact zero at declared padding |
| U30.iii | contract-smoothed | live | occlusion fade (U30.md); saturates near 1 on crowded fixtures |
| U31.headline | contract-smoothed | live | signed-band logistic over chord direction cosines (U31.md); feedback mask and same-rank carve-out detached |
| U32.L_iso | naturally-smooth | live | axis-projection isotonic (PAVA) stress residuals (U32.md) |
| U32.L_crisp | contract-smoothed | live | resolution-gated crispness logistic (U32.md) |
| U32.L_overlap | contract-smoothed | live | margin-shifted overlap logistic (U32.md) |
| U33.layered.1 | naturally-smooth | live | signed clearance projections + penetration depths (U33.md); hull/SAT axis selection detached |
| U33.layered.2 | naturally-smooth | live | smooth centering loss on the node-mass child centroid (U33.md); stationary exactly at centered parents |
| U33.layered.3 | naturally-smooth | live | parent-child unit-direction cosines (U33.md) |
| U33.layered.4 | contract-smoothed | live | sibling order sigmoid((cos70deg - q)/0.03) (U33.md:27); stationary at the antiparallel extremum |
| U33.radial.1 | naturally-smooth | live | traced radii + live PAVA residuals (U33.md); radial scenes only |
| U33.radial.2 | naturally-smooth | live | torch.atan2 sector fractions, detached sort/gap decisions (U33.md); radial scenes only |
| U34.L_back | contract-smoothed | live | sp_tau(x) = tau*log1p(exp(x/tau)), tau = 0.01*mean_segment_length (U34.md:31); live tau |
| U34.L_mono | contract-smoothed | live | same sp_tau closed form on monotone progress |
| U34.L_cont | contract-smoothed | live | same family on junction continuity; population excludes degree-2 junctions (contract), honest float 0.0 where empty |
| U35.headline | naturally-smooth | live | stratum stress over the shared traced PAVA fit (U35.md; PAVA continuous in drawn distances); sqrt kink at zero residual takes the constant arm |
| U36.headline | contract-smoothed | live | margin sigmoid ell_ef = sigmoid(z/0.03) (U36.md); comparison orientation decided on declared strengths (input-owned) |
| U37.ell_e | contract-smoothed | input-owned, diagnostic | StyleContract stroke widths vs GraphSemantics targets (U37.md); no position enters; zero-gradient exemption |
| U37.ell_ord | contract-smoothed | input-owned, diagnostic | same |
| U38.L_clear | contract-smoothed | live | clearance sigmoid through component boxes/routes (U38.md); hard-min pair selection detached, value gathered live |
| U38.L_pack | contract-smoothed | live | log-sigmoid (U38.md); raster cell-count numerator is irreducibly discrete (documented), robust-frame denominator live |
| U38.L_prop | irreducibly-discrete | detached-channel | raster area shares vs input masses (U38.md); zero-gradient exemption, float path preserved |
| U39.1 | naturally-smooth | live | port terminals/anchors ride live node boxes (U39.md); NA where ports absent |
| U39.2 | naturally-smooth | live | tangent alignment (U39.md) |
| U39.3 | naturally-smooth | live | side coordinates (U39.md) |
| U39.4 | naturally-smooth | live | arc-length congestion samples (U39.md) |
| U40.1 | naturally-smooth | input-owned | temporal stability closed form; the temporal scene is input-owned history this version (traced as constant) |
| U40.2 | naturally-smooth | input-owned | same |
| U40.3 | naturally-smooth | input-owned | same |
| U41.L_conv | contract-smoothed | live | reflex sigmoids + exp saturations over live face vertices (U41.md); dedupe/sort/hull membership detached |
| U41.L_area | contract-smoothed | live | balance ratios + exp saturations over signed/hull areas (U41.md) |
| U42.i | naturally-smooth | detached-channel | colour/visibility contrast (U42.md); visibility/backdrop aggregate through polygon-union floats this version |
| U42.ii | contract-smoothed | live | C1 proximity-gate smoothstep for box-box pairs (U42.md); colour deltas input-owned |
| U42.iv | contract-smoothed | live | same gate on the fourth channel |

Tally: naturally-smooth 28, contract-smoothed 61, irreducibly-discrete 2
(U02.headline, U38.L_prop -- both carrying documented zero-gradient
exemptions with citations; U38.L_pack additionally documents its
discrete raster numerator inside a contract-smoothed closed form).

Channel tally this seam version: live 75, detached-channel 10 (U04a.2u,
U04a.8u, U04b.part_1, U04b.part_2, U15.i, U19.headline, U21.d_overflow,
U22.headline, U38.L_prop, U42.i), input-owned 5 (U37.ell_e, U37.ell_ord,
U40.1, U40.2, U40.3), traced-detached discrete 1 (U02.headline, its
zero-gradient exemption documented above). Diagnostic weight-0 rows are
flagged per row; "live" includes rows whose fixture-typical state is an
anchored zero or saturated plateau (exact zero derivative there, live
off it -- flagged in the note).
