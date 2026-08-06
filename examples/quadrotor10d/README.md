# Decomposed 10D Near-Hover Quadrotor

Single reference for the math, the settings, and the reasoning behind this
benchmark. Code lives in `quad_config.py` (all numeric settings),
`odp/dynamics/QuadrotorHover.py` (the 3 subsystem dynamics classes),
`solve_subsystem_{x,y,z}.py` (solve scripts), and `reconstruct.py`
(evaluate the reconstructed 10D value function).

**Source paper:** Mo Chen, Sylvia L. Herbert, Claire J. Tomlin,
*"Decomposition of Reachable Sets and Tubes for a Class of Nonlinear
Systems,"* IEEE Transactions on Automatic Control, vol. 63, no. 11, 2018
(arXiv:[1611.00122](https://arxiv.org/abs/1611.00122)). The 10D near-hover
quadrotor is their example in **Section VII-B, Eq. 61-64**. Every mention of
"the paper" or "the source paper" below refers to this one. All later
"leaking-corner" / threshold-strategy references are a *different*, follow-up
paper — cited explicitly where used (Section 6).

---

## 1. Units

**SI throughout. Not cm, not degrees.**

| quantity | unit |
|---|---|
| position (px, py, pz) | meters (m) |
| velocity (vx, vy, vz) | m/s |
| angle (theta, phi) | radians |
| angular rate (q, p) | rad/s |
| control u_theta, u_phi | rad/s² |
| control u_z | m/s² |

This isn't a convention choice, it's forced by the dynamics: `g = 9.81` is
in m/s², and it appears directly in `dot(vx) = g*theta` — for that equation
to be dimensionally consistent, `vx` must be in m/s and `px` in meters. If
you fed in centimeters, gravity's effect would be 100x too weak relative to
position.

---

## 2. Full 10D system

State (near-hover linearization around the origin):

```
x = (px, vx, theta, q,   py, vy, phi, p,   pz, vz)
```

| symbol | meaning | unit |
|---|---|---|
| px, py, pz | position | m |
| vx, vy, vz | velocity | m/s |
| theta | pitch angle | rad |
| q | pitch rate | rad/s |
| phi | roll angle | rad |
| p | roll rate | rad/s |

Dynamics (g = 9.81 m/s²):

```
dot(px) = vx              dot(py) = vy              dot(pz) = vz
dot(vx) = g * theta       dot(vy) = -g * phi         dot(vz) = u_z
dot(theta) = q            dot(phi) = p
dot(q) = u_theta          dot(p)  = u_phi
```

`g*theta` is the small-angle approximation of `g*tan(theta)` used in the
source paper's model (Eq. 61 — see Section 4 below for why this matters for
the grid). There is no disturbance in this benchmark (pure reachability
under control, no adversary/wind) — the source paper's model has wind
disturbance terms `dx,dy,dz`; this baseline omits them.

---

## 3. Decomposition: 10D -> 4D + 4D + 2D

Nothing in `dot(px),dot(vx),dot(theta),dot(q)` depends on `py,vy,phi,p,pz,vz`
or on `u_phi,u_z` — and symmetrically for the other two axes. So the state
splits into 3 **self-contained subsystems (SCS)** that share no state and no
control:

```
x1 = (px, vx, theta, q)   4D, control u_theta   -> odp/dynamics: QuadrotorHoverX4D
x2 = (py, vy, phi,   p)   4D, control u_phi     -> odp/dynamics: QuadrotorHoverY4D
x3 = (pz, vz)             2D, control u_z       -> odp/dynamics: QuadrotorHoverZ2D
```

**Why this matters (exactness):** because the subsystems share no state,
control, or disturbance, the theory in the source paper (Chen, Herbert &
Tomlin 2018, Proposition 4 / Section IV-V) guarantees that solving each
subsystem's HJ PDE independently and reconstructing via

```
V(x) = max( Vx(x1), Vy(x2), Vz(x3) )
```

is the **exact** value function of the full 10D coupled system — not an
approximation. This is only true for the `independent_control` configuration
(Section 6). Cost-wise, this is what makes the problem tractable at all:
solving 10D directly at even a modest 41 points/dim would be `41^10 ≈ 1.3e16`
grid points; the 3 decomposed subsystems total `2*41^4 + 81^2 ≈ 5.7e6` points
— an ~11 order-of-magnitude reduction.

**Target set** (rectangle around hover, per subsystem):

```
T1 = { |px|<=r1, |vx|<=r2, |theta|<=r3, |q|<=r4 }
T2 = { |py|<=r1, |vy|<=r2, |phi|<=r3,   |p|<=r4 }
T3 = { |pz|<=r5, |vz|<=r6 }
T  = T1 ∩ T2 ∩ T3   (intersection, i.e. max of the 3 implicit surface functions)
```

Note: this is a different target *shape* than the source paper's example
(Eq. 63), which only bounds position (`|px|,|py|<=1 m, |pz|<=2.5 m`,
velocity/angle/rate unconstrained — a "stay in a box" safety framing). Ours
bounds **all 4** components per subsystem (a "recover near hover" framing).
The exactness argument above doesn't depend on which target shape you pick,
so this substitution is fine, but the two problems aren't numerically
comparable — don't compare our reachable-set sizes to numbers reported in
the paper.

---

## 4. Current numeric settings (`quad_config.py`)

### Grid / target / resolution

All distances in meters, angles in radians (Section 1). Target shape is an
axis-aligned box (L∞/Chebyshev box via `ShapeRectangle`), not a ball —
`|state_i| <= target radius` independently in each dimension.

| subsystem | dim | unit | domain (±) | target radius (±) | grid points/dim | dx | target span in grid points |
|---|---|---|---|---|---|---|---|
| X: px | 4D | m | 2.0 | 0.2 | 41 | 0.100 | 4.0 |
| X: vx | 4D | m/s | 2.0 | 0.2 | 41 | 0.100 | 4.0 |
| X: theta | 4D | rad | **0.3** | 0.2 | 41 | 0.015 | 26.7 |
| X: q | 4D | rad/s | 2.0 | 0.2 | 41 | 0.100 | 4.0 |
| Y: (py,vy,phi,p) | 4D | m, m/s, rad, rad/s | same as X | same as X | 41 | same as X | same as X |
| Z: pz | 2D | m | 2.0 | 0.2 | 81 | 0.050 | 8.0 |
| Z: vz | 2D | m/s | 2.0 | 0.2 | 81 | 0.050 | 8.0 |

Time horizon: `LOOKBACK_LENGTH_XY = LOOKBACK_LENGTH_Z = 3.0 s` for **all**
3 subsystems (kept as two separate constants in `quad_config.py` for
flexibility, but they must be set to the same value). This is required, not
optional: `V(x) = max(Vx,Vy,Vz)` is only the joint system's BRT at a single
horizon T if every subsystem was solved over that same T -- and it's the
horizon that must match `--tMax` in any DeepReach model trained on the full
10D system for the two to be comparable at all. `T_STEP = 0.05 s` for all
3 (the PDE's actual internal integration step is finer, set automatically
by the CFL condition — `T_STEP` only controls how often the solver
checkpoints).

3.0s was chosen because 1.0s was too short for the X/Y chains (4th-order/
underactuated) to converge (~38% domain coverage at 3.0s vs. barely-expanded
at 1.0s). At 3.0s, Z (a direct 2nd-order integrator with strong control
authority relative to its small domain) reaches **100% domain coverage
under both control configs** -- fully saturated, no visible BRT boundary in
a raw Vz plot, and no visible difference between `independent_control` and
`shared_l2_control` in that plot. This is a real property of the system at
T=3.0s (Z is simply not the limiting subsystem at this horizon), not a bug
to fix by shortening Z's horizon -- doing that once already, to make the
plot look better, silently broke the reconstruction's correctness (see git
history). If you want a nicer-looking illustrative Z plot, generate it
separately and label it clearly as a different, shorter, non-comparable
horizon -- never by changing `LOOKBACK_LENGTH_Z` in this file.

### Control bounds

| config | ux (u_theta) | uy (u_phi) | uz | exact reconstruction? |
|---|---|---|---|---|
| `independent_control` | ±2.0 | ±2.0 | ±3.0 | yes |
| `shared_l2_control` | ±2.0 (projected) | ±2.0 (projected) | ±2.0 (projected) | **no** (Section 6) |

### Reasonableness check (asked to verify this — here's the audit)

- **theta domain (±0.3 rad ≈ 17°) is deliberately tight**, not arbitrary: the
  dynamics use `g*theta`, the small-angle approximation of `g*tan(theta)`.
  At 0.3 rad the approximation error is ~3%; the source paper's domain is
  unconstrained because *they* use the exact `tan(theta)` term (Eq. 61), so
  this substitution is the correct tightening for a linearized model.

- **theta target-to-domain margin is thin: only ±0.10 rad of buffer** between
  the target boundary (0.2) and the domain edge (0.3). This caps how far the
  BRT can be seen to expand in that dimension before hitting the (artificial)
  grid boundary — the true reachable set in theta may extend further than
  what this grid can show. If you want to see the full expansion, either
  shrink the target's theta tolerance (e.g. 0.1) or accept that theta
  results near the domain edge are boundary-truncated, not physical. I did
  not change this without checking with you since the target tolerance is
  an experiment-design choice, not a pure numerics one.

- **px, vx, q are resolved at only ~4 grid points across the target** —
  on the low side of the "5-10+ points" rule of thumb for well-resolved
  ENO/WENO boundaries. theta (26.7 pts) and pz/vz (8 pts) are comfortably
  resolved. If you want tighter resolution here, raising `GRID_N_X`/`GRID_N_Y`
  further will cost roughly `N^4`-ish compute (current 41⁴ already takes
  ~35s per subsystem on the head node; expect a large jump in runtime and
  likely need `sbatch` rather than interactive use).

- **Control bounds** are in the right ballpark versus the source paper: its
  effective peak angular acceleration is `n0*Sx_max = 10 * 10° ≈ 1.75 rad/s²`
  (before its internal `-d0*theta` feedback term, Eq. 61, d0=10,n0=10),
  close to our `u_theta=2 rad/s²`. `u_z=3 m/s²` is a deliberately gentle
  near-hover vertical authority (the paper's real thrust channel spans up to
  `2g-g ≈ 8 m/s²` net, with `Tz in [0, 2g]` and `dot(vz)=kT*Tz-g`).

- **1.0s horizon**: reasonable for a quick check, plausibly short for the
  full picture — the X/Y chains are 4th-order/underactuated (px <- vx <-
  theta <- q <- u_theta) and expand much more slowly than Z's direct double
  integrator (confirmed empirically: `pz=0.5` became reachable within 1s,
  `px=0.5` did not). A longer horizon (2-3s) would show more of the X/Y BRT
  growth, at proportionally higher compute cost.

---

## 5. Internal logic (how a solve script actually works)

Each `solve_subsystem_*.py` does 4 things using **only existing ODP APIs**
(no changes to `odp/solver.py` or the computeGraphs):

1. Build a `Grid` for that subsystem from `quad_config.py` bounds/resolution.
2. Build the target set with `ShapeRectangle(grid, target_min, target_max)`.
3. Instantiate the subsystem's dynamics class (`QuadrotorHoverX4D` etc.) with
   `uMode="min"` (control tries to *reach* the target) and the control bound
   for the requested `--config`.
4. Call `HJSolver(..., compMethod={"TargetSetMode": "minVWithV0"},
   saveAllTimeSteps=True)`, which computes the backward reachable tube (BRT)
   — the union over the time horizon of all backward reachable sets (BRS).
   The result is saved to `results/<config>/V{x,y,z}.npy`.

   Gotcha (bit us during development): `HJSolver`'s saved array is indexed
   **backwards in time** — index `0` is the fully-propagated final result,
   index `-1` is the untouched t=0 target. All 3 scripts use `result[..., 0]`.

`reconstruct.py` then loads the 3 saved arrays and, for any 10D query state,
projects it onto `(x1,x2,x3)`, looks up each subsystem's value via the
existing `Grid.get_values` (nearest-neighbor interpolation), and returns
`max(Vx,Vy,Vz)`. No 10D array is ever built.

---

## 6. Shared control extension (`shared_l2_control`)

The baseline above uses **independent** control bounds (a Cartesian product
`|ux|<=ux_max, |uy|<=uy_max, |uz|<=uz_max`) — subsystems don't compete for a
control budget, which is exactly the condition that makes the decomposition
exact (Section 3).

The shared extension replaces this with a single **coupled L2 ball**:

```
u = (ux, uy, uz),   ux^2 + uy^2 + uz^2 <= U_MAX^2
```

Now the 3 axes *do* compete for a shared thrust/control budget (physically:
a real rotor's total control authority is finite and shared across pitch,
roll, and thrust commands — this is closer to reality than 3 independent
budgets).

**Each subsystem is still solved independently**, using the exact projection
of the L2 ball onto its own axis, `ui in [-U_MAX, U_MAX]` — this is what
"projected control set" means: it's the largest interval that axis could
ever reach *if the other two axes contribute nothing* (e.g. `ux=U_MAX,
uy=uz=0` does satisfy `ux^2<=U_MAX^2`). The dynamics and solve mechanics are
otherwise identical to Section 5.

**Why the reconstruction is only an approximation:** `V=max(Vx,Vy,Vz)<=0`
requires **all three** `Vx<=0`, `Vy<=0`, `Vz<=0` simultaneously — i.e. it
implicitly assumes there's a single joint control trajectory `u(t)` under
which all 3 subsystems simultaneously follow their own independently-optimal
(typically bang-bang) control law. But each subsystem's optimal control was
computed *as if it alone had the full budget*; if, at some time t, subsystem
X wants `ux=U_MAX`, Y wants `uy=U_MAX`, and Z wants `uz=U_MAX`
simultaneously, the combined vector `(U_MAX,U_MAX,U_MAX)` has norm
`U_MAX*sqrt(3) > U_MAX` — **infeasible** under the true L2 constraint, even
though each component individually satisfies its own projected interval.

This is the **leaking-corner** approximation: extra states "leak into" the
reconstructed reachable set at the corners of the Cartesian product region
that the true coupled ball doesn't actually contain. Concretely, the
reconstructed set can be an **over-approximation** — it may claim some
states are recoverable when they actually aren't under the true shared
constraint. `reconstruct.py` labels this explicitly:

```python
recon = Quadrotor10DReconstructedValue("shared_l2_control")
# prints: "-> reconstructed value is APPROXIMATE (leaking-corner)"
recon.exact   # False
```

vs.

```python
recon = Quadrotor10DReconstructedValue("independent_control")
# prints: "-> reconstructed value is EXACT"
recon.exact   # True
```

Fixing the leaking-corner gap (making the shared-control reconstruction
exact, or bounding the approximation error) is exactly the subject of a
different, follow-up paper: Chong He, Mugilan Mariappan, Keval Vora, Mo Chen,
*"Threshold Strategy for Leaking Corner-Free Hamilton-Jacobi Reachability
with Decomposed Computations,"* submitted to CDC 2025
(arXiv:[2505.10020](https://arxiv.org/abs/2505.10020)). Implementing that
strategy is out of scope for this benchmark — the leaking-corner gap is
intentionally left here as the documented difference between the two
configurations.
