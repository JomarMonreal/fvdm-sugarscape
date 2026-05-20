# How a Baseline Agent Decides — Step by Step

---

## Overview

Each timestep, every agent faces a single decision:

> **"Which cell should I move to?"**

The agent does not separately decide to fight, trade, or reproduce. It selects a destination cell, and any social interactions that become possible at that location trigger automatically as a consequence. This section traces the full decision procedure from candidate enumeration to final movement.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `φ` | Selfishness factor — 1.0 = pure Egoist, 0.5 = Bentham, 0.0 = Altruist |
| `c` | A candidate cell the agent is considering |
| `h(c)` | The total happiness score assigned to cell `c` |
| `i_k` | Intensity — how urgently agent *k* needs resources right now |
| `d_k` | Duration — how long the cell's resources would sustain agent *k* |
| `i_f` | Future intensity — resource richness of the cell's immediate neighborhood |
| `d_f_k` | Future duration — resources remaining at `c` after agent *k* eats one step's worth |
| `e` | Immediate extent — ratio of neighbors visible from the agent's **current** cell to cells in range |
| `e_f` | Future extent — ratio of neighbors visible from the **candidate** cell to cells in range (varies per candidate) |
| `γ` | Temporal discount factor — weight given to future welfare vs. immediate welfare |

---

## Step-by-Step Decision Procedure

```
AGENT (standing at current position)
│
├─── STEP 1: Enumerate all reachable candidate cells
│
│    The set of candidate cells is determined by:
│
│        cellRange = min(vision, movement)
│
│    In the default cardinal mode, only cells sharing the agent's
│    row or column qualify — up to cellRange steps in any of the
│    four cardinal directions (N, S, E, W). Diagonal and off-axis
│    cells are excluded. In radial mode, all cells within Euclidean
│    distance cellRange are included instead.
│
│    Illustration (cellRange = 2, cardinal mode):
│
│        . . C . .          C  =  candidate cell
│        . . C . .          🧑 =  this agent
│        C C 🧑 C C         .  =  not reachable
│        . . C . .
│        . . C . .
│
│    The agent's current cell is always retained as a fallback
│    candidate in case all other cells are blocked.
│
│
├─── STEP 2: Score each candidate cell c
│
│    For each candidate cell, the agent asks:
│    "How does moving here affect me and every neighbor who
│     could also reach this cell?"
│
│    Setup — the agent considers cell C, with two visible
│    neighbors N1 and N2 in its vision range.
│
│    In cardinal mode, agents can only reach cells in the SAME ROW
│    or SAME COLUMN as themselves. The grid is designed so that:
│      • 🧑 and C share the same column  (col 2)
│      • N1 and C share the same row     (row 0)
│      • N2 is on the same row as C but beyond its own range
│
│        col: 0  1  2  3  4
│    row 0:   N2  .  C  N1  .      ← C, N1, N2 all in row 0
│    row 1:   .   .  .  .   .
│    row 2:   .   .  🧑 .   .      ← 🧑 in col 2, same column as C
│    row 3:   .   .  .  .   .
│    row 4:   .   .  .  .   .
│
│    ┌─────────────────────────────────────────────────────────────┐
│    │ CELL C  — at (col=2, row=0)                                │
│    │   sugar                    =  3                            │
│    │   spice                    =  2                            │
│    │   resources (W_c)          =  sugar + spice  =  5         │
│    │   max sugar                =  5                            │
│    │   max spice                =  3                            │
│    │   capacity  (W_cmax)       =  maxSugar + maxSpice  =  8   │
│    │   pollution                =  0                            │
│    │   adj. neighbor wealth     = 10  (sum of sugar+spice       │
│    │                                   across 4 adjacent cells) │
│    │   global max wealth        = 400  (W_gmax, env-wide)       │
│    ├─────────────────────────────────────────────────────────────┤
│    │ 🧑  SELF — at (col=2, row=2)                               │
│    │   vision / movement range  =  2                            │
│    │   distance to C            =  2  (same col, 2 rows north)  │
│    │   TTL (time to live)       =  3                            │
│    │   sugar metabolism         =  1                            │
│    │   spice metabolism         =  1                            │
│    │   metabolism (m)           =  sugarMetab + spiceMetab  =  2│
│    │   cells in range (|V|)     =  4  (2 north + 2 south,       │
│    │                                   2 east + 2 west)         │
│    │   neighbors in range       =  1  (only N1 is within range) │
│    │   γ (discount)             =  0.5                          │
│    ├─────────────────────────────────────────────────────────────┤
│    │ N1 — at (col=3, row=0)                                     │
│    │   vision / movement range  =  2                            │
│    │   distance to C            =  1  (same row, 1 col west)    │
│    │   TTL                      =  5                            │
│    │   sugar metabolism         =  1                            │
│    │   spice metabolism         =  1                            │
│    │   metabolism (m)           =  sugarMetab + spiceMetab  =  2│
│    │   cells in range (|V|)     =  4                            │
│    │   neighbors in range       =  1  (🧑 is within range)      │
│    ├─────────────────────────────────────────────────────────────┤
│    │ N2 — at (col=0, row=0)                                     │
│    │   vision / movement range  =  1                            │
│    │   distance to C            =  2  (same row, 2 cols east)   │
│    │   → range(1) < distance(2) → CANNOT reach C → skipped     │
│    └─────────────────────────────────────────────────────────────┘
│
│    Reachability check (per person):
│        🧑  range = 2  →  distance to C = 2  →  CAN reach  →  scored
│        N1  range = 2  →  distance to C = 1  →  CAN reach  →  scored
│        N2  range = 1  →  distance to C = 2  →  CANNOT     →  skipped
│
│    ── For each person k who CAN reach cell c ──────────────────────
│
│    a) Immediate welfare components:
│
│           e   = |live neighbors visible from agent's current cell| / |cells in range|
│                 — social density weight from the agent's present position
│
│           i_k = 1 / ( (1 + TTL_k) × (1 + pollution) )
│                 — higher when the agent is close to dying
│                 — higher when the cell carries more pollution
│
│           d_k = (resources / metabolism_k) / capacity
│                 — higher when the cell can sustain the agent longer
│
│    b) Future welfare components (same cell, one timestep later):
│
│           e_f = |live neighbors visible from candidate cell c| / |cells in range|
│                 — social density weight projected from the candidate cell's position
│                 — recomputed per candidate; generally ≠ e
│
│           i_f  = adj. neighbor wealth / (global_max × num_adj_cells)
│                  — higher when surrounding cells are resource-rich
│
│           d_f_k = (resources − metabolism_k) / (metabolism_k × capacity)
│                   — residual resources remaining after one round of eating
│
│    c) Per-person welfare score:
│
│    The happiness h for a given cell is formalised as:
│
│           h = cp [ e(i + d)  +  γ e_f(i_f + d_f) ]
│
│    where certainty (c) and propinquity (p) weigh the overall outcome,
│    immediate rewards are captured by extent (e), intensity (i), and
│    duration (d), and future rewards use future intensity (i_f) and
│    future duration (d_f), discounted by γ to represent uncertainty
│    in long-term predictions.
│
│    This is embedded in a Markov Decision Process (MDP), where the
│    Bellman optimality equation guides decision-making:
│
│           V*(s) = max_a  Σ_s'  T(s,a,s') [ R(s,a,s') + γ V*(s') ]
│
│    The mapping between both formulations is:
│
│    ┌───────────────┬──────────────────────────────────────────────┐
│    │ Bellman term  │ Hedonic calculus equivalent                  │
│    ├───────────────┼──────────────────────────────────────────────┤
│    │ T(s, a, s')   │ cp   — certainty × propinquity               │
│    │               │        (how likely and how immediate)        │
│    │ R(s, a, s')   │ e(i+d) — immediate reward                    │
│    │               │        (extent-weighted intensity+duration)  │
│    │ γ V*(s')      │ γ e_f(i_f+d_f) — discounted future reward    │
│    │               │        (neighborhood richness × residual)    │
│    └───────────────┴──────────────────────────────────────────────┘
│
│    Applied per person k, the full per-person welfare score is:
│
│           h_k = (c_k × p) × [ e × (i_k + d_k)  +  γ × e_f × (i_f + d_f_k) ]
│                               └──── immediate ────┘  └──── future (discounted) ──┘
│
│           where  c_k = 1 if person k can reach c, else 0
│                  p   = 1  (one-step lookahead, always)
│                  e   = |neighbors visible from current cell| / |cells in range|
│                  e_f = |neighbors visible from candidate cell c| / |cells in range|
│                        (e_f is computed fresh per candidate; differs from e)
│
│    Numerical example — 🧑 (self) evaluating cell C:
│    (TTL=3, m=2, |V|=4, 1 neighbor visible from current cell,
│     1 neighbor visible from candidate cell C, γ=0.5)
│
│           i_self = 1 / ((1+3)×(1+0))    =  0.250   (TTL=3, pollution=0)
│           d_self = (5 / 2) / 8           =  0.313   (resources/m / capacity)
│           i_f    = 10 / (400 × 4)        =  0.006   (adj wealth / globalMax×n_adj)
│           d_f    = (5 − 2) / (2 × 8)    =  0.188   (residual resources after eating)
│           e      = 1 / 4                 =  0.250   (neighbors from current cell / |V|)
│           e_f    = 1 / 4                 =  0.250   (neighbors from candidate cell C / |V|)
│                                                     (coincidentally equal here; varies in general)
│
│           h_self = e × (i_self + d_self)  +  γ × e_f × (i_f + d_f)
│                  = 0.250 × (0.250 + 0.313)  +  0.5 × 0.250 × (0.006 + 0.188)
│                  = 0.250 × 0.563            +  0.125 × 0.194
│                  = 0.141 + 0.024
│                  = 0.165
│                  (same result as before because e = e_f in this example)
│
│    Numerical example — N1 evaluating cell C:
│    (TTL=5, m=2, |V|=2, 1 neighbor in range, γ=0.5)
│
│           i_N1   = 1 / ((1+5)×(1+0))    =  0.167   (TTL=5, less urgent than self)
│           d_N1   = (5 / 2) / 8           =  0.313   (same cell, same formula)
│           i_f    = 10 / (400 × 4)        =  0.006   (same cell neighborhood)
│           d_f_N1 = (5 − 2) / (2 × 8)    =  0.188
│           e_N1   = 1 / 2                 =  0.500   (N1 has fewer cells in range)
│
│           h_N1   = 0.500 × [(0.167 + 0.313) + 0.5 × (0.006 + 0.188)]
│                  = 0.500 × [0.480 + 0.097]
│                  = 0.500 × 0.577
│                  = 0.289
│
│    Both h_self = 0.165 and h_N1 = 0.289 now feed into Step 3.
│
│
├─── STEP 3: Apply the selfishness factor φ
│
│    The agent in this example is a BENTHAM agent (φ = 0.5).
│    It now holds h_self = 0.165 and h_N1 = 0.289 from Step 2.
│    φ determines how self-gain and neighbor-harm are weighted:
│
│    ┌──────────────────────────────────────────────────────────────┐
│    │   h(c)  =  φ × h_self  −  (1 − φ) × Σ h_other_k            │
│    └──────────────────────────────────────────────────────────────┘
│
│    Applying the Bentham formula to cell C:
│
│       h(c) = 0.5 × h_self      −  (1 − 0.5) × h_N1
│            = 0.5 × 0.165       −  0.5 × 0.289
│            = 0.083              −  0.145
│            = −0.062
│
│    Cell C scores negative. The Bentham agent's self-gain (0.083)
│    is outweighed by the harm moving here causes to N1 (0.145),
│    because N1 also wants this cell and would lose access to it.
│    The agent will prefer any cell with a higher (less negative) score.
│
│    Visualised as a balance between self-gain and neighbor-harm:
│
│    EGOIST (φ = 1.0)
│      Self  ████████████████   Neighbors  ░░░░░░░░░░░░░░░░
│            full weight                   zero weight
│      → h(c) = h_self  (neighbors ignored entirely)
│      → h(C) = 1.0 × 0.165 = 0.165
│
│    BENTHAM (φ = 0.5)  ← this agent
│      Self  ████████           Neighbors  ████████
│            half weight                   half weight
│      → h(c) = 0.5 × h_self − 0.5 × Σ h_other_k
│      → h(C) = 0.5 × 0.165 − 0.5 × 0.289 = −0.062
│
│    ALTRUIST (φ = 0.0)
│      Self  ░░░░░░░░░░░░░░░░   Neighbors  ████████████████
│            zero weight                   full weight
│      → h(c) = − Σ h_other_k  (own welfare ignored entirely)
│      → h(C) = −1.0 × 0.289 = −0.289
│
│    RAW SUGARSCAPE  (formula bypassed)
│      Skips Steps 2–3. Score = cell sugar + cell spice only.
│      → h(C) = 3 + 2 = 5
│
│    ┌─────────────┬──────┬──────────────────────────────────────────┐
│    │ Agent Type  │  φ   │ h(C) for this example                    │
│    ├─────────────┼──────┼──────────────────────────────────────────┤
│    │ Egoist      │ 1.0  │  0.165  (full self-gain, no penalty)     │
│    │ Bentham     │ 0.5  │ −0.062  (self-gain < neighbor harm)      │
│    │ Altruist    │ 0.0  │ −0.289  (only neighbor harm counts)      │
│    │ Raw Sugar.  │  —   │  5      (sugar + spice, no calculus)     │
│    └─────────────┴──────┴──────────────────────────────────────────┘
│
│
├─── STEP 4: Rank all candidate cells
│
│    Steps 2–3 are repeated for every other candidate cell.
│    Each cell's score depends on which neighbors can reach it —
│    cells contested by nearby agents score lower for the Bentham agent.
│
│    Cell C (north-2) = −0.062  (computed in Steps 2–3 above).
│    Other cells have fewer or no competing neighbors, so they score
│    higher. The full scored map over the reachable cross:
│
│    col:  0     1     2      3     4
│                    -0.06              ← north-2 (cell C, computed)
│                     0.14             ← north-1 (no competing neighbor)
│    0.07  0.11        🧑   0.13  0.09  ← west/east cells
│                     0.12             ← south-1
│                     0.08             ← south-2
│
│    Ranked list (Bentham agent, φ = 0.5):
│        1st  north-1 :  h =  0.14   ← c*  (winner — uncontested)
│        2nd  east-1  :  h =  0.13
│        3rd  west-1  :  h =  0.11
│        4th  south-1 :  h =  0.12
│        5th  east-2  :  h =  0.09
│        6th  south-2 :  h =  0.08
│        7th  west-2  :  h =  0.07
│        8th  north-2 :  h = −0.062  ← cell C, penalised by N1
│
│    The Bentham agent avoids cell C even though it has the most
│    resources (W_c = 5), because moving there displaces N1.
│
│
└─── STEP 5: Move to the highest-scoring cell
│
│    c* = argmax h(c)
│
│    The agent moves directly to c* in a single step —
│    intermediate cells are not visited.
│
│    BEFORE:                            AFTER:
│
│    .  .  c* .  .                      .  .  🧑 .  .
│    .  .  .  .  .     ────────────►    .  .  .  .  .
│    .  .  🧑 .  .                      .  .  .  .  .
│
│    Tie rule: when two cells share the highest score,
│    the closer one is preferred.

NOTE — Automatic consequences at destination:
  Once the agent arrives, the environment checks whether any social
  interactions are now possible and triggers them without further
  input from the agent: combat if the cell was occupied, reproduction
  if a compatible mate is adjacent, trade if a trade partner is
  adjacent, and loan settlement if a lending condition is met.
  The agent's decision ends at Step 5; everything that follows is
  a mechanical consequence of where it chose to stand.
```

---

## How Agent Type Changes the Outcome

The scoring formula in Steps 2–3 is identical for all hedonic agents. Only φ differs, and that single parameter produces qualitatively distinct behaviours:

```
EGOIST  (φ = 1.0)
  Neighbour welfare contributes nothing to h(c).
  The agent pursues whichever cell maximises its own gain,
  including occupied cells whose resources can be looted.

BENTHAM  (φ = 0.5)
  Self-gain and collective harm are weighted equally.
  The agent avoids entering cells that impose large costs on
  many neighbours, but will compete if its own gain outweighs
  the total harm caused.

ALTRUIST  (φ = 0.0)
  Own welfare is excluded from h(c).
  The agent selects whichever cell minimises the summed welfare
  loss of its neighbours, even at the cost of its own survival.
  Occupied cells receive strongly negative scores because
  displacing an occupant always harms someone.

RAW SUGARSCAPE  (no hedonic formula)
  Skips the welfare calculation entirely.
  The agent selects the cell with the highest raw resource level
  (sugar + spice), with no ethical weighting whatsoever.
```

---

## Worked Example

**Agent parameters:** metabolism = 2, TTL = 3, vision/movement range = 4, 2 neighbours in range, γ = 0.5

**Candidate cell:** resources = 5, capacity = 8, pollution = 0, adjacent neighbour wealth = 10 across 4 adj. cells

```
IMMEDIATE WELFARE (self):
  i = 1 / ((1+3)(1+0))  =  0.25
  d = (5/2) / 8          =  0.31

FUTURE WELFARE (self):
  i_f = 10 / (400 × 4)  ≈  0.006   (resource-poor neighbourhood)
  d_f = (5−2) / (2×8)   =  0.19

EXTENT:
  e = 2 neighbours / 4 cells in range  =  0.50

PER-PERSON RAW WELFARE (self):
  h_self = 0.50 × [(0.25 + 0.31) + 0.5 × (0.006 + 0.19)]
         = 0.50 × [0.56 + 0.098]
         = 0.329

FINAL CELL SCORE  (assuming each neighbour also scores ≈ 0.28):
  Egoist:   h(c)  =  1.0 × 0.329                =  0.329
  Bentham:  h(c)  =  0.5 × 0.329 − 0.5 × 0.28  =  0.025
  Altruist: h(c)  =  0.0         − 1.0 × 0.28  = −0.280
```

The Altruist returns a **negative score** for this cell. Moving here would harm at least one neighbour, so the agent treats it as a cell to avoid — even if staying put worsens its own survival odds.

---

## One-Line Summary

| Agent type | Decision rule |
|------------|---------------|
| **Raw Sugarscape** | Move to the cell with the most food |
| **Egoist** | Move to the cell that benefits me the most |
| **Bentham** | Move to the cell that maximises total welfare across all affected agents |
| **Altruist** | Move to the cell that causes the least harm to others |

---

---

# FVDM: Felicific Vector Distance Matching

## What Changes from the Baseline

The baseline agent scores every candidate cell by computing a happiness value
`h(c)` through the hedonic formula, then picks the highest-scoring cell.
FVDM replaces that scoring rule with **distance matching in opportunity-cost
space**: instead of asking "which cell gives me the most welfare?", it asks
"which cell's welfare fingerprint most closely matches my derived profile?"

```
BASELINE AGENT                    FVDM AGENT  (Consistent Argmin in v_net space)
──────────────────────────────    ────────────────────────────────────────────────
For each candidate cell c:        For each candidate cell c:

  Compute h(c) via hedonic           Compute net fingerprint v_net(c):
  formula with designer-set φ          v_net_imm(c) = φ·v_imm_self(c)
                                                     − (1−φ)·mean_k[v_imm_k(c)]
  Pick c* = argmax h(c)               v_net_fut(c) = φ·v_fut_self(c)
                                                     − (1−φ)·mean_k[v_fut_k(c)]

                                     dist(c) = ‖μ_imm − v_net_imm(c)‖₂
                                             + ‖μ_fut  − v_net_fut(c)‖₂

                                     Pick c* = argmin dist(c)
```

The profile `(μ_imm, μ_fut)` is derived offline from the same v_net formula
applied to a baseline agent's observed choices. Because the target (μ) and
the candidates (v_net(c)) both use the same opportunity-cost decomposition,
the comparison is in a **consistent space** — unlike earlier argmin attempts
that compared a v_net profile against a raw v_self candidate.

## Part 1 — The Felicific Effect Vectors

Every candidate cell is described by two 5-dimensional vectors.
These are computed analytically from the simulation state at decision time.

```
CELL c  ──────────────────────────────────────────────────────────────────
        │
        ├─► IMMEDIATE EFFECT VECTOR   v_imm(c)  =  [ I,  D,  C,  P,  E ]
        │
        │     Dimension   Symbol   Formula                    Meaning
        │     ─────────── ──────   ─────────────────────────  ─────────────────────────────
        │     Intensity    I       1 / ((1+TTL)(1+pollution)) Urgency: higher when near death
        │                                                      or when cell is polluted
        │     Duration     D       (W_c / m) / W_cmax         Sustenance: how long cell feeds
        │                                                      agent relative to cell capacity
        │     Certainty    C       1  (always)                 Agent can definitely reach c
        │     Propinquity  P       1  (always)                 One step away (current timestep)
        │     Extent       E       |N(agent)| / |V|            Immediate social density:
        │                                                      neighbors visible from the
        │                                                      agent's current cell
        │
        └─► FUTURE EFFECT VECTOR      v_fut(c)  =  [ J,  Df, C,  P,  E ]
        │
        │     Dimension        Symbol  Formula                       Meaning
        │     ──────────────── ──────  ────────────────────────────  ────────────────────────────
        │     Future intensity  J      W_adj / (W_gmax × n_adj)      How resource-rich the cell's
        │                                                             neighbourhood is right now
        │     Future duration   Df     max(0, W_c − m) / (m×W_cmax) Resources left after eating
        │                                                             one step's worth
        │     Certainty         C      1  (always)
        │     Propinquity       P      γ  (lookahead discount)        Future reward counts less
        │     Extent            E      |N'(c)| / |V|             Future social density:
        │                                                          neighbors visible from
        │                                                          candidate cell c
        │                                                          (varies per candidate)
```

**Variable reference:**

| Symbol    | Meaning |
|-----------|---------|
| `TTL`     | Agent's time-to-live estimate (timesteps remaining before starvation) |
| `pollution` | Pollution level of cell `c` |
| `W_c`     | Total resources at cell `c` (sugar + spice) |
| `m`       | Agent's total metabolism (sugar + spice) |
| `W_cmax`  | Maximum possible resources at cell `c` (cell capacity) |
| `W_adj`   | Sum of resources in cells immediately adjacent to `c` |
| `W_gmax`  | Global maximum wealth across the entire environment |
| `n_adj`   | Number of cells immediately adjacent to `c` (4 for Von Neumann, 8 for Moore) |
| `\|V\|`   | Number of cells in the agent's vision range |
| `γ`       | Temporal discount factor (e.g. 0.5) |

**Numerical example** — same cell as before (resources=5, capacity=8, pollution=0,
adj. wealth=10, 4 adj. cells, TTL=3, metabolism=2, |V|=4, γ=0.5):

```
v_imm(c):
  I  =  1 / ((1+3)(1+0))    =  0.250
  D  =  min(1, (5/2) / 8)   =  0.313
  C  =  1.000
  P  =  1.000
  E  =  1/4                  =  0.250

  v_imm(c)  =  [ 0.250,  0.313,  1.000,  1.000,  0.250 ]

v_fut(c):
  J  =  10 / (400 × 4)       ≈  0.006
  Df =  max(0, 5−2)/(2×8)   =  0.188
  C  =  1.000
  P  =  0.500  (γ)
  E  =  0.250

  v_fut(c)  =  [ 0.006,  0.188,  1.000,  0.500,  0.250 ]
```

Each candidate cell maps to a point in this 5-dimensional felicific space.
The two vectors together form the cell's **welfare fingerprint**.

---

## Part 2 — Deriving the BFE Profiles (Opportunity-Cost Formulation)

The BFE profile `(mu_imm, mu_fut)` is derived **offline**, before any FVDM
experiment runs. The derivation agent is a **FVDMBFEAgent** — it decides
exactly like a normal hedonic agent (argmax h(c)), and after each move it
logs the net opportunity-cost fingerprint of the chosen cell.

Six steps take it from a simulation run to a stored profile.

```
FVDM BFE AGENT (standing at current position, deciding each timestep)
|
+--- STEP 1: Pick the best cell by ph-welfare  (normal hedonic rule)
|
|    FVDMBFEAgent scores every candidate cell with the same formula
|    the baseline agent uses:
|
|         h(c)  =  ph x h_self(c)  -  (1-ph) x SUM_k h_other_k(c)
|
|    c* = argmax h(c)   -- the agent moves to c*.
|    No profile is consulted. The derivation agent behaves
|    identically to the standard Egoist / Bentham / Altruist.
|
|    Four types are derived separately, each in its own homogeneous batch:
|
|      phiBfeEgoist    ph = 1.0   ->  profile key "egoist"
|      phiBfeBentham   ph = 0.5   ->  profile key "bentham"
|      phiBfeAltruist  ph = 0.0   ->  profile key "altruist"
|
|
+--- STEP 2: Compute feature vectors for self and every reachable neighbour
|
|    After the agent moves to c*, it computes the immediate and future
|    feature vectors that c* offers to every person k who could have
|    reached c*. These are the same five-dimensional vectors from Part 1.
|
|    For each person k that CAN reach c*:
|
|         v_imm_k(c*)  =  [ I_k,  D_k,  1.0,  1.0,  E_k  ]
|         v_fut_k(c*)  =  [ J,    Df_k, 1.0,  gamma, E_f_k ]
|
|    Numerical example -- Bentham agent (ph=0.5) chose c* = cell C.
|    (Same cell as the baseline example:
|     resources=5, capacity=8, pollution=0,
|     adj. wealth=10 over 4 cells, W_gmax=400, gamma=0.5)
|
|    +-------------------------------------------------------------+
|    |  SELF   (TTL=3, m=2, |V|=4, 1 neighbour from current cell) |
|    |    I_self  =  1 / ((1+3)x(1+0))   =  0.250                 |
|    |    D_self  =  (5/2) / 8            =  0.313                 |
|    |    J       =  10 / (400x4)         =  0.006  (cell's nbrs)  |
|    |    Df_self =  max(0, 5-2) / (2x8) =  0.188                 |
|    |    E       =  1/4  =  0.250  (neighbours from current cell) |
|    |    E_f     =  1/4  =  0.250  (neighbours from cell C)       |
|    |                                                             |
|    |    v_imm_self  =  [0.250, 0.313, 1.0, 1.0, 0.250]          |
|    |    v_fut_self  =  [0.006, 0.188, 1.0, 0.5, 0.250]          |
|    +-------------------------------------------------------------+
|    |  N1  (TTL=5, m=3, range=1, |V|=3, CAN reach c*)            |
|    |    I_N1   =  1 / ((1+5)x(1+0))   =  0.167  (less urgent)   |
|    |    D_N1   =  (5/3) / 8            =  0.208  (higher m)      |
|    |    J      =  10 / (400x4)         =  0.006  (same cell)     |
|    |    Df_N1  =  max(0, 5-3) / (3x8) =  0.083  (higher m)      |
|    |    E_N1   =  1/3  =  0.333  (range=1: only 3 cells in range)|
|    |    E_f_N1 =  1/3  =  0.333  (same |V| applies to fut too)   |
|    |                                                             |
|    |    v_imm_N1    =  [0.167, 0.208, 1.0, 1.0, 0.333]          |
|    |    v_fut_N1    =  [0.006, 0.083, 1.0, 0.5, 0.333]          |
|    +-------------------------------------------------------------+
|    |  N2  (CANNOT reach c* -- range < distance)  -> skipped     |
|    +-------------------------------------------------------------+
|
|    One reachable neighbour (N1). mean_k[v_k] = v_N1.
|
|
+--- STEP 3: Compute v_net(c*) = ph x v_self  -  (1-ph) x mean_k[v_k]
|
|    This is the hedonic formula h(c) written coordinate-by-coordinate.
|    Instead of collapsing to a scalar, we keep all five dimensions.
|
|    For Bentham (ph = 0.5), immediate vectors:
|
|    Dim    ph x self          (1-ph) x mean_k     v_net_imm
|    -----  ------------------  ------------------  ----------
|    I      0.5 x 0.250 = 0.125  0.5 x 0.167 = 0.083  +0.042
|    D      0.5 x 0.313 = 0.156  0.5 x 0.208 = 0.104  +0.052
|    C      0.5 x 1.000 = 0.500  0.5 x 1.000 = 0.500  +0.000
|    P      0.5 x 1.000 = 0.500  0.5 x 1.000 = 0.500  +0.000
|    E      0.5 x 0.250 = 0.125  0.5 x 0.333 = 0.167  -0.042
|
|         v_net_imm(c*)  =  [+0.042, +0.052, +0.000, +0.000, -0.042]
|
|    Future vectors (N1 has m=3 and |V|=3, so Df and E_f now differ):
|
|    Dim    ph x self          (1-ph) x mean_k     v_net_fut
|    -----  ------------------  ------------------  ----------
|    J      0.5 x 0.006 = 0.003  0.5 x 0.006 = 0.003  +0.000
|    Df     0.5 x 0.188 = 0.094  0.5 x 0.083 = 0.042  +0.052
|    C      0.5 x 1.000 = 0.500  0.5 x 1.000 = 0.500  +0.000
|    P      0.5 x 0.500 = 0.250  0.5 x 0.500 = 0.250  +0.000
|    E_f    0.5 x 0.250 = 0.125  0.5 x 0.333 = 0.167  -0.042
|
|         v_net_fut(c*)  =  [+0.000, +0.052, +0.000, +0.000, -0.042]
|
|    Reading the result:
|      I_net   = +0.042  ->  self more urgently needs this cell (TTL=3 vs 5)
|      D_net   = +0.052  ->  cell sustains self longer per metabolism unit
|                            (self m=2: sustenance=0.313; N1 m=3: sustenance=0.208)
|      C_net   =  0.000  ->  certainty cancels -- both can definitely reach c*
|      P_net   =  0.000  ->  propinquity cancels -- both one timestep away
|      E_net   = -0.042  ->  N1 has fewer candidate cells (|V|=3 vs 4); the
|                            opportunity cost to N1 is socially amplified
|      J_net   =  0.000  ->  neighbourhood richness depends only on the cell,
|                            not on the agent asking
|      Df_net  = +0.052  ->  more residual resources remain after self eats
|                            (3 units / 16 capacity) vs N1 (2 units / 24 capacity)
|      E_f_net = -0.042  ->  same |V| effect: N1's candidate space is narrower,
|                            amplifying its social signal from cell C
|
|    For comparison -- what Egoist and Altruist would log at this step:
|
|      Egoist   (ph=1.0):  v_net = v_self = [0.250, 0.313, 1.0, 1.0, 0.250]
|                           opportunity-cost term drops out entirely
|
|      Altruist (ph=0.0):  v_net = -mean_k = -[0.167, 0.208, 1.0, 1.0, 0.333]
|                           = [-0.167, -0.208, -1.0, -1.0, -0.333]
|                           records only the cost imposed on N1
|
|
+--- STEP 4: Check neighbours > 0 -- keep or discard this row
|
|    This agent had 1 living neighbour in range (N=1 > 0) -> INCLUDED.
|
|    If the agent were isolated (no neighbours at all), v_net carries
|    no social signal and the row is excluded from the derivation.
|    This filter keeps only socially-contextual moves.
|
|    +------------------------------------------------------+
|    |  neighbours  =  0  ->  DISCARD (no social context)   |
|    |  neighbours  >  0  ->  INCLUDE (social choice made)  |
|    +------------------------------------------------------+
|
|    Note: even with neighbours > 0, the code may fall back to logging
|    v_self if none of those neighbours can actually reach c*. This
|    happens in ~4% of altruist rows and ~33% of bentham rows, slightly
|    inflating the C and P coordinates away from their theoretical values
|    (-1.0 for altruist, 0.0 for bentham). The actual profile reflects
|    this mix -- it is the empirical average, not a theoretical prediction.
|
|
+--- STEP 5: Repeat across all timesteps and seeds; accumulate
|
|    Each qualifying timestep adds one (v_net_imm, v_net_fut) pair.
|    The pipeline maintains a running sum and count per agent type.
|
|    timestep t1:  v_net_imm = [+0.042, +0.052, +0.000, +0.000, -0.042]
|    timestep t2:  v_net_imm = [+0.031, +0.018, +0.000, +0.000, +0.002]
|    timestep t3:  v_net_imm = [+0.012, +0.044, +0.000, +0.000, +0.000]
|    ...
|    timestep tN:  v_net_imm = [+0.028, +0.011, +0.000, +0.000, +0.001]
|
|    sum_imm  (after N qualifying steps, Bentham example):
|      [ sum(I), sum(D), sum(C), sum(P), sum(E) ]
|
|    Run 10 seeds x ~1000 timesteps each, filter to neighbours > 0:
|      Bentham  ->  ~7.57 million qualifying observations
|      Altruist ->  ~3.25 million  (fewer: population dies faster at ph=0)
|      Egoist   ->  ~4.50 million
|
|
+--- STEP 6: Divide sum by count -- the profile
|
|    mu_imm  =  sum_imm / N_qualifying
|    mu_fut  =  sum_fut  / N_qualifying
|
|    These two 5D vectors are the BFE profile for this agent type.
|    Stored in bfe_profiles_phi.json under the type's key.
|
|    Actual values from the current derivation (10 seeds, 1000 timesteps):
|
|    Profile      mu_imm[ I ]   mu_imm[ D ]   mu_imm[ C ]   mu_imm[ P ]
|    -----------  -----------   -----------   -----------   -----------
|    egoist        +0.043        +0.297        +1.000        +1.000
|    bentham       +0.018        +0.119        +0.332        +0.332
|    altruist      -0.046        -0.258        -0.922        -0.922
|
|    Why bentham C,P ~ 0.332 (not 0.0):  at ph=0.5 the C coordinate
|    should cancel to 0 when a reachable neighbour exists. But ~33% of
|    qualifying rows had no reachable neighbour for c* and fell back to
|    v_self (C=+1.0). The average of 0 and +1.0 weighted 67/33 gives
|    ~0.33. Same effect explains P ~ 0.332.
|
|    Why altruist C,P ~ -0.922 (not -1.0):  at ph=0.0 the formula
|    gives C = -1.0 when a reachable neighbour exists. About 4% of
|    rows fell back to v_self (C=+1.0). The average of -1.0 and +1.0
|    weighted 96/4 gives ~-0.92.
```

**Connection to the h(c) formula — why v_net is not a new idea:**

```
Scalar form:  h(c)  =  ph x h_self(c)   -   (1-ph) x SUM_k h_k(c)
                              |                              |
                     one scalar per cell           one scalar per neighbour

Vector form:  v_net(c*)  =  ph x v_self(c*)  -  (1-ph) x mean_k[v_k(c*)]
                               |                                |
                    five-dim vector per cell        five-dim vector per neighbour
```

h(c) collapses the five dimensions into a single score for the argmax.
v_net(c*) keeps the five-dimensional structure, enabling the derivation
to average it across thousands of observations and recover a stable
characteristic fingerprint per agent type.

**Script:** `python3 derive_vectors_phi.py --homogeneous -s 10 -t 5000 -a 250 -j 30`
**Output:** `fvdm_vectors/bfe_profiles_phi.json`

---

## Part 3 — FVDM Action Selection (Consistent Argmin)

At runtime the FVDM agent replaces the baseline h(c) scoring with
distance-minimising matching against its loaded prioritization profile.
Both the profile and the candidates are expressed as v_net vectors —
the opportunity-cost decomposition of the welfare formula — so the
comparison is in the same space.

```
FVDM AGENT — DECISION PROCEDURE  (FVDMArgminAgent)
────────────────────────────────────────────────────────────────────────────

At startup:
  Load profile (μ_imm, μ_fut) from bfe_profiles_phi.json
  Set φ for this agent's type:
    fvdmArgminEgoist   → φ = 1.0,  profile = "egoist"
    fvdmArgminBentham  → φ = 0.5,  profile = "bentham"
    fvdmArgminAltruist → φ = 0.0,  profile = "altruist"

Each timestep, for every candidate cell c:

  ┌──────────────────────────────────────────────────────────────────┐
  │  1. Identify reachable neighbours k (agents that can reach c)    │
  │                                                                  │
  │  2. Compute net fingerprint of c for this agent:                 │
  │       v_net_imm(c) = φ·v_imm_self(c) − (1−φ)·mean_k[v_imm_k(c)]│
  │       v_net_fut(c) = φ·v_fut_self(c) − (1−φ)·mean_k[v_fut_k(c)]│
  │                                                                  │
  │  3. Compute distance to profile:                                 │
  │       dist(c) = ‖μ_imm − v_net_imm(c)‖₂                        │
  │               + ‖μ_fut  − v_net_fut(c)‖₂                        │
  │                                                                  │
  │  4. Pick c* = argmin dist(c)                                     │
  └──────────────────────────────────────────────────────────────────┘

After moving to c*, log v_net of the chosen cell:
  argmin_v_imm_[I,D,C,P,E]  and  argmin_v_fut_[I,D,C,P,E]
  (enables post-hoc BFS: does the observed mean match the profile?)
```

**Why this comparison is consistent:**

The profile μ was derived by averaging v_net(c*) logged by FVDMBFEAgent
during homogeneous baseline runs. At decision time, FVDMArgminAgent
computes v_net(c) using the same formula for every candidate cell. So
both sides of the distance computation use the same opportunity-cost
decomposition. Compare with earlier attempts:

```
Earlier attempt (INCONSISTENT):
  Profile μ   derived from v_net(c*)        ← opportunity-cost space
  Candidate   v_self(c) or v_imm(c)         ← raw gain space
  → comparing apples to oranges → extinction

Current approach (CONSISTENT):
  Profile μ   derived from v_net(c*)        ← opportunity-cost space
  Candidate   v_net(c) = φ·v_self(c)−...   ← same space
  → argmin is a valid distance in a shared space
```

### Worked Example — Egoist FVDMArgminAgent Choosing Between Three Cells

Loaded profile (φ=1.0, egoist):
```
  μ_imm  =  [+0.27,  +0.31,  +1.0,  +1.0,  +0.21]
  μ_fut  =  [+0.04,  +0.12,  +1.0,  +0.50, +0.21]
```

For egoist (φ=1.0): v_net(c) = v_self(c) — cost term vanishes.

Three candidate cells:

```
  Cell A  (resource-rich):
    v_net_imm  =  [ 0.25, 0.31, 1.00, 1.00, 0.25 ]
    v_net_fut  =  [ 0.006, 0.19, 1.00, 0.50, 0.25 ]
    dist(A)    =  ‖[−0.02, 0.00, 0, 0, +0.04]‖ + ‖[−0.034, +0.07, 0, 0, +0.04]‖
               ≈  0.045 + 0.090  =  0.135

  Cell B  (depleted):
    v_net_imm  =  [ 0.25, 0.05, 1.00, 1.00, 0.25 ]
    dist(B)    ≈  0.263 + 0.190  =  0.453

  Cell C  (polluted):
    v_net_imm  =  [ 0.12, 0.09, 1.00, 1.00, 0.25 ]
    dist(C)    ≈  0.228 + 0.170  =  0.398
```

**Result:** c* = Cell A  (smallest distance to the egoist profile).
The agent moves to the resource-rich cell — matching the egoist profile
which is anchored at high positive D and I values.

## Full Pipeline Summary

```
OFFLINE: BFE DERIVATION                           ONLINE: DECISION + LOGGING
(derive_vectors_phi.py, run once)                 (run_experiments_fvdm_argmin.py)
──────────────────────────────────────────────    ──────────────────────────────────

 Homogeneous sims per agent type                  FVDMArgminAgent starts at cell
 (FVDMBFEAgent, one type per batch)                        │
         │                                                 ▼
         ▼                                        For each candidate cell c:
 Filter: neighbours > 0                             compute v_net(c) = φ·v_self(c)
         │                                                 − (1−φ)·mean_k[v_k(c)]
         ▼                                                 │
 FVDMBFEAgent logs v_net(c*):                              ▼
   v_net = φ·v_self(c*) − (1−φ)·mean_k[v_k(c*)]  dist(c) = ‖μ_imm − v_net_imm(c)‖
         │                                                + ‖μ_fut  − v_net_fut(c)‖
         ▼                                                 │
 Average logged v_net per type                             ▼
         │                                        c* = argmin dist(c)
         ▼                                                 │
 Save μ_imm, μ_fut                                         ▼
 → bfe_profiles_phi.json                          Log v_net(c*) for BFS

                                                  POST-HOC:
                                                    μ_obs = mean(logged v_net)
                                                    BFS = cosine_sim(μ_obs, μ_profile)
                                                    → bfs_vs_derived.csv
```


---

---

# Experimental Narrative: What We Run and Why It Proves FVDM Is Needed

This section walks through the full simulation pipeline as a story. Each
experiment asks a question. Each answer either reveals a problem with the
baseline or demonstrates that FVDM solves it.

---

## The Story in One Paragraph

The baseline hedonic calculus (KH2024) is a sound model — but it has a hidden
assumption: you must already know what φ to set for an agent to behave
ethically. FVDM asks the opposite question. Instead of *"what do I set φ to?"*
it asks *"given how this agent behaves, what does its ethical profile look
like?"* The experiments below build the case that this question matters — and
that FVDM answers it in a way the baseline cannot.

---

## Experiment 1 — Baseline Homogeneous Conditions

**Script:** `run_experiments_baseline.py` (4 conditions, 30 seeds each)

**What we run:**

```
Condition 1:  rawSugarscape  — no ethical formula, pure resource grab
Condition 2:  egoist         — φ = 1.0, only own welfare matters
Condition 3:  altruist       — φ = 0.0, only others' welfare matters
Condition 4:  bentham        — φ = 0.5, own and others' welfare balanced
```

Each condition runs 30 independent seeds (30 random starting configurations)
for 5,000 timesteps each. All other parameters are identical — same map,
same number of agents, same metabolism range.

**What we observe:**

```
Metric            rawSugar    Egoist    Altruist    Bentham
──────────────    ────────    ──────    ────────    ───────
Extinction %         ?          ?          ?           ?
Final population     ?          ?          ?           ?
Mean wealth          ?          ?          ?           ?
Gini (inequality)    ?          ?          ?           ?
Time-to-live         ?          ?          ?           ?
```

*(Values filled in after experiments run.)*

**What this reveals:**

The four conditions give us the performance baseline — a reference point for
how each ethical stance affects the population over time. By itself this is a
replication of KH2024. The important thing this experiment sets up is: *we now
know what Egoist, Altruist, and Bentham populations look like at the societal
level.*

**The problem it exposes:**

The only thing distinguishing the three hedonic conditions is the value of φ.
That value is set by the researcher. There is no way to look at an agent
mid-simulation and ask *"how altruistic is this agent actually behaving right
now?"* — only *"what φ was it given at the start?"* φ is a design parameter,
not a measurement.

---

## Experiment 2 — The Selfishness Factor Sweep

**Script:** `run_experiments_selfishness.py` (21 φ levels, 30 seeds each)

**What we run:**

```
φ = 0.00  (pure altruist)
φ = 0.05
φ = 0.10
...
φ = 0.50  (bentham)
...
φ = 0.95
φ = 1.00  (pure egoist)
```

Every agent in the simulation uses the same φ (homogeneous population). This
is a direct replication of Herman & Kremer (2024) Section VII-C.

**What we observe (KH-style plots, median + Q1/Q3 bands):**

```
     Final population
     │
 250 ┤  ╭──────╮
     │ ╱        ╲
 150 ┤╱           ╲────────
     │
     └──────────────────────
      0.0    0.5    1.0
            φ

     Gini coefficient
     │
 1.0 ┤                  ╭──
     │           ╭──────╯
 0.5 ┤───────────╯
     │
     └──────────────────────
      0.0    0.5    1.0
            φ
```

**What this reveals:**

The relationship between φ and societal outcomes is not simple. Population
peaks at a middle φ value — neither fully selfish nor fully altruistic societies
thrive best. Some metrics change monotonically with φ; others have inflection
points or plateau regions.

**The problem it exposes:**

Two different φ values can produce nearly identical final population and mean
wealth. From outcome metrics alone, you cannot distinguish a φ=0.3 society
from a φ=0.7 society if they happen to have similar populations. The same
outcomes can arise from fundamentally different ethical stances.

This is **Gap 1** in the framework: outcome equivalence ≠ ethical equivalence.
The baseline has no tool to detect the difference. FVDM does — because it
operates in behavioral space, not outcome space.

---

## Experiment 3 — The Heterogeneous Population Sweep

**Script:** `run_experiments_baseline.py` (hetero sweep, 11 mixes, 30 seeds)

**What we run:**

```
Mix 1:   0% Bentham, 100% Egoist
Mix 2:  10% Bentham,  90% Egoist
...
Mix 6:  50% Bentham,  50% Egoist
...
Mix 11: 100% Bentham,  0% Egoist
```

The selfishness factor of each agent is set by its type (Egoist: φ=1,
Bentham: φ=0.5). The mix ratio changes but individual φ values do not.

**What we observe:**

```
     Final population vs. Bentham proportion
     │
 250 ┤          ╭─────────╮
     │         ╱           ╲
 150 ┤────────╱             ╲────
     │
     └──────────────────────────
      0%   Bentham proportion  100%
           Egoist ←───→ Bentham

     Spearman r (% Bentham vs. population) = ?
```

**What this reveals:**

Mixing ethical types changes societal outcomes in ways that a purely
homogeneous model cannot predict. The population curve has inflection points —
small proportions of Bentham agents in an Egoist population can improve
outcomes more than proportionally, suggesting ethical minority effects.

**The problem it exposes:**

This is **Gap 2**: a Bentham agent's behavior is coupled to who surrounds it.
The hedonic formula for φ=0.5 weights neighbor welfare at 50%. As the
proportion of Egoists increases, the neighbors whose welfare is being counted
are Egoists — agents who pursue high-resource cells aggressively. The Bentham
agent's decision changes not because its φ changed, but because its
*neighborhood changed.*

In other words: the same φ=0.5 agent behaves differently in a 10% Bentham
population than in a 90% Bentham population. The baseline has no way to
measure or control this behavioral drift. It only knows the design parameter φ.

---

## Experiment 4 — BFE Derivation (The Core Contribution)

**Script:** `derive_vectors_phi.py` (homogeneous batches, one type per run)

**What we run:**

```
Four separate batches — one per agent type:
  phiBfeEgoist   (φ=1.0): 250 agents, all egoist
  phiBfeAltruist (φ=0.0): 250 agents, all altruist
  phiBfeBentham  (φ=0.5): 250 agents, all bentham

  N seeds per type (e.g. 10 seeds × 5000 timesteps).
  Each FVDMBFEAgent decides by φ-welfare and logs v_net every timestep.
  Homogeneous batches eliminate cross-type competitive bias.

Command:
  python3 derive_vectors_phi.py --homogeneous -s 10 -t 5000 -a 250 -j 30
```

**What we compute:**

```
For each agent type, collect all logged v_net from timesteps
where neighbours > 0:

  Egoist agents logged net fingerprints:
      obs 1: v_net_imm = [+0.31, +0.28, +1.0, +1.0, +0.20]  (φ=1 → pure gain)
      obs 2: v_net_imm = [+0.28, +0.33, +1.0, +1.0, +0.25]
      ...
      obs N: v_net_imm = [+0.22, +0.31, +1.0, +1.0, +0.19]
                          ↓ average
      mu_imm_egoist  =  [+0.27, +0.31, +1.0, +1.0, +0.21]  ← profile

  Altruist agents (φ=0 → pure cost term):
      obs 1: v_net_imm = [-0.28, -0.24, -1.0, -1.0, -0.18]
      ...
      mu_imm_altruist  =  [-0.24, -0.21, -1.0, -1.0, -0.17]  ← profile

  Bentham (φ=0.5 → gain and cost balanced):
      mu_imm_bentham  ≈  [+0.01, +0.04, 0.0, 0.0, +0.01]  ← near zero

  Same procedure for mu_fut in each case.

Variance analysis:
  Per-seed mu vectors are computed before pooling.
  Their variance across seeds determines whether enough seeds were run.
  The pipeline reports convergence diagnostics automatically.
```

**Output:** `fvdm_vectors/bfe_profiles_phi.json`

**What this reveals:**

Each agent type has a characteristic *opportunity-cost signature* when making
socially-contextual moves. The signatures are distinct and interpretable:

- Egoist profile is all-positive (the agent gains; no cost term)
- Altruist profile is all-negative (no gain; only cost to others is counted)
- Bentham profile clusters near zero (gain and cost cancel at φ=0.5)

These signatures are derivable purely from observed behaviour — no φ is assumed.

**Why this is the key result:**

The BFE profile is the first tool in this framework that characterises ethical
behaviour as a measurable object. Given any agent's sequence of choices, we can
compute its empirical v_net profile and ask: *how close is this to the known
egoist profile? to the altruist profile?* That question was previously unanswerable.

---

## Experiment 5 — Consistent-Argmin FVDM Agents

**Script:** `run_experiments_fvdm_argmin.py` (4 conditions, 30 seeds each)

**What we run:**

```
Condition 1:  argminRaw       — FVDMArgminAgent φ=1.0  (raw/greedy)
Condition 2:  argminEgoist    — FVDMArgminAgent φ=1.0  (egoist)
Condition 3:  argminAltruist  — FVDMArgminAgent φ=0.0  (altruist)
Condition 4:  argminBentham   — FVDMArgminAgent φ=0.5  (bentham)

30 seeds × 5000 timesteps × 250 agents each.
Profile loaded from bfe_profiles_phi.json (corrected opportunity-cost profiles).

Command:
  python3 run_experiments_fvdm_argmin.py -s 30 -t 5000 -a 250 -j 30
```

**What we observe:**

```
Primary outputs:

  bfs_vs_derived.csv  — BFS(μ_obs, μ_profile) per condition
                        (does the argmin agent's observed fingerprint
                         match the profile it was matching against?)

Secondary outputs:

  per_seed_summary.csv     — extinction rate, final pop, mean wealth, Gini, TTL
  condition_aggregates.csv — mean ± sd across seeds
```

**What this reveals — the three comparison tests:**

```
TEST A — Does consistent argmin produce the correct behavioral fingerprint?

  If the agent matches profile μ_egoist by argmin, its observed choices
  should also have a mean v_net close to μ_egoist → BFS ≈ 1.0.
  A low BFS would mean the argmin pulls agents toward cells whose v_net
  differs from the profile (e.g. "average" cells rather than
  characteristically egoist cells).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Expected: BFS close to 1.0 for all conditions.                 │
  │  If BFS < 0.9: the argmin rule finds cells that are "close to   │
  │  the profile" by distance but whose average fingerprint         │
  │  drifts from μ — the "average cell" effect.                     │
  └─────────────────────────────────────────────────────────────────┘


TEST B — Do agents survive?

  Extinction rate and final population tell us whether consistent
  argmin in v_net space is ecologically viable.

  ┌─────────────────────────────────────────────────────────────────┐
  │  If high extinction: consistent argmin still fails ecologically. │
  │  If low extinction: argmin is a viable decision rule and the     │
  │  "average cell" concern was overstated.                          │
  └─────────────────────────────────────────────────────────────────┘


TEST C — Can FVDM detect ethical equivalence the baseline misses?

  Find two φ values from Experiment 2 that produce identical societal
  outcomes. Compute their BFE profiles. Show they are measurably
  different in v_net space even though outcome metrics are identical.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Panel asks: "So what? If outcomes are the same, who cares?"    │
  │  Answer: A hospital that heals the same number of patients      │
  │  using two different treatments is not ethically equivalent.    │
  │  The process matters. FVDM characterises the process via BFE.   │
  │  The baseline cannot — it only reports outcomes.                │
  └─────────────────────────────────────────────────────────────────┘
```
---

## Experiment 6 — Profile Space Visualization

**Script:** Custom plot using PCA on derived BFE profiles.

**What we run:** No new simulation. Post-process the profiles already derived
in Experiment 4 and 5.

**What we plot:**

```
Project all 10D profiles (mu_imm + mu_fut concatenated) into 2D via PCA.
Plot one point per agent type — derived profile vs. observed empirical BFE.

  BFE space (PCA projection, v_net opportunity-cost space):

  PC2
  │
  │  Altruist ●                        ← derived profile (bfe_profiles_phi.json)
  │             ╲
  │              ╲
  │    argminAltruist ★                ← empirical BFE from experiment
  │
  │             Bentham ●
  │           argminBentham ★          ← expected: tightly clustered
  │
  │                        Egoist ●
  │                      argminEgoist ★
  │
  └─────────────────────────────────── PC1

Note: because v_net vectors have negative components (altruist profile
is all-negative, egoist is all-positive), the PCA axis separates ethical
types more cleanly than raw v_self vectors would.
```

**What this reveals:**

If FVDM-derived agents cluster near their corresponding baseline profiles in
this space, the plot shows visually that:
  1. The four ethical types are genuinely distinct in behavioral space.
  2. FVDM correctly positions agents relative to those types without using φ.
  3. The distance between profiles is a meaningful measure — Egoist and
     Altruist are furthest apart; Bentham sits between them.

This is the figure a panelist can look at for ten seconds and understand the
entire claim of the thesis.

---

## The Full Experimental Arc — At a Glance

```
Experiment 1:  WHAT DOES EACH ETHICAL TYPE DO?
               Baseline homogeneous conditions.
               → Establishes reference outcomes per agent type.

Experiment 2:  WHAT HAPPENS WHEN φ IS CONTINUOUS?
               Selfishness sweep (φ = 0.0 to 1.0).
               → Shows outcome equivalence problem: same outcomes,
                 different φ. Baseline cannot distinguish them.

Experiment 3:  WHAT HAPPENS IN MIXED POPULATIONS?
               Heterogeneous Egoist–Bentham sweep.
               → Shows behavioral drift: same φ, different neighbors,
                 different effective behavior. Baseline cannot measure this.

Experiment 4:  CAN WE CHARACTERIZE BEHAVIOR WITHOUT φ?
               BFE derivation (derive_vectors_phi.py, homogeneous batches).
               → YES. Each agent type has a stable opportunity-cost fingerprint
                 derivable from observed choices alone.

Experiment 5:  DOES CONSISTENT ARGMIN FVDM WORK?
               run_experiments_fvdm_argmin.py + BFS verification.
               → BFS tells us if argmin agents produce the expected fingerprint.
               → Extinction rate tells us if argmin is ecologically viable.
               → BFE profiles distinguish ethical stances with same outcomes (Test C).

Experiment 6:  CAN WE SEE THE FRAMEWORK WORKING?
               PCA projection of all profiles.
               → Visual confirmation that ethical types are distinct in
                 behavioral space and FVDM maps agents correctly.
```

**The one-sentence answer to "why is FVDM needed?"**

> Because φ tells you what an agent is designed to be. FVDM measures what
> an agent actually does — and those two things are not always the same.
