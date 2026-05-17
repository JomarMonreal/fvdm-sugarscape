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
| `e` | Extent — social density weight: ratio of neighbors to cells in vision range |
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
│           i_k = 1 / ( (1 + TTL_k) × (1 + pollution) )
│                 — higher when the agent is close to dying
│                 — higher when the cell carries more pollution
│
│           d_k = (resources / metabolism_k) / capacity
│                 — higher when the cell can sustain the agent longer
│
│    b) Future welfare components (same cell, one timestep later):
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
│           h_k = (c_k × p) × [ e × (i_k + d_k)  +  γ × e × (i_f + d_f_k) ]
│                               └──── immediate ────┘  └──── future (discounted) ──┘
│
│           where  c_k = 1 if person k can reach c, else 0
│                  p   = 1  (one-step lookahead, always)
│                  e   = |neighbors in range| / |cells in range|
│
│    Numerical example — 🧑 (self) evaluating cell C:
│    (TTL=3, m=2, |V|=4, 1 neighbor in range, γ=0.5)
│
│           i_self = 1 / ((1+3)×(1+0))    =  0.250   (TTL=3, pollution=0)
│           d_self = (5 / 2) / 8           =  0.313   (resources/m / capacity)
│           i_f    = 10 / (400 × 4)        =  0.006   (adj wealth / globalMax×n_adj)
│           d_f    = (5 − 2) / (2 × 8)    =  0.188   (residual resources after eating)
│           e      = 1 / 4                 =  0.250   (neighbors in range / |V|)
│
│           h_self = 0.250 × [(0.250 + 0.313) + 0.5 × (0.006 + 0.188)]
│                  = 0.250 × [0.563 + 0.097]
│                  = 0.250 × 0.660
│                  = 0.165
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

The baseline agent scores every candidate cell by computing a happiness value `h(c)` through the hedonic formula. The FVDM agent replaces that scoring function with a **geometric matching** approach: instead of calculating how much happiness a cell produces, it asks how closely a cell's welfare fingerprint matches a pre-learned target profile.

```
BASELINE AGENT                        FVDM AGENT
──────────────────────────────────    ──────────────────────────────────
For each candidate cell c:            For each candidate cell c:

  Compute h(c) via hedonic formula      Compute effect vectors v_imm(c)
  (intensity, duration, extent, φ)      and v_fut(c) from cell properties

  Pick c* = argmax h(c)                 Pick c* = argmin distance to
                                        learned profile (mu_imm, mu_fut)
```

The profile `(mu_imm, mu_fut)` is derived offline from observed behaviour
and loaded at the start of each simulation. It encodes what kind of cell
a particular agent type characteristically chooses.

---

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
        │     Extent       E       1 / |V|                     Social density weight
        │                                                      (1 per cell in vision range)
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
        │     Extent            E      1 / |V|
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

## Part 2 — Deriving the Prioritization Vectors (BFE)

The prioritization profile `(mu_imm, mu_fut)` for each agent type is derived
offline, before any FVDM experiment runs. The method is called
**Behavioral Feature Expectation (BFE)**.

The core idea: observe what cells the baseline agents actually choose under
pressure, compute the welfare fingerprint of each chosen cell, and average
those fingerprints across many observations. The average fingerprint is the
agent's characteristic target — the kind of cell it typically moves toward.

```
DERIVATION PIPELINE
────────────────────────────────────────────────────────────────────────────

STEP 1: Run mixed-population simulations
│
│   A single simulation contains all four baseline agent types
│   (rawSugarscape, egoist, bentham, altruist) in a round-robin mix.
│   Many seeds are run to reduce variance.
│
│   ┌───────────────────────────────────────────────┐
│   │  Sim seed 1  →  agent log (all timesteps)     │
│   │  Sim seed 2  →  agent log (all timesteps)     │
│   │  ...                                           │
│   │  Sim seed N  →  agent log (all timesteps)     │
│   └───────────────────────────────────────────────┘

STEP 2: Filter to contested moves only
│
│   A move is contested when the chosen cell was occupied by an
│   agent of a different tribe. These moments reveal the agent's
│   true preference under conflict — the ethically meaningful case.
│   Uncontested moves are excluded (the agent had no real choice).
│
│   All timesteps:  ████░████░░███░████░░░████░░██
│                         ↑   ↑        ↑   ↑  ↑
│   Contested only:       ●   ●        ●   ●  ●
│                   (only these rows enter the derivation)

STEP 3: Compute effect vectors for each contested observation
│
│   For every qualifying row in the agent log:
│
│     row  →  compute v_imm(c)  =  [I,  D,  1,  1,   1/|V|]
│          →  compute v_fut(c)  =  [J,  Df, 1,  γ,   1/|V|]
│
│   Each observation produces one pair (v_imm, v_fut).

STEP 4: Average per agent type
│
│                      observations
│   egoist    :   v1   v2   v3  ...  vN
│                 ───────────────────────
│                 mean → mu_imm_egoist
│                         mu_fut_egoist
│
│   altruist  :   same procedure → mu_imm_altruist, mu_fut_altruist
│   bentham   :   same procedure → mu_imm_bentham,  mu_fut_bentham
│   rawSugar  :   same procedure → mu_imm_raw,      mu_fut_raw

STEP 5: Save profiles to bfe_profiles.json
│
│   {
│     "egoist":   { "mu_imm": [I, D, C, P, E],
│                   "mu_fut":  [J, Df, C, P, E] },
│     "altruist": { ... },
│     "bentham":  { ... },
│     ...
│   }
│
│   These profiles are the learned target fingerprints.
│   One profile per agent type. Each profile is two 5D vectors.
```

**Optional validation — φ-linearity check:**
Because the baseline Bentham agent is defined as φ = 0.5 (exactly halfway
between Egoist and Altruist), a consistency check tests whether:

```
  mu_bentham  ≈  0.5 × mu_egoist  +  0.5 × mu_altruist
```

The similarity is measured by cosine similarity. If the result is close to
1.0, the three hedonic profiles lie on the same line in felicific space,
confirming that φ linearly parameterises the space.

---

## Part 3 — FVDM Action Selection

At runtime, the FVDM agent replaces the baseline h(c) scoring with
distance-based matching against its loaded prioritization profile.

```
FVDM AGENT — DECISION PROCEDURE
────────────────────────────────────────────────────────────────────────────

At startup: load profile (mu_imm, mu_fut) from bfe_profiles.json
            for this agent's type  (e.g. "egoist")

            mu_imm  =  [ 0.21,  0.30,  1.00,  1.00,  0.18 ]   ← target
            mu_fut  =  [ 0.04,  0.15,  1.00,  0.50,  0.18 ]   ← target

Each timestep, for every candidate cell c:

  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Compute v_imm(c) and v_fut(c) from current cell state      │
  │                                                                 │
  │  2. Measure distance from profile:                              │
  │                                                                 │
  │       dist(c) = ‖ mu_imm − v_imm(c) ‖₂                        │
  │               + ‖ mu_fut  − v_fut(c) ‖₂                        │
  │                                                                 │
  │  3. Pick c* = argmin dist(c)                                    │
  └─────────────────────────────────────────────────────────────────┘

Visualised — each cell is a point in felicific space.
The agent targets the cell closest to its learned profile:

  Felicific space (schematic, 2D projection of 5D):

  mu_imm ★                             ← target (learned profile)
           \
            \  dist = 0.31
             \
              ● Cell A  (dist = 0.31)   ← closest → c*
              
              ● Cell B  (dist = 0.55)
              
                     ● Cell C  (dist = 0.80)

  The agent moves to Cell A — not because it gives the most sugar,
  but because it most closely matches the type of cell this
  agent characteristically chooses.
```

### Worked Example — Egoist FVDM Agent Choosing Between Three Cells

Suppose the loaded egoist profile is:

```
  mu_imm  =  [ 0.21,  0.30,  1.00,  1.00,  0.18 ]
  mu_fut  =  [ 0.04,  0.15,  1.00,  0.50,  0.18 ]
```

Three candidate cells are in range:

```
  Cell A  (rich, no pollution):
    v_imm  =  [ 0.25,  0.31,  1.00,  1.00,  0.25 ]
    v_fut  =  [ 0.006, 0.19,  1.00,  0.50,  0.25 ]
    dist   =  ‖[0.21-0.25, 0.30-0.31, 0, 0, 0.18-0.25]‖
            + ‖[0.04-0.006, 0.15-0.19, 0, 0, 0.18-0.25]‖
           ≈  0.081 + 0.082  =  0.163

  Cell B  (depleted):
    v_imm  =  [ 0.25,  0.05,  1.00,  1.00,  0.25 ]
    v_fut  =  [ 0.002, 0.00,  1.00,  0.50,  0.25 ]
    dist   ≈  0.263 + 0.195  =  0.458

  Cell C  (heavily polluted):
    v_imm  =  [ 0.12,  0.09,  1.00,  1.00,  0.25 ]
    v_fut  =  [ 0.003, 0.04,  1.00,  0.50,  0.25 ]
    dist   ≈  0.228 + 0.172  =  0.400
```

**Result:** Cell A has the smallest total distance (0.163).
The FVDM egoist moves to Cell A — matching the typical egoist choice
of a resource-rich, low-pollution destination.

---

## Full Pipeline Summary

```
OFFLINE (run once, before experiments)            ONLINE (each simulation timestep)
──────────────────────────────────────────────    ──────────────────────────────────
                                                  
 Baseline sims (mixed population)                 FVDM agent stands at current cell
         │                                                 │
         ▼                                                 ▼
 Filter contested moves                           Load profile (mu_imm, mu_fut)
         │                                                 │
         ▼                                                 ▼
 Compute v_imm(c), v_fut(c)                       For each candidate cell c:
 per contested observation                          compute v_imm(c), v_fut(c)
         │                                                 │
         ▼                                                 ▼
 Average per agent type                           dist(c) = ‖mu_imm−v_imm‖
         │                                                + ‖mu_fut−v_fut‖
         ▼                                                 │
 Save mu_imm, mu_fut                                       ▼
 → bfe_profiles.json                              Move to c* = argmin dist(c)
```
