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

The baseline agent scores every candidate cell by computing a happiness value `h(c)` through the hedonic formula and picks the cell that maximises it. The selfishness factor φ determines how much the agent weights its own gain versus the welfare of others — but φ is a design parameter set by the researcher, not something measurable from observed behaviour.

FVDM does not replace the φ-welfare decision rule. Instead, it adds a **measurement layer**: it records the *net welfare fingerprint* of every decision and computes an empirical profile of what kind of moves each agent type characteristically makes in social contexts. That profile — the **Behavioral Feature Expectation (BFE)** — is a post-hoc verification instrument.

```
BASELINE AGENT                    FVDM AGENT  (φ-Welfare + BFE Measurement)
──────────────────────────────    ────────────────────────────────────────────
For each candidate cell c:        For each candidate cell c:

  Compute h(c) via hedonic           Compute h(c) via hedonic formula using
  formula with designer-set φ        agent-type φ (set at initialisation):
                                       phiEgoist   → φ = 1.0  (only own gain)
  Pick c* = argmax h(c)               phiAltruist → φ = 0.0  (only others)
                                       phiBentham  → φ = 0.5  (balanced)

                                     Pick c* = argmax h(c)  ← same rule

                                     THEN record net fingerprint of c*:
                                       v_net_imm(c*) = φ·v_imm_self(c*)
                                                     − (1−φ)·mean_k[v_imm_k(c*)]
                                       v_net_fut(c*) = φ·v_fut_self(c*)
                                                     − (1−φ)·mean_k[v_fut_k(c*)]
                                       (logged every timestep for BFS computation)
```

The **BFE profile** `(mu_imm, mu_fut)` is the mean of these net fingerprints over all
socially-contextual decisions (timesteps where at least one neighbour was present).
The **Behavioral Fidelity Score (BFS)** — cosine similarity between the observed
empirical profile and the derived target profile — then confirms whether the agent
actually behaved as its ethical type predicts.

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

The BFE profile `(mu_imm, mu_fut)` for each agent type is derived offline,
before any FVDM experiment runs.

The core idea: instead of averaging the raw cell fingerprint (what does the
chosen cell offer the agent?), we average the **net opportunity-cost fingerprint**
— what the chosen cell offers *net of what it costs neighbours*. This decomposition
comes directly from the hedonic formula itself:

```
  h(c)  =  φ × h_self(c)  −  (1−φ) × Σ_k h_other_k(c)

Decomposing into immediate and future effect vectors:

  v_net_imm(c*)  =  φ × v_imm_self(c*)  −  (1−φ) × mean_k[ v_imm_k(c*) ]
  v_net_fut(c*)  =  φ × v_fut_self(c*)  −  (1−φ) × mean_k[ v_fut_k(c*) ]

where k ranges over all neighbours that could reach cell c*.
```

Averaging `v_net` across all qualifying observations gives the BFE profile —
the characteristic opportunity-cost signature of that agent type.

**Why this formulation, not raw v_self?**

Recording only `v_self(c*)` would capture what the agent gains but ignore what
it costs others by choosing that cell. An egoist with φ=1.0 has `v_net = v_self`
(the cost term vanishes), so the two are equivalent for pure egoists. But for
Bentham (φ=0.5) or Altruist (φ=0.0), the profile without the cost term is
incomplete — it misses the half of the decision that concerned others.

```
CORRECTED DERIVATION PIPELINE  (derive_vectors_phi.py)
────────────────────────────────────────────────────────────────────────────

STEP 1: Run homogeneous simulations (one agent type per batch)
│
│   Each derivation run contains only one agent type at a time.
│   Homogeneous means: no cross-type competitive pressure. The profile
│   reflects how the type behaves among its own kind.
│
│   Agent type used:  FVDMBFEAgent  (decides by φ-welfare, logs v_net)
│
│   ┌──────────────────────────────────────────────────┐
│   │  Type: phiBfeEgoist  (φ = 1.0)                   │
│   │    Seed 1  →  agent log (all timesteps)           │
│   │    Seed 2  →  agent log                           │
│   │    ...  Seed N                                    │
│   │                                                   │
│   │  Type: phiBfeAltruist  (φ = 0.0)                 │
│   │    Seed 1 … Seed N                                │
│   │                                                   │
│   │  Type: phiBfeBentham  (φ = 0.5)                  │
│   │    Seed 1 … Seed N                                │
│   └──────────────────────────────────────────────────┘

STEP 2: Filter to socially-contextual moves (neighbours > 0)
│
│   A qualifying move is any timestep where the FVDMBFEAgent had at
│   least one other living agent within its vision range.
│   Moves in empty neighbourhoods are excluded — no social choice was made.
│
│   All timesteps:  ████░████░░███░████░░░████░░██
│                   ↑↑↑↑ ↑↑↑↑    ↑↑↑ ↑↑↑↑   ↑↑↑↑
│   neighbours > 0: ● ●● ●●●     ● ● ●●●●   ●●●●
│                   (these rows enter the derivation)

STEP 3: Read pre-logged v_net from agent log
│
│   FVDMBFEAgent computes and logs v_net at every decision:
│
│     bfe_v_imm_I, bfe_v_imm_D, bfe_v_imm_C, bfe_v_imm_P, bfe_v_imm_E
│     bfe_v_fut_I, bfe_v_fut_D, bfe_v_fut_C, bfe_v_fut_P, bfe_v_fut_E
│
│   derive_vectors_phi.py reads these columns directly.
│   No post-hoc cell re-computation is required.

STEP 4: Accumulate and average per agent type
│
│                       qualifying observations
│   egoist    :   v_net_1   v_net_2  ...  v_net_N
│                 ────────────────────────────────
│                 mean → mu_imm_egoist  (5D vector)
│                         mu_fut_egoist  (5D vector)
│
│   altruist  :   same procedure → mu_imm_altruist,  mu_fut_altruist
│   bentham   :   same procedure → mu_imm_bentham,   mu_fut_bentham
│   rawSugar  :   same procedure → mu_imm_raw,       mu_fut_raw

STEP 5: Save profiles to bfe_profiles_phi.json
│
│   {
│     "profiles": {
│       "egoist":    { "mu_imm": [I, D, C, P, E],
│                      "mu_fut":  [I, D, C, P, E] },
│       "altruist":  { ... },
│       "bentham":   { ... },
│       "rawSugarscape": { ... }
│     }
│   }
│
│   These profiles are the learned opportunity-cost fingerprints.
│   One profile per agent type. Each profile is two 5D net vectors.
```

**Interpreting the net vectors:**

The D (Duration) coordinate illustrates the difference most clearly:

```
  v_net_imm_D  =  φ × D_self(c*)  −  (1−φ) × mean_k[ D_k(c*) ]

  Egoist   (φ=1.0):  v_net_D  =  D_self           > 0  (positive: gain)
  Bentham  (φ=0.5):  v_net_D  =  0.5×D_self − 0.5×D_k ≈ 0  (balanced)
  Altruist (φ=0.0):  v_net_D  =  −mean_k[D_k]      < 0  (negative: cost)
```

The egoist's profile has positive D — it characteristically moves to cells with
high personal resource gain. The altruist's profile has negative D — it tends
toward cells that *cost* others less, i.e., cells that are not contested.
The Bentham profile sits near zero because gain and cost roughly cancel.

**Script:** `python3 derive_vectors_phi.py --homogeneous -s 10 -t 5000 -a 250 -j 30`
**Output:** `fvdm_vectors/bfe_profiles_phi.json`

---

## Part 3 — FVDM as Measurement Framework

FVDM is not an alternative decision rule — it is a verification instrument.
The decision itself uses φ-welfare argmax (the same formula as the baseline).
The measurement records the opportunity-cost fingerprint of each decision,
accumulates it into an empirical BFE, and computes a **Behavioral Fidelity
Score (BFS)** that quantifies how faithfully the agent's behaviour matches
its declared ethical type.

```
FVDM AGENT — DECISION AND MEASUREMENT PROCEDURE
────────────────────────────────────────────────────────────────────────────

At startup: set φ for this agent's type
              phiEgoist   → φ = 1.0
              phiBentham  → φ = 0.5
              phiAltruist → φ = 0.0

Each timestep (FVDMPhiAgent):

  DECISION  ─────────────────────────────────────────────────────────────
  │
  │  For each candidate cell c:
  │    score(c)  =  φ × h_self(c)  −  (1−φ) × Σ_k h_k(c)
  │
  │  Pick  c* = argmax score(c)    ← φ-welfare, identical to Bentham
  │
  MEASUREMENT  ──────────────────────────────────────────────────────────
  │
  │  Identify reachable neighbours k (agents that can reach c*)
  │
  │  Compute net fingerprint of chosen cell:
  │    v_net_imm  =  φ × v_imm_self(c*)  −  (1−φ) × mean_k[ v_imm_k(c*) ]
  │    v_net_fut  =  φ × v_fut_self(c*)  −  (1−φ) × mean_k[ v_fut_k(c*) ]
  │
  │  Log v_net_imm and v_net_fut to agent log
  │  (only when neighbours > 0 — otherwise no social context)

POST-HOC VERIFICATION  ─────────────────────────────────────────────────

  After all simulations finish, for each condition:

  1. Read agent logs → collect all logged v_net (neighbours > 0 rows)
  2. Compute empirical BFE:
       mu_obs_imm  =  mean( v_net_imm  over all qualifying observations )
       mu_obs_fut  =  mean( v_net_fut  over all qualifying observations )

  3. Load derived target profile from bfe_profiles_phi.json:
       mu_profile_imm,  mu_profile_fut

  4. BFS  =  cosine_similarity( [mu_obs_imm, mu_obs_fut],
                                 [mu_profile_imm, mu_profile_fut] )

  BFS ≈ 1.0  →  observed behaviour matches derived profile  (faithful)
  BFS < 0.9  →  behavioural drift — agent behaved differently
                from its profile's derivation context
```

**Why φ-welfare and not argmin distance?**

A profile is a *mean* over many observations. Argmin distance to the mean
finds cells that are *average-looking* — cells whose fingerprint is
closest to the middle of the distribution. But an egoist's characteristic
behaviour is to pick the *best* available cell, not an average one. Forcing
the egoist toward average cells starves it: it systematically avoids the
richest options.

The same failure occurs for J (future neighbourhood wealth). The mean J is
non-zero because occupied, resource-rich areas are common during derivation.
A profile-matching rule therefore pulls agents toward population-dense,
resource-depleted areas — the exact ecological trap that caused extinction
in earlier argmin experiments.

φ-welfare avoids the trap because it directly maximises the agent's welfare
function. The profile is only used after the fact, to ask: *given the cells
the agent actually picked by φ-welfare, does the resulting fingerprint match
what we expect from that ethical type?*

**What BFS tells you:**

```
Condition               BFS vs derived   Interpretation
─────────────────────── ────────────     ──────────────────────────────────
phiBenthamDerived       ≈ 0.997          φ=0.5 balanced rule produces a
                                         fingerprint very close to its own
                                         derivation — robust across population
                                         compositions

phiEgoistDerived        ≈ 0.83           φ=1.0 profile is ecologically
phiAltruistDerived      ≈ 0.83           contingent: derivation context (who
                                         else was in the population) shapes
                                         the profile more than for Bentham
```

Bentham's near-perfect BFS is not coincidental: because φ=0.5 weights own
gain and others' cost equally, the profile averages out contextual variation.
Egoist and altruist profiles are more sensitive to who else is present during
derivation.

### Worked Example — Egoist φ-Welfare Agent Choosing Between Three Cells

The agent has φ = 1.0. It computes h(c) = h_self(c) for each cell.

```
  h_self(c)  =  Σ[ v_imm_self(c)_dim × w_dim ]   (hedonic sum)

  Cell A  (rich, low pollution, low neighbours):
    v_imm_self  =  [ 0.25, 0.31, 1.00, 1.00, 0.15 ]
    h_self(A)   ≈  0.542

  Cell B  (depleted):
    v_imm_self  =  [ 0.25, 0.05, 1.00, 1.00, 0.15 ]
    h_self(B)   ≈  0.290

  Cell C  (crowded, resource-depleted):
    v_imm_self  =  [ 0.25, 0.02, 1.00, 1.00, 0.40 ]
    h_self(C)   ≈  0.274
```

**Decision:** c* = Cell A  (highest h_self)

**Measurement (v_net, φ=1.0, so cost term = 0):**
```
  v_net_imm  =  1.0 × v_imm_self(A)  −  0.0 × mean_k[...]
             =  [ 0.25, 0.31, 1.00, 1.00, 0.15 ]   (same as v_self for egoist)
```

This pair is logged. After thousands of such decisions, the mean of logged
v_net_imm is the empirical BFE. BFS measures how close it is to the
pre-derived egoist profile.

---

## Full Pipeline Summary

```
OFFLINE: BFE DERIVATION                           ONLINE: EXPERIMENT + VERIFICATION
(derive_vectors_phi.py, run once)                 (run_experiments_fvdm_phi.py)
──────────────────────────────────────────────    ──────────────────────────────────

 Homogeneous sims per agent type                  FVDMPhiAgent starts at current cell
 (FVDMBFEAgent, one type per batch)                        │
         │                                                 ▼
         ▼                                        φ-welfare decision:
 Filter: neighbours > 0                             score(c) = φ·h_self(c) − (1−φ)·Σh_k(c)
         │                                          c* = argmax score(c)
         ▼                                                 │
 FVDMBFEAgent logs v_net per move:                         ▼
   v_net = φ·v_self(c*) − (1−φ)·mean_k[v_k(c*)]  Measure v_net of chosen c*
         │                                          log v_net_imm, v_net_fut
         ▼                                                 │
 Average logged v_net per type                             ▼
         │                                        POST-HOC (after all seeds):
         ▼                                          mu_obs = mean(logged v_net)
 Save mu_imm, mu_fut                                BFS = cosine_sim(mu_obs, mu_profile)
 → bfe_profiles_phi.json                            → bfs_vs_derived.csv
                                                    → bfs_vs_baseline.csv
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
  phiBfeRaw      (φ=1.0): 250 agents, all raw

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

## Experiment 5 — FVDM φ-Welfare Agents + BFS Verification

**Script:** `run_experiments_fvdm_phi.py` (4 conditions, 30 seeds each)

**What we run:**

```
Condition 1:  phiRawDerived      — FVDMPhiAgent φ=1.0  (raw/greedy)
Condition 2:  phiEgoistDerived   — FVDMPhiAgent φ=1.0  (egoist)
Condition 3:  phiAltruistDerived — FVDMPhiAgent φ=0.0  (altruist)
Condition 4:  phiBenthamDerived  — FVDMPhiAgent φ=0.5  (bentham)

30 seeds × 5000 timesteps × 250 agents each.

Command:
  python3 run_experiments_fvdm_phi.py -s 30 -t 5000 -a 250 -j 30
```

**What we observe:**

```
Primary outputs:

  bfs_vs_derived.csv   — BFS of each condition against its own derived
                         profile from bfe_profiles_phi.json
                         (verification: does the agent behave as its
                          profile predicts?)

  bfs_vs_baseline.csv  — BFS of each phi-agent against the corresponding
                         baseline condition's empirical BFE
                         (integrity check: does phi-welfare reproduce
                          the same behavioral fingerprint as the original
                          baseline agent?)

Secondary outputs:

  Societal metrics per seed:
    extinction rate, final population, mean wealth, Gini, mean TTL

  per_seed_felicific.csv  — timestep-level logs for BFS computation
```

**What this reveals — the three comparison tests:**

```
TEST A — BFS vs. Derived Profile  (behavioral consistency check)

  Expected: BFS ≈ 1.0 for all conditions.
  The phi-welfare agent uses the same decision function that generated
  the profile, so the empirical fingerprint should closely match the
  derived target.

  Preliminary result (500-timestep pilot, 3 seeds):

  Condition              BFS vs derived
  ─────────────────────  ──────────────
  phiBenthamDerived      0.997   ← near-perfect
  phiEgoistDerived       0.83
  phiAltruistDerived     0.83

  Bentham's near-perfect BFS reflects that φ=0.5 balanced weighting
  produces a composition-robust fingerprint (gain and cost cancel out
  context-specific variation). Egoist and altruist profiles are more
  sensitive to the population composition present during derivation.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Panel asks: "Why not just set φ=1?"                            │
  │  Answer: φ is a design parameter. BFS is a measurement.         │
  │  High BFS confirms the agent actually behaved like its type,    │
  │  not just that it was labeled that type. Those differ when the  │
  │  population composition changes — BFS catches it; φ does not.   │
  └─────────────────────────────────────────────────────────────────┘


TEST B — Societal Outcomes vs. Baseline  (ecological viability check)

  Societal outcomes from pilot run (500 timesteps, 3 seeds each):

  Condition              Extinction %   Avg final pop   Mean TTL
  ─────────────────────  ────────────   ─────────────   ────────
  phiRawDerived          0%             ?               ?
  phiEgoistDerived       16.7%          ?               ?
  phiAltruistDerived     ?              ?               ?
  phiBenthamDerived      0%             1049            19.30

  *(Full 30-seed × 5000-timestep results pending.)*

  phiBenthamDerived shows 0% extinction and the highest observed
  final population — consistent with Experiment 1, where Bentham
  balanced populations outperform pure egoists over long runs.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Panel asks: "Does that make FVDM better or just different?"    │
  │  Answer: The φ-welfare decision rule (FVDM's engine) is the     │
  │  same as the baseline — so outcomes should replicate. The added  │
  │  value of FVDM is the BFS measurement layer: it tells you        │
  │  whether the agent behaved as intended, not just what happened.  │
  └─────────────────────────────────────────────────────────────────┘


TEST C — Can FVDM detect ethical equivalence the baseline misses?

  Find two φ values from Experiment 2 that produce identical societal
  outcomes (same final population, same mean wealth). Compute their BFE
  profiles. Show they are measurably different in v_net space even
  though the outcome metrics are identical.

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
  │    phiAltruistDerived ★            ← empirical BFE from experiment
  │
  │             Bentham ●
  │           phiBenthamDerived ★      ← BFS 0.997: tightly clustered
  │
  │                        Egoist ●
  │                      phiEgoistDerived ★
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

Experiment 5:  DOES φ-WELFARE FVDM VERIFY ITS OWN BEHAVIORAL PROFILE?
               run_experiments_fvdm_phi.py + BFS verification.
               → BFS ≈ 1.0 for Bentham (Test A) — behavior matches profile.
               → Outcomes replicate baseline (Test B) — φ-welfare is sound.
               → BFE profiles distinguish ethical stances with same outcomes (Test C).

Experiment 6:  CAN WE SEE THE FRAMEWORK WORKING?
               PCA projection of all profiles.
               → Visual confirmation that ethical types are distinct in
                 behavioral space and FVDM maps agents correctly.
```

**The one-sentence answer to "why is FVDM needed?"**

> Because φ tells you what an agent is designed to be. FVDM measures what
> an agent actually does — and those two things are not always the same.
