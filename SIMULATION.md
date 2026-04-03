# NBA Seeding Probability Simulator — Simulation Logic

## Overview

This simulator uses a **Monte Carlo method** to estimate the probability of each NBA team finishing at each playoff seed (1–10) in their conference. It runs thousands of independent simulated seasons, where every remaining game is decided by a fair 50/50 coin flip, then aggregates the results into a probability distribution.

---

## Phase 1: Data Ingestion

### Source data

On startup, the app fetches two datasets from the NBA:

1. **Schedule** (`cdn.nba.com` — `scheduleLeagueV2.json`)
   Contains every game of the season grouped by date. Each game entry includes the home and away team IDs, the game status, and (for completed games) the final score.

2. **Standings** (`stats.nba.com` — `leaguestandingsv3`, optional enrichment)
   Provides official playoff seedings. Used only to set the "current seed" display column; the simulation does not depend on it.

### Game filtering

Only **regular-season games** are included. These are identified by a `gameId` prefix of `"002"` (the NBA encodes game type in the third character: `1` = preseason, `2` = regular season, `4` = playoffs).

Games are classified by status:
- `gameStatus == 3` → **Completed**. Used to build base records and head-to-head history.
- `gameStatus == 1 or 2` → **Remaining** (scheduled or in-progress). These are the games the simulator will decide.

### What is extracted from completed games

For each completed game, the app determines the winner (by comparing `homeTeam.score` vs `awayTeam.score`) and updates:

| Stat | Description |
|------|-------------|
| `wins` / `losses` | Overall W-L record |
| `conf_w` / `conf_l` | Conference record (only if both teams are in the same conference) |
| `div_w` / `div_l` | Division record (only if both teams are in the same division) |
| `h2h_base[i][j]` | Head-to-head matrix: number of wins team `i` has over team `j` |

Conference and division membership is determined from a hardcoded lookup table (`TEAM_META`) containing all 30 NBA teams and their 2025–26 conference/division assignments.

### Pre-computation for remaining games

For each remaining game, the app pre-computes and stores:
- `game_is_conf[g]` — boolean: is this a conference game?
- `game_is_div[g]` — boolean: is this a division game?
- `team_game_indices[team]` — list of `(game_index, opponent_tricode, is_home)` for each team's remaining games (used for tooltip detail tracking)

---

## Phase 2: Monte Carlo Simulation

### Core loop

Given `N` simulations (user-selectable: 1,000 / 5,000 / 10,000 / 25,000):

1. **Pre-generate all coin flips** as a NumPy matrix of shape `(N, num_remaining_games)` with values `{0, 1}` — where `1` means the home team wins.

2. **For each simulation** (`sim = 0 .. N-1`):

   a. **Copy base stats** — start with the current W-L, conference, division, and H2H records from completed games.

   b. **Resolve each remaining game** — iterate through all remaining games. For game `g`:
      - If `flips[sim][g] == 1`: home team wins.
      - If `flips[sim][g] == 0`: away team wins.
      - Update the winner's `wins` and loser's `losses`.
      - Update the H2H matrix: `h2h[winner][loser] += 1`.
      - If the game is a conference game, update `conf_w` / `conf_l`.
      - If the game is a division game, update `div_w` / `div_l`.

   c. **Rank each conference** — call `rank_conference()` for the East and West separately (see Phase 3 below).

   d. **Record the outcome** — for each team that finishes in the top 10 of their conference, increment `seed_counts[conference][team][seed]`.

   e. **Record detail stats** (for tooltip summaries):
      - Accumulate the team's final win total for averaging.
      - Track min/max win totals seen for this (team, seed) combination.
      - For each of the team's remaining games, record whether the team won that game in simulations where they landed at this seed.

### Performance characteristics

- Coin flips are generated in bulk via `np.random.randint` (vectorized, fast).
- The per-simulation game loop runs in pure Python (~96 iterations per sim at this point in the season).
- 10,000 simulations complete in approximately 2 seconds.

---

## Phase 3: Conference Ranking and Tiebreakers

After all remaining games are resolved in a simulation, each conference's 15 teams must be ranked 1–15. The top 10 positions are recorded (seeds 1–6 are direct playoff berths; seeds 7–10 enter the Play-In Tournament).

### Step 1: Sort by overall win percentage

All 15 teams in a conference are sorted by **overall win percentage** (wins / total games) in descending order. Teams with distinct win percentages are immediately assigned their rank.

### Step 2: Identify tied groups

Teams with **identical win percentages** (compared with a tolerance of `1e-9` to handle floating-point arithmetic) form a tied group. Each tied group is resolved independently using the tiebreaker cascade below.

### Tiebreaker cascade

The `break_tie()` function receives a group of 2 or more teams with the same win percentage and attempts to fully order them. Tiebreakers are applied **in sequence** — each tiebreaker is only used if the previous one fails to produce a complete ordering.

#### Tiebreaker 1: Head-to-head record among tied teams

For each team in the tied group, compute their **win percentage in games played against the other tied teams only**:

```
h2h_wpct[team] = (wins vs other tied teams) / (total games vs other tied teams)
```

This includes both completed-season H2H results *and* the simulated game outcomes for the current simulation run (since the H2H matrix is updated with each coin flip).

- If all teams have **distinct** H2H win percentages → order by H2H win% descending. Tie resolved.
- If any teams share the same H2H win% → proceed to tiebreaker 2.

**Important detail:** For a 2-team tie, this is simply their season series. For a 3+ team tie, this is the **aggregate** record against all other teams in the tied group, not pairwise elimination.

#### Tiebreaker 2: Conference record

For each tied team, use their **conference win percentage**:

```
conf_pct[team] = conf_wins / (conf_wins + conf_losses)
```

- If all tied teams have distinct conference win percentages → order by conference win% descending. Tie resolved.
- If any teams share the same conference win% → proceed to tiebreaker 3.

#### Tiebreaker 3: Division record (conditional)

This tiebreaker **only applies if all tied teams are in the same division**. If the tied teams span multiple divisions, this step is skipped entirely.

For each tied team, use their **division win percentage**:

```
div_pct[team] = div_wins / (div_wins + div_losses)
```

- If all tied teams have distinct division win percentages → order by division win% descending. Tie resolved.
- If any teams share the same division win%, or teams are in different divisions → proceed to tiebreaker 4.

#### Tiebreaker 4: Random coin flip (composite fallback)

If all previous tiebreakers fail to produce a complete ordering, the remaining teams are sorted by a **composite key** that combines all prior tiebreaker values with a random number:

```
sort key = (-h2h_wpct, -conf_pct, -random_value)
```

Each team receives an independent random value from `np.random.random()` (uniform [0, 1)). This ensures:
- Teams that were partially separated by earlier tiebreakers retain their relative order.
- Teams that are still tied after all deterministic tiebreakers are separated randomly.

This is equivalent to flipping a coin among the remaining tied teams.

### Edge cases in tiebreaker resolution

| Scenario | Behavior |
|----------|----------|
| 2-team tie | H2H is their direct season series (e.g., 2-1 vs 1-2) |
| 3+ team tie with H2H | Aggregate H2H among all tied teams (not round-robin elimination) |
| No H2H games played | H2H win% defaults to 0.5 (neutral) |
| Cross-division tie | Division record tiebreaker is skipped entirely |
| All tiebreakers produce ties | Random coin flip decides |

---

## Phase 4: Result Aggregation

### Probability matrix

After all `N` simulations complete, the raw seed counts are converted to percentages:

```
probability[team][seed] = seed_counts[team][seed] / N × 100
```

This produces a 15×10 matrix per conference (15 teams × 10 trackable seeds), where each cell represents the percentage of simulations in which that team finished at that seed.

### Tooltip detail data

For each (team, seed) combination with at least 0.5% probability (or minimum 5 occurrences), the API returns:

| Field | Description |
|-------|-------------|
| `avg_wins` | Average final win total across simulations where the team landed at this seed |
| `avg_losses` | `82 - avg_wins` |
| `min_wins` / `max_wins` | Range of win totals observed |
| `count` | Number of simulations producing this outcome |
| `games` | Top 5 most impactful remaining games (see below) |

### Key remaining games (tooltip)

For each of the team's remaining games, the app tracks how often the team **won** that game in simulations where they finished at the given seed. The **impact** of a game is measured as:

```
impact = |win_pct_in_this_seed_outcome - 50%|
```

Games where the win rate deviates most from the baseline 50% are the most "predictive" of the seeding outcome. For example, if a team wins a particular game in 78% of simulations where they finish as the #4 seed (vs. the 50% base rate), that game is strongly correlated with achieving the #4 seed.

The top 5 games by impact are returned, showing the opponent, home/away status, and conditional win percentage.

---

## Simplifications and Known Limitations

| Simplification | Impact |
|----------------|--------|
| **All games are 50/50 coin flips** | Does not account for team strength, home-court advantage, rest days, injuries, or motivation. This is by design — the simulator answers "what are the seeding odds given pure randomness?" |
| **Simplified tiebreaker rules** | The real NBA tiebreaker rules include additional steps (e.g., record vs. playoff-eligible teams, point differential, strength of schedule). This simulator implements the four most impactful tiebreakers. |
| **Aggregate H2H for multi-team ties** | The real NBA uses a more complex procedure for 3+ team ties that may involve reducing to pairwise comparisons after eliminating teams. This simulator uses a single aggregate H2H pass. |
| **No Play-In Tournament simulation** | The simulator determines seeds 7–10 but does not simulate the Play-In games themselves. |
| **82-game season assumed** | The total games denominator is hardcoded to 82 for average record calculations. |
