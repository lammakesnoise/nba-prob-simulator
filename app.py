from flask import Flask, jsonify, request, send_file
import requests
import numpy as np
from collections import defaultdict
import time

app = Flask(__name__)

# --- Constants ---

NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
}

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
STANDINGS_URL = "https://stats.nba.com/stats/leaguestandingsv3?LeagueID=00&Season=2025-26&SeasonType=Regular+Season"

# Team metadata (static for 2025-26 season)
TEAM_META = {
    1610612737: ("ATL", "Atlanta", "Hawks", "East", "Southeast"),
    1610612738: ("BOS", "Boston", "Celtics", "East", "Atlantic"),
    1610612751: ("BKN", "Brooklyn", "Nets", "East", "Atlantic"),
    1610612766: ("CHA", "Charlotte", "Hornets", "East", "Southeast"),
    1610612741: ("CHI", "Chicago", "Bulls", "East", "Central"),
    1610612739: ("CLE", "Cleveland", "Cavaliers", "East", "Central"),
    1610612742: ("DAL", "Dallas", "Mavericks", "West", "Southwest"),
    1610612743: ("DEN", "Denver", "Nuggets", "West", "Northwest"),
    1610612765: ("DET", "Detroit", "Pistons", "East", "Central"),
    1610612744: ("GSW", "Golden State", "Warriors", "West", "Pacific"),
    1610612745: ("HOU", "Houston", "Rockets", "West", "Southwest"),
    1610612754: ("IND", "Indiana", "Pacers", "East", "Central"),
    1610612746: ("LAC", "Los Angeles", "Clippers", "West", "Pacific"),
    1610612747: ("LAL", "Los Angeles", "Lakers", "West", "Pacific"),
    1610612763: ("MEM", "Memphis", "Grizzlies", "West", "Southwest"),
    1610612748: ("MIA", "Miami", "Heat", "East", "Southeast"),
    1610612749: ("MIL", "Milwaukee", "Bucks", "East", "Central"),
    1610612750: ("MIN", "Minnesota", "Timberwolves", "West", "Northwest"),
    1610612740: ("NOP", "New Orleans", "Pelicans", "West", "Southwest"),
    1610612752: ("NYK", "New York", "Knicks", "East", "Atlantic"),
    1610612760: ("OKC", "Oklahoma City", "Thunder", "West", "Northwest"),
    1610612753: ("ORL", "Orlando", "Magic", "East", "Southeast"),
    1610612755: ("PHI", "Philadelphia", "76ers", "East", "Atlantic"),
    1610612756: ("PHX", "Phoenix", "Suns", "West", "Pacific"),
    1610612757: ("POR", "Portland", "Trail Blazers", "West", "Northwest"),
    1610612758: ("SAC", "Sacramento", "Kings", "West", "Pacific"),
    1610612759: ("SAS", "San Antonio", "Spurs", "West", "Southwest"),
    1610612761: ("TOR", "Toronto", "Raptors", "East", "Atlantic"),
    1610612762: ("UTA", "Utah", "Jazz", "West", "Northwest"),
    1610612764: ("WAS", "Washington", "Wizards", "East", "Southeast"),
}

# --- Global state ---

teams = {}            # team_id -> {tricode, name, city, conference, division, wins, losses, conf_w, conf_l, div_w, div_l, seed}
remaining_games = []  # [(home_id, away_id), ...]
h2h_base = None       # np array shape (n_teams, n_teams)
team_ids = []         # ordered team_ids
team_idx = {}         # team_id -> index
conference_teams = {"East": [], "West": []}
game_is_conf = []
game_is_div = []
season_year = ""  # e.g. "2025-26"
data_fetched_at = ""  # ISO timestamp of last data fetch
team_game_indices = {}  # team_idx -> [(game_index, opponent_tricode, is_home), ...]


def fetch_data():
    """Fetch schedule from CDN, optionally enrich with standings."""
    global teams, remaining_games, h2h_base, team_ids, team_idx
    global conference_teams, game_is_conf, game_is_div, data_fetched_at

    # Reset mutable globals
    teams = {}
    remaining_games = []
    game_is_conf = []
    game_is_div = []
    team_game_indices.clear()
    conference_teams = {"East": [], "West": []}

    # Initialize teams from metadata
    for tid, (tri, city, name, conf, div) in TEAM_META.items():
        teams[tid] = {
            "tricode": tri, "name": name, "city": city,
            "conference": conf, "division": div,
            "wins": 0, "losses": 0,
            "conf_w": 0, "conf_l": 0,
            "div_w": 0, "div_l": 0,
            "seed": 0,
        }

    # Build ordered index
    team_ids = sorted(teams.keys())
    team_idx = {tid: i for i, tid in enumerate(team_ids)}
    n = len(team_ids)

    # Fetch schedule
    print("Fetching schedule from CDN...")
    resp = requests.get(SCHEDULE_URL, headers=NBA_HEADERS, timeout=30)
    resp.raise_for_status()
    global season_year
    schedule = resp.json()["leagueSchedule"]
    season_year = schedule.get("seasonYear", "")

    h2h_base = np.zeros((n, n), dtype=np.int32)
    remaining_games = []
    completed = 0

    for date_entry in schedule["gameDates"]:
        for game in date_entry["games"]:
            gid = game["gameId"]
            if not gid.startswith("002"):
                continue

            status = game["gameStatus"]
            home_id = game["homeTeam"]["teamId"]
            away_id = game["awayTeam"]["teamId"]

            if home_id not in team_idx or away_id not in team_idx:
                continue

            hi = team_idx[home_id]
            ai = team_idx[away_id]
            is_conf = teams[home_id]["conference"] == teams[away_id]["conference"]
            is_div = teams[home_id]["division"] == teams[away_id]["division"]

            if status == 3:
                # Completed game
                home_score = game["homeTeam"]["score"]
                away_score = game["awayTeam"]["score"]
                if home_score > away_score:
                    winner_id, loser_id = home_id, away_id
                    h2h_base[hi][ai] += 1
                else:
                    winner_id, loser_id = away_id, home_id
                    h2h_base[ai][hi] += 1

                teams[winner_id]["wins"] += 1
                teams[loser_id]["losses"] += 1
                if is_conf:
                    teams[winner_id]["conf_w"] += 1
                    teams[loser_id]["conf_l"] += 1
                if is_div:
                    teams[winner_id]["div_w"] += 1
                    teams[loser_id]["div_l"] += 1
                completed += 1

            elif status in (1, 2):
                remaining_games.append((home_id, away_id))

    print(f"Schedule: {completed} completed, {len(remaining_games)} remaining")

    # Compute current seeds per conference
    for conf in ("East", "West"):
        conf_tids = [tid for tid in team_ids if teams[tid]["conference"] == conf]
        conf_tids.sort(key=lambda tid: (-teams[tid]["wins"], teams[tid]["losses"]))
        for rank, tid in enumerate(conf_tids, 1):
            teams[tid]["seed"] = rank

    # Build conference team index lists
    conference_teams["East"] = [team_idx[tid] for tid in team_ids if teams[tid]["conference"] == "East"]
    conference_teams["West"] = [team_idx[tid] for tid in team_ids if teams[tid]["conference"] == "West"]

    # Pre-compute game metadata
    game_is_conf.clear()
    game_is_div.clear()
    team_game_indices.clear()
    for g_idx, (h, a) in enumerate(remaining_games):
        game_is_conf.append(teams[h]["conference"] == teams[a]["conference"])
        game_is_div.append(teams[h]["division"] == teams[a]["division"])
        hi = team_idx[h]
        ai = team_idx[a]
        team_game_indices.setdefault(hi, []).append((g_idx, teams[a]["tricode"], True))
        team_game_indices.setdefault(ai, []).append((g_idx, teams[h]["tricode"], False))

    # Try to get proper seeds from standings API
    try:
        print("Fetching standings from stats.nba.com...")
        resp = requests.get(STANDINGS_URL, headers=NBA_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "resultSets" in data:
            headers_list = data["resultSets"][0]["headers"]
            rows = data["resultSets"][0]["rowSet"]
            for row in rows:
                d = dict(zip(headers_list, row))
                tid = d["TeamID"]
                if tid in teams:
                    teams[tid]["seed"] = d["PlayoffRank"]
            print("Standings enriched with official seeds")
    except Exception as e:
        print(f"Standings API unavailable ({e}), using computed seeds")

    from datetime import datetime, timezone
    data_fetched_at = datetime.now(timezone.utc).isoformat()

    print(f"Ready: {len(teams)} teams, {len(remaining_games)} games to simulate")


def run_simulation(n_sims):
    """Run Monte Carlo simulation. Returns seed counts and per-cell detail stats."""
    n_games = len(remaining_games)
    n_teams = len(team_ids)

    # Base stats arrays
    base_wins = np.array([teams[tid]["wins"] for tid in team_ids], dtype=np.int32)
    base_losses = np.array([teams[tid]["losses"] for tid in team_ids], dtype=np.int32)
    base_conf_w = np.array([teams[tid]["conf_w"] for tid in team_ids], dtype=np.int32)
    base_conf_l = np.array([teams[tid]["conf_l"] for tid in team_ids], dtype=np.int32)
    base_div_w = np.array([teams[tid]["div_w"] for tid in team_ids], dtype=np.int32)
    base_div_l = np.array([teams[tid]["div_l"] for tid in team_ids], dtype=np.int32)

    home_idx = np.array([team_idx[h] for h, a in remaining_games], dtype=np.int32)
    away_idx = np.array([team_idx[a] for h, a in remaining_games], dtype=np.int32)
    is_conf = np.array(game_is_conf, dtype=bool)
    is_div = np.array(game_is_div, dtype=bool)

    seed_counts = {
        "East": np.zeros((n_teams, 10), dtype=np.int32),
        "West": np.zeros((n_teams, 10), dtype=np.int32),
    }

    # Detail tracking: per (team_idx, seed) -> {sum_wins, min_wins, max_wins, game_wins: {game_idx: count}}
    # Use flat dicts keyed by (team_idx, seed_0)
    detail_sum_wins = defaultdict(int)
    detail_min_wins = {}
    detail_max_wins = {}
    # Per-game win counts when team lands at a seed: (team_idx, seed_0, game_idx) -> count
    detail_game_wins = defaultdict(int)

    # All coin flips: 1 = home wins
    all_flips = np.random.randint(0, 2, size=(n_sims, n_games), dtype=np.int8)

    for sim in range(n_sims):
        flips = all_flips[sim]

        wins = base_wins.copy()
        losses = base_losses.copy()
        conf_w = base_conf_w.copy()
        conf_l = base_conf_l.copy()
        div_w = base_div_w.copy()
        div_l = base_div_l.copy()
        h2h = h2h_base.copy()

        for g in range(n_games):
            hi = home_idx[g]
            ai = away_idx[g]
            if flips[g]:
                wins[hi] += 1
                losses[ai] += 1
                h2h[hi][ai] += 1
                if is_conf[g]:
                    conf_w[hi] += 1
                    conf_l[ai] += 1
                if is_div[g]:
                    div_w[hi] += 1
                    div_l[ai] += 1
            else:
                wins[ai] += 1
                losses[hi] += 1
                h2h[ai][hi] += 1
                if is_conf[g]:
                    conf_w[ai] += 1
                    conf_l[hi] += 1
                if is_div[g]:
                    div_w[ai] += 1
                    div_l[hi] += 1

        # Determine division leaders for tiebreakers
        div_leaders = determine_division_leaders(wins, losses, h2h)

        # Rank each conference and record details
        for conf_name, conf_idx_list in conference_teams.items():
            ranked = rank_conference(conf_idx_list, wins, losses, conf_w, conf_l, div_w, div_l, h2h, div_leaders)
            for seed_0, idx in enumerate(ranked[:10]):
                seed_counts[conf_name][idx][seed_0] += 1
                key = (idx, seed_0)
                w = int(wins[idx])
                detail_sum_wins[key] += w
                if key not in detail_min_wins or w < detail_min_wins[key]:
                    detail_min_wins[key] = w
                if key not in detail_max_wins or w > detail_max_wins[key]:
                    detail_max_wins[key] = w
                # Track game outcomes for this team's games
                for g_idx, opp_tri, is_home in team_game_indices.get(idx, []):
                    team_won = (flips[g_idx] == 1) if is_home else (flips[g_idx] == 0)
                    if team_won:
                        detail_game_wins[(idx, seed_0, g_idx)] += 1

    return seed_counts, detail_sum_wins, detail_min_wins, detail_max_wins, detail_game_wins


def determine_division_leaders(wins, losses, h2h):
    """Determine the leader of each division. Returns set of team indices."""
    divisions = defaultdict(list)
    for tid in team_ids:
        idx = team_idx[tid]
        div = teams[tid]["division"]
        total = wins[idx] + losses[idx]
        wpct = wins[idx] / total if total > 0 else 0.0
        divisions[div].append((idx, wpct))

    leaders = set()
    for div, div_teams in divisions.items():
        div_teams.sort(key=lambda x: -x[1])
        best_wpct = div_teams[0][1]
        tied_for_lead = [t for t in div_teams if abs(t[1] - best_wpct) < 1e-9]
        if len(tied_for_lead) == 1:
            leaders.add(tied_for_lead[0][0])
        else:
            # Break division leader tie with H2H among tied
            best_idx = tied_for_lead[0][0]
            best_h2h = -1.0
            for idx, _ in tied_for_lead:
                w = sum(h2h[idx][o] for o, _ in tied_for_lead if o != idx)
                total = sum(h2h[idx][o] + h2h[o][idx] for o, _ in tied_for_lead if o != idx)
                h2h_pct = w / total if total > 0 else 0.5
                if h2h_pct > best_h2h:
                    best_h2h = h2h_pct
                    best_idx = idx
            leaders.add(best_idx)

    return leaders


def rank_conference(conf_indices, wins, losses, conf_w, conf_l, div_w, div_l, h2h, div_leaders):
    """Rank teams in a conference with tiebreakers."""
    team_data = []
    for idx in conf_indices:
        total = wins[idx] + losses[idx]
        wpct = wins[idx] / total if total > 0 else 0.0
        ctotal = conf_w[idx] + conf_l[idx]
        cpct = conf_w[idx] / ctotal if ctotal > 0 else 0.0
        dtotal = div_w[idx] + div_l[idx]
        dpct = div_w[idx] / dtotal if dtotal > 0 else 0.0
        team_data.append((idx, wpct, cpct, dpct))

    team_data.sort(key=lambda x: -x[1])

    result = []
    i = 0
    while i < len(team_data):
        j = i + 1
        while j < len(team_data) and abs(team_data[j][1] - team_data[i][1]) < 1e-9:
            j += 1
        group = team_data[i:j]
        if len(group) == 1:
            result.append(group[0][0])
        else:
            result.extend(break_tie(group, h2h, div_leaders))
        i = j

    return result


def break_tie(tied, h2h, div_leaders):
    """Break ties using official NBA rules. 2-team and 3+ team have different order."""
    if len(tied) == 2:
        return _break_tie_two(tied, h2h, div_leaders)
    else:
        return _break_tie_multi(tied, h2h, div_leaders)


def _break_tie_two(tied, h2h, div_leaders):
    """2-team: (1) H2H, (2) div leader, (3) div record if same div, (4) conf record, (5) random."""
    a, b = tied[0], tied[1]
    idx_a, idx_b = a[0], b[0]

    # (1) Head-to-head
    w_a = h2h[idx_a][idx_b]
    w_b = h2h[idx_b][idx_a]
    if w_a != w_b:
        return [idx_a, idx_b] if w_a > w_b else [idx_b, idx_a]

    # (2) Division leader wins over non-leader
    a_leads = idx_a in div_leaders
    b_leads = idx_b in div_leaders
    if a_leads and not b_leads:
        return [idx_a, idx_b]
    if b_leads and not a_leads:
        return [idx_b, idx_a]

    # (3) Division record (only if same division)
    if teams[team_ids[idx_a]]["division"] == teams[team_ids[idx_b]]["division"]:
        if abs(a[3] - b[3]) > 1e-9:
            return [idx_a, idx_b] if a[3] > b[3] else [idx_b, idx_a]

    # (4) Conference record
    if abs(a[2] - b[2]) > 1e-9:
        return [idx_a, idx_b] if a[2] > b[2] else [idx_b, idx_a]

    # (5-7) Playoff-eligible opponent win%, point differential — not available
    # Fall back to random
    if np.random.random() > 0.5:
        return [idx_a, idx_b]
    return [idx_b, idx_a]


def _break_tie_multi(tied, h2h, div_leaders):
    """3+ teams: (1) div leader, (2) H2H, (3) div record if same div, (4) conf record, (5) random."""
    indices = [t[0] for t in tied]

    # (1) Division leader wins over non-leaders
    leaders_in = [t for t in tied if t[0] in div_leaders]
    non_leaders = [t for t in tied if t[0] not in div_leaders]
    if leaders_in and non_leaders:
        # Rank leaders first, then non-leaders (restart criteria for each subgroup)
        if len(leaders_in) == 1:
            leader_result = [leaders_in[0][0]]
        else:
            leader_result = break_tie(leaders_in, h2h, div_leaders)
        if len(non_leaders) == 1:
            non_leader_result = [non_leaders[0][0]]
        else:
            non_leader_result = break_tie(non_leaders, h2h, div_leaders)
        return leader_result + non_leader_result

    # (2) H2H among all tied teams
    h2h_wpct = {}
    for idx in indices:
        w = sum(h2h[idx][o] for o in indices if o != idx)
        total = sum(h2h[idx][o] + h2h[o][idx] for o in indices if o != idx)
        h2h_wpct[idx] = w / total if total > 0 else 0.5

    # Group by H2H win% — if any differentiation, split and restart
    by_h2h = defaultdict(list)
    for t in tied:
        by_h2h[round(h2h_wpct[t[0]], 6)].append(t)
    if len(by_h2h) > 1:
        result = []
        for _, group in sorted(by_h2h.items(), reverse=True):
            if len(group) == 1:
                result.append(group[0][0])
            else:
                result.extend(break_tie(group, h2h, div_leaders))
        return result

    # (3) Division record (only if ALL in same division)
    divs = set(teams[team_ids[i]]["division"] for i in indices)
    if len(divs) == 1:
        by_div = defaultdict(list)
        for t in tied:
            by_div[round(t[3], 6)].append(t)
        if len(by_div) > 1:
            result = []
            for _, group in sorted(by_div.items(), reverse=True):
                if len(group) == 1:
                    result.append(group[0][0])
                else:
                    result.extend(break_tie(group, h2h, div_leaders))
            return result

    # (4) Conference record
    by_conf = defaultdict(list)
    for t in tied:
        by_conf[round(t[2], 6)].append(t)
    if len(by_conf) > 1:
        result = []
        for _, group in sorted(by_conf.items(), reverse=True):
            if len(group) == 1:
                result.append(group[0][0])
            else:
                result.extend(break_tie(group, h2h, div_leaders))
        return result

    # (5-6) Playoff-eligible opponent win%, point differential — not available
    # Random
    rng = {i: np.random.random() for i in indices}
    return sorted(indices, key=lambda i: -rng[i])


# --- Routes ---

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/teams")
def get_teams():
    result = {"East": [], "West": []}
    for conf in ("East", "West"):
        conf_list = []
        for idx in conference_teams[conf]:
            tid = team_ids[idx]
            t = teams[tid]
            conf_list.append({
                "tricode": t["tricode"],
                "name": f"{t['city']} {t['name']}",
                "wins": t["wins"],
                "losses": t["losses"],
                "conf_w": t["conf_w"],
                "conf_l": t["conf_l"],
                "div_w": t["div_w"],
                "div_l": t["div_l"],
                "seed": t["seed"],
            })
        conf_list.sort(key=lambda x: x["seed"])
        result[conf] = conf_list
    result["remaining_games"] = len(remaining_games)
    result["season"] = season_year
    result["data_fetched_at"] = data_fetched_at
    return jsonify(result)


@app.route("/api/refresh", methods=["POST"])
def refresh_data():
    """Re-fetch NBA schedule and standings data."""
    try:
        fetch_data()
        return jsonify({
            "success": True,
            "remaining_games": len(remaining_games),
            "data_fetched_at": data_fetched_at,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _auto_refresh_if_stale():
    """Re-fetch data if it's older than 1 hour."""
    from datetime import datetime, timezone
    if not data_fetched_at:
        return
    fetched = datetime.fromisoformat(data_fetched_at)
    age = (datetime.now(timezone.utc) - fetched).total_seconds()
    if age > 3600:  # 1 hour
        print(f"Data is {age/3600:.1f}h old, auto-refreshing...")
        fetch_data()


@app.route("/api/simulate", methods=["POST"])
def simulate():
    _auto_refresh_if_stale()

    data = request.get_json()
    n_sims = min(data.get("n_simulations", 10000), 50000)

    start = time.time()
    seed_counts, detail_sum, detail_min, detail_max, detail_gw = run_simulation(n_sims)
    elapsed = time.time() - start

    result = {"East": {}, "West": {}}
    details = {"East": {}, "West": {}}

    for conf in ("East", "West"):
        for idx in conference_teams[conf]:
            tid = team_ids[idx]
            tri = teams[tid]["tricode"]
            counts = seed_counts[conf][idx]
            probs = (counts / n_sims * 100).tolist()
            result[conf][tri] = probs

            # Build details for seeds with meaningful probability
            team_details = {}
            for seed_0 in range(10):
                count = int(counts[seed_0])
                if count < max(5, n_sims * 0.005):
                    continue
                key = (idx, seed_0)
                avg_w = detail_sum[key] / count
                total_games = 82
                avg_l = total_games - avg_w

                # Find most impactful games (win% furthest from 50%)
                game_info = []
                for g_idx, opp_tri, is_home in team_game_indices.get(idx, []):
                    gw = detail_gw.get((idx, seed_0, g_idx), 0)
                    win_pct = gw / count * 100
                    impact = abs(win_pct - 50)
                    game_info.append({
                        "vs": opp_tri,
                        "home": is_home,
                        "win_pct": round(win_pct, 1),
                        "impact": impact,
                    })
                # Sort by impact descending, take top 5
                game_info.sort(key=lambda x: -x["impact"])
                top_games = game_info[:5]
                for g in top_games:
                    del g["impact"]

                team_details[str(seed_0)] = {
                    "avg_wins": round(avg_w, 1),
                    "avg_losses": round(avg_l, 1),
                    "min_wins": detail_min.get(key, 0),
                    "max_wins": detail_max.get(key, 0),
                    "count": count,
                    "games": top_games,
                }
            details[conf][tri] = team_details

    return jsonify({
        "probabilities": result,
        "details": details,
        "n_simulations": n_sims,
        "elapsed_seconds": round(elapsed, 2),
        "remaining_games": len(remaining_games),
        "data_fetched_at": data_fetched_at,
    })


@app.route("/api/h2h")
def get_h2h():
    """Return head-to-head records. Optional ?team=TRI to filter."""
    tri_filter = request.args.get("team", "").upper()

    # Build tricode -> index lookup
    tri_to_idx = {}
    for tid, t in teams.items():
        tri_to_idx[t["tricode"]] = team_idx[tid]

    # Build all tricodes list in a stable order
    all_tricodes = sorted(tri_to_idx.keys())

    if tri_filter and tri_filter in tri_to_idx:
        # Return H2H for one team vs all others
        idx_a = tri_to_idx[tri_filter]
        records = {}
        for tri_b in all_tricodes:
            if tri_b == tri_filter:
                continue
            idx_b = tri_to_idx[tri_b]
            w = int(h2h_base[idx_a][idx_b])
            l = int(h2h_base[idx_b][idx_a])
            if w + l > 0:
                records[tri_b] = {"wins": w, "losses": l}
        return jsonify({"team": tri_filter, "records": records})
    else:
        # Return full H2H matrix (all pairs with at least 1 game)
        result = {}
        for tri_a in all_tricodes:
            idx_a = tri_to_idx[tri_a]
            team_records = {}
            for tri_b in all_tricodes:
                if tri_b == tri_a:
                    continue
                idx_b = tri_to_idx[tri_b]
                w = int(h2h_base[idx_a][idx_b])
                l = int(h2h_base[idx_b][idx_a])
                if w + l > 0:
                    team_records[tri_b] = {"wins": w, "losses": l}
            result[tri_a] = team_records
        return jsonify({"records": result, "teams": all_tricodes})


if __name__ == "__main__":
    import os
    fetch_data()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
