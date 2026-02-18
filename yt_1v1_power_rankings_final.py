import json
import os
import random
import csv
import math
from collections import defaultdict, deque

import numpy as np

SAVE_FILE = "games.json"

# ==============================
# OVERVIEW
# PURE NET RATING via LEAST SQUARES (TRUE FIX)
# - margin-only
# - opponent difficulty via rating differences
# - order-independent
# - disconnected components solved separately + centered per component
#
# Quality/Polish Pack:
# (4) Searchable delete
# (5) Edit a game
# (7) Value-tier leaderboard shows tier sizes + cutoffs
# (8) Player comparison card
# (9) Strength-of-schedule (avg opponent rating faced; doesn't change ratings)
# (10) Non-arbitrary certainty % from regression standard errors:
#      certainty = Phi(|r|/SE) where r is rating (points vs component avg)
# (11) Undo last add (removes most recently added game)
# ==============================


# ==============================
# HELPERS (math + names)
# ==============================

def _phi(z: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _certainty_percent(r_pts: float, se_pts: float) -> float:
    """
    Probability (0..100) that the true rating is on the same side of 0
    as the estimate, using regression SE:
      certainty = Phi(|r|/SE) * 100
    """
    if se_pts is None or se_pts <= 1e-12:
        return 50.0 if abs(r_pts) <= 1e-12 else 100.0
    return float(_phi(abs(r_pts) / se_pts) * 100.0)


def _resolve_name_case_insensitive(name_in, known_names):
    if name_in in known_names:
        return name_in
    lower_map = {p.lower(): p for p in known_names}
    return lower_map.get(name_in.lower())


# ==============================
# MODEL
# ==============================

class YT1v1Ranker:
    def __init__(self):
        self.ratings = defaultdict(float)          # normalized (per component mean 0)
        self.rating_se = defaultdict(lambda: None) # normalized SE (same unit as ratings)
        self.games_count = defaultdict(int)

        self.components = []   # list[list[str]] sorted largest->smallest
        self.membership = {}   # player -> component_id (1-based)

    @staticmethod
    def _build_graph_and_counts(games):
        graph = defaultdict(set)
        games_count = defaultdict(int)

        for g in games:
            p1, p2 = g["p1"], g["p2"]
            graph[p1].add(p2)
            graph[p2].add(p1)
            games_count[p1] += 1
            games_count[p2] += 1

        return graph, games_count

    @staticmethod
    def _connected_components(graph):
        visited = set()
        comps = []

        for node in graph.keys():
            if node in visited:
                continue
            q = deque([node])
            visited.add(node)
            comp = []
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nxt in graph[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            comps.append(comp)

        return comps

    @staticmethod
    def _solve_component_least_squares_with_se(component_players, component_games):
        n = len(component_players)
        if n == 0:
            return {}, {}
        if n == 1:
            p = component_players[0]
            return {p: 0.0}, {p: 0.0}

        idx = {p: i for i, p in enumerate(component_players)}
        m = len(component_games)
        if m == 0:
            return {p: 0.0 for p in component_players}, {p: 0.0 for p in component_players}

        A = np.zeros((m, n), dtype=float)
        b = np.zeros((m,), dtype=float)

        for row, g in enumerate(component_games):
            p1, p2 = g["p1"], g["p2"]
            s1, s2 = g["s1"], g["s2"]
            A[row, idx[p1]] = 1.0
            A[row, idx[p2]] = -1.0
            b[row] = (s1 - s2) / 30.0

        r, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)

        # Center (ratings identifiable up to constant)
        C = np.eye(n) - (1.0 / n) * np.ones((n, n))
        r_center = C @ r

        # rss
        if residuals is not None and len(residuals) > 0:
            rss = float(residuals[0])
        else:
            res = (A @ r) - b
            rss = float(res.T @ res)

        dof = max(m - rank, 0)
        sigma2 = (rss / dof) if dof > 0 else 0.0

        AtA = A.T @ A
        Cov_r = sigma2 * np.linalg.pinv(AtA)
        Cov_center = C @ Cov_r @ C.T
        se_center = np.sqrt(np.maximum(np.diag(Cov_center), 0.0))

        ratings_centered = {p: float(r_center[idx[p]]) for p in component_players}
        se_centered = {p: float(se_center[idx[p]]) for p in component_players}
        return ratings_centered, se_centered

    def train(self, games):
        self.ratings = defaultdict(float)
        self.rating_se = defaultdict(lambda: None)
        self.components = []
        self.membership = {}

        graph, self.games_count = self._build_graph_and_counts(games)
        if not games or not graph:
            return

        components = self._connected_components(graph)
        components.sort(key=len, reverse=True)

        membership = {}
        for cid, comp in enumerate(components, 1):
            for p in comp:
                membership[p] = cid

        for comp in components:
            comp_set = set(comp)
            comp_games = [g for g in games if g["p1"] in comp_set and g["p2"] in comp_set]
            comp_ratings, comp_se = self._solve_component_least_squares_with_se(comp, comp_games)
            for p in comp:
                self.ratings[p] = comp_ratings.get(p, 0.0)
                self.rating_se[p] = comp_se.get(p, 0.0)

        self.components = components
        self.membership = membership

    def rankings(self):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def predict_margin_points(self, p1, p2):
        return (self.ratings[p1] - self.ratings[p2]) * 30.0


# ==============================
# PERSISTENCE (save/load)
# ==============================

def load_games():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    return []


def save_games(games):
    with open(SAVE_FILE, "w") as f:
        json.dump(games, f, indent=2)


# ==============================
# DATA INTEGRITY (duplicate detection)
# ==============================

def find_duplicate_game_index(games, p1, p2, s1, s2):
    for i, g in enumerate(games):
        a, b = g["p1"], g["p2"]
        x, y = g["s1"], g["s2"]
        same_order = (a == p1 and b == p2 and x == s1 and y == s2)
        flipped_order = (a == p2 and b == p1 and x == s2 and y == s1)
        if same_order or flipped_order:
            return i
    return None


# ==============================
# DATA MANAGEMENT (add / undo / edit / delete)
# ==============================

def add_game(games):
    print("\nEnter a completed game (to 30):")
    p1 = input("Player 1 name: ").strip()
    p2 = input("Player 2 name: ").strip()
    s1 = int(input(f"{p1} score: ").strip())
    s2 = int(input(f"{p2} score: ").strip())

    dup_idx = find_duplicate_game_index(games, p1, p2, s1, s2)
    if dup_idx is not None:
        print(
            f"\nWARNING: This looks like a duplicate of game #{dup_idx + 1}: "
            f"{games[dup_idx]['p1']} {games[dup_idx]['s1']} - {games[dup_idx]['p2']} {games[dup_idx]['s2']}"
        )
        ans = input("Add anyway? (y/N): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Canceled.\n")
            return

    games.append({"p1": p1, "p2": p2, "s1": s1, "s2": s2})
    save_games(games)
    print("Game saved.\n")


def undo_last_add(games):
    if not games:
        print("No games to undo.\n")
        return
    last = games[-1]
    print("\nLast saved game is:")
    print(f"  {last['p1']} {last['s1']} - {last['p2']} {last['s2']}")
    ans = input("Undo (delete) this last game? (y/N): ").strip().lower()
    if ans not in ("y", "yes"):
        print("Canceled.\n")
        return
    removed = games.pop()
    save_games(games)
    print(
        f"Undone. Removed: {removed['p1']} {removed['s1']} - {removed['p2']} {removed['s2']}\n"
    )


def _games_for_player(games, player_name):
    out = []
    for i, g in enumerate(games, 1):
        if g["p1"] == player_name or g["p2"] == player_name:
            out.append((i, g))
    return out


def _print_game_line(i, g):
    return f"#{i}: {g['p1']} {g['s1']} - {g['p2']} {g['s2']}"


def edit_game(games):
    if not games:
        print("No games to edit.\n")
        return

    mode = input("\nEdit by (1) game # or (2) filter by player? Enter 1 or 2: ").strip()
    if mode not in ("1", "2"):
        print("Canceled.\n")
        return

    if mode == "1":
        pick = input("Enter game # to edit: ").strip()
        try:
            idx = int(pick)
        except ValueError:
            print("Invalid number.\n")
            return
        if idx < 1 or idx > len(games):
            print("Game # out of range.\n")
            return
        target_idx = idx
    else:
        name = input("Enter player name to filter: ").strip()
        if not name:
            return

        all_names = set()
        for g in games:
            all_names.add(g["p1"])
            all_names.add(g["p2"])

        resolved = _resolve_name_case_insensitive(name, all_names)
        if not resolved:
            print("Player not found in games (check spelling).\n")
            return

        matches = _games_for_player(games, resolved)
        if not matches:
            print("No games found for that player.\n")
            return

        print(f"\nGames involving {resolved}:")
        for i, g in matches:
            print("  " + _print_game_line(i, g))

        pick = input("\nEnter game # to edit (or press Enter to cancel): ").strip()
        if not pick:
            print("Canceled.\n")
            return
        try:
            target_idx = int(pick)
        except ValueError:
            print("Invalid number.\n")
            return
        if target_idx < 1 or target_idx > len(games):
            print("Game # out of range.\n")
            return

    g = games[target_idx - 1]
    print("\nEditing this game:")
    print("  " + _print_game_line(target_idx, g))
    print("\nPress Enter to keep an existing field.\n")

    new_p1 = input(f"Player 1 name [{g['p1']}]: ").strip() or g["p1"]
    new_p2 = input(f"Player 2 name [{g['p2']}]: ").strip() or g["p2"]

    s1_in = input(f"{new_p1} score [{g['s1']}]: ").strip()
    s2_in = input(f"{new_p2} score [{g['s2']}]: ").strip()
    try:
        new_s1 = int(s1_in) if s1_in else int(g["s1"])
        new_s2 = int(s2_in) if s2_in else int(g["s2"])
    except ValueError:
        print("Invalid score(s). Canceled.\n")
        return

    tmp_games = games[:target_idx - 1] + games[target_idx:]
    dup_idx = find_duplicate_game_index(tmp_games, new_p1, new_p2, new_s1, new_s2)
    if dup_idx is not None:
        print(
            f"\nWARNING: Edited game would duplicate an existing game: "
            f"{tmp_games[dup_idx]['p1']} {tmp_games[dup_idx]['s1']} - {tmp_games[dup_idx]['p2']} {tmp_games[dup_idx]['s2']}"
        )
        ans = input("Save anyway? (y/N): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Canceled.\n")
            return

    games[target_idx - 1] = {"p1": new_p1, "p2": new_p2, "s1": new_s1, "s2": new_s2}
    save_games(games)
    print("Saved edit.\n")


def searchable_delete_game(games):
    if not games:
        print("No games to delete.\n")
        return

    name = input("\nEnter a player name to filter games: ").strip()
    if not name:
        return

    all_names = set()
    for g in games:
        all_names.add(g["p1"])
        all_names.add(g["p2"])

    resolved = _resolve_name_case_insensitive(name, all_names)
    if not resolved:
        print("Player not found in games (check spelling).\n")
        return

    matches = _games_for_player(games, resolved)
    if not matches:
        print("No games found for that player.\n")
        return

    print(f"\nGames involving {resolved}:")
    for i, g in matches:
        print("  " + _print_game_line(i, g))

    pick = input("\nEnter game # to delete (or press Enter to cancel): ").strip()
    if not pick:
        print("Canceled.\n")
        return
    try:
        idx = int(pick)
    except ValueError:
        print("Invalid number.\n")
        return

    if idx < 1 or idx > len(games):
        print("Game # out of range.\n")
        return

    removed = games.pop(idx - 1)
    save_games(games)
    print(f"Deleted: {removed['p1']} {removed['s1']} - {removed['p2']} {removed['s2']}\n")


# ==============================
# OUTPUTS (leaderboard + player tools)
# ==============================

def _percentile(values_sorted_asc, p):
    n = len(values_sorted_asc)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values_sorted_asc[0])
    pos = (p / 100.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(values_sorted_asc[lo])
    frac = pos - lo
    return float(values_sorted_asc[lo] * (1 - frac) + values_sorted_asc[hi] * frac)


def _tier_label(i):
    return [
        "TIER 1 (>= 75th percentile rating)",
        "TIER 2 (50th–75th percentile rating)",
        "TIER 3 (25th–50th percentile rating)",
        "TIER 4 (< 25th percentile rating)",
    ][i]


def show_leaderboard_value_tiers(model, component_id=1):
    if not model.ratings:
        print("No ratings yet. Train the model first.")
        return
    if not model.components:
        print("Component info not available. Train the model again.")
        return
    if component_id < 1 or component_id > len(model.components):
        print(f"Invalid component. Choose 1 to {len(model.components)}.")
        return

    comp_players = set(model.components[component_id - 1])
    comp_rankings = [(name, rating) for (name, rating) in model.rankings() if name in comp_players]
    if not comp_rankings:
        print(f"No players in Component {component_id}.")
        return

    pts_values_desc = [r * 30.0 for _, r in comp_rankings]
    pts_values_asc = sorted(pts_values_desc)

    p75 = _percentile(pts_values_asc, 75)
    p50 = _percentile(pts_values_asc, 50)
    p25 = _percentile(pts_values_asc, 25)

    tiers = [[], [], [], []]
    for overall_rank, (name, rating) in enumerate(comp_rankings, 1):
        pts = rating * 30.0
        gp = model.games_count.get(name, 0)
        se_pts = (model.rating_se.get(name, 0.0) or 0.0) * 30.0
        cert = _certainty_percent(pts, se_pts)

        if pts >= p75:
            tiers[0].append((overall_rank, name, pts, gp, cert))
        elif pts >= p50:
            tiers[1].append((overall_rank, name, pts, gp, cert))
        elif pts >= p25:
            tiers[2].append((overall_rank, name, pts, gp, cert))
        else:
            tiers[3].append((overall_rank, name, pts, gp, cert))

    tier_sizes = [len(t) for t in tiers]

    print(f"\n=== LEADERBOARD (COMPONENT {component_id}) BY VALUE-BASED TIERS ===")
    print(f"Players in component: {len(comp_rankings)}")
    print(f"Cutoffs (pts vs avg): P75={p75:.2f}, P50={p50:.2f}, P25={p25:.2f}")
    print(f"Tier sizes: {tier_sizes[0]} / {tier_sizes[1]} / {tier_sizes[2]} / {tier_sizes[3]}")
    print("(Certainty = probability your rating is above/below avg (same sign as estimate), from regression SE.)\n")

    for i in range(4):
        if not tiers[i]:
            continue
        print(f"--- {_tier_label(i)} | {len(tiers[i])} players ---")
        for overall_rank, name, pts, gp, cert in tiers[i]:
            print(f"{overall_rank}. {name:<22}  {pts:+6.1f} pts   (GP: {gp:>3})   Cert: {cert:>5.1f}%")
        print()


def compute_strength_of_schedule_pts(model, games):
    opp_sum = defaultdict(float)
    opp_cnt = defaultdict(int)

    for g in games:
        p1, p2 = g["p1"], g["p2"]
        c1 = model.membership.get(p1)
        c2 = model.membership.get(p2)
        if c1 is None or c2 is None or c1 != c2:
            continue

        r1_pts = model.ratings[p1] * 30.0
        r2_pts = model.ratings[p2] * 30.0

        opp_sum[p1] += r2_pts
        opp_cnt[p1] += 1
        opp_sum[p2] += r1_pts
        opp_cnt[p2] += 1

    sos = {}
    for p in model.ratings.keys():
        sos[p] = (opp_sum[p] / opp_cnt[p]) if opp_cnt[p] > 0 else 0.0
    return sos


def player_card(model, games):
    if not model.ratings:
        print("No ratings yet. Train the model first.")
        return
    if not model.components:
        print("Component info not available. Train the model again.")
        return

    name_in = input("\nPlayer name: ").strip()
    if not name_in:
        return

    name = _resolve_name_case_insensitive(name_in, model.ratings.keys())
    if not name:
        print("Player not found (check spelling).")
        return

    cid = model.membership.get(name, None)
    rating_pts = model.ratings[name] * 30.0
    se_pts = (model.rating_se.get(name, 0.0) or 0.0) * 30.0
    cert = _certainty_percent(rating_pts, se_pts)
    gp = model.games_count.get(name, 0)

    comp_players = set(model.components[cid - 1]) if cid else set()
    comp_rankings = [(n, r) for (n, r) in model.rankings() if n in comp_players]
    comp_rank_index = next((i for i, (n, _) in enumerate(comp_rankings, 1) if n == name), None)

    opponents = set()
    player_games = []
    for i, g in enumerate(games, 1):
        p1, p2 = g["p1"], g["p2"]
        if p1 == name or p2 == name:
            opponents.add(p2 if p1 == name else p1)
            player_games.append((i, g))

    sos_map = compute_strength_of_schedule_pts(model, games)
    sos = sos_map.get(name, 0.0)

    print("\n=== PLAYER CARD ===")
    print(f"Name: {name}")
    print(f"Component: {cid} {'(Main league)' if cid == 1 else '(Separate league)'}")
    if comp_rank_index is not None:
        print(f"Rank in Component {cid}: {comp_rank_index}/{len(comp_rankings)}")
    print(f"Rating: {rating_pts:+.1f} pts vs component avg")
    print(f"Certainty: {cert:.1f}%  (SE: {se_pts:.2f} pts)")
    print(f"Games played: {gp}")
    print(f"Unique opponents: {len(opponents)}")
    print(f"SoS (avg opponent rating): {sos:+.1f} pts")

    last_n = 10
    if player_games:
        print(f"\nLast {min(last_n, len(player_games))} games (most recent last):")
        for idx, g in player_games[-last_n:]:
            p1, p2, s1, s2 = g["p1"], g["p2"], g["s1"], g["s2"]
            if p1 == name:
                margin = s1 - s2
                line = f"#{idx}: {p1} {s1} - {p2} {s2}  (margin {margin:+d})"
            else:
                margin = s2 - s1
                line = f"#{idx}: {p2} {s2} - {p1} {s1}  (margin {margin:+d})"
            print("  " + line)
    print()


def player_comparison(model, games):
    if not model.ratings:
        print("No ratings yet. Train the model first.")
        return
    if not model.components:
        print("Component info not available. Train the model again.")
        return

    a_in = input("\nPlayer A: ").strip()
    b_in = input("Player B: ").strip()
    if not a_in or not b_in:
        return

    a = _resolve_name_case_insensitive(a_in, model.ratings.keys())
    b = _resolve_name_case_insensitive(b_in, model.ratings.keys())
    if not a or not b:
        print("One or both players not found (check spelling).")
        return

    ca = model.membership.get(a)
    cb = model.membership.get(b)

    a_pts = model.ratings[a] * 30.0
    b_pts = model.ratings[b] * 30.0
    a_se = (model.rating_se.get(a, 0.0) or 0.0) * 30.0
    b_se = (model.rating_se.get(b, 0.0) or 0.0) * 30.0
    a_cert = _certainty_percent(a_pts, a_se)
    b_cert = _certainty_percent(b_pts, b_se)

    def comp_rank(name):
        cid = model.membership[name]
        comp_players = set(model.components[cid - 1])
        comp_rankings = [(n, r) for (n, r) in model.rankings() if n in comp_players]
        rk = next((i for i, (n, _) in enumerate(comp_rankings, 1) if n == name), None)
        return rk, len(comp_rankings), cid

    a_rk, a_n, _ = comp_rank(a)
    b_rk, b_n, _ = comp_rank(b)

    sos = compute_strength_of_schedule_pts(model, games)
    a_sos = sos.get(a, 0.0)
    b_sos = sos.get(b, 0.0)

    print("\n=== PLAYER COMPARISON ===")
    print(f"{a}: Comp {ca} | Rank {a_rk}/{a_n} | Rating {a_pts:+.1f} | Cert {a_cert:.1f}% | GP {model.games_count.get(a,0)} | SoS {a_sos:+.1f}")
    print(f"{b}: Comp {cb} | Rank {b_rk}/{b_n} | Rating {b_pts:+.1f} | Cert {b_cert:.1f}% | GP {model.games_count.get(b,0)} | SoS {b_sos:+.1f}")

    if ca != cb:
        print("\nNOTE: Different components. Rating gap and predicted margins are not meaningful across components.\n")
        return

    diff = a_pts - b_pts
    pred = model.predict_margin_points(a, b)
    print(f"\nRating gap (A - B): {diff:+.1f} pts")
    print(f"Predicted margin (A - B): {pred:+.1f} points\n")


def predict_matchup(model):
    print("\n=== MATCHUP PREDICTION ===")
    p1 = input("Player 1: ").strip()
    p2 = input("Player 2: ").strip()

    if len(model.ratings) == 0:
        print("Train the model first.\n")
        return

    p1_res = _resolve_name_case_insensitive(p1, model.ratings.keys())
    p2_res = _resolve_name_case_insensitive(p2, model.ratings.keys())
    if not p1_res or not p2_res:
        print("One or both players not found (check spelling).\n")
        return

    p1, p2 = p1_res, p2_res

    c1 = model.membership.get(p1)
    c2 = model.membership.get(p2)
    if c1 is None or c2 is None:
        print("Missing component info. Train again.\n")
        return

    if c1 != c2:
        print(f"\nWARNING: {p1} is in Component {c1} and {p2} is in Component {c2}.")
        print("Predicted margins across different components are NOT meaningful (no shared opponents).\n")
        return

    expected_margin = model.predict_margin_points(p1, p2)
    print(f"\nExpected point margin ({p1} - {p2}): {expected_margin:+.1f} points\n")


# ==============================
# DIAGNOSTICS / ANALYSIS
# ==============================

def upset_finder(model, games):
    if not model.ratings:
        print("No ratings yet. Train the model first.\n")
        return

    results = []
    for i, g in enumerate(games, 1):
        p1, p2, s1, s2 = g["p1"], g["p2"], g["s1"], g["s2"]
        c1 = model.membership.get(p1)
        c2 = model.membership.get(p2)
        if c1 is None or c2 is None or c1 != c2:
            continue
        actual = s1 - s2
        predicted = model.predict_margin_points(p1, p2)
        surprise = actual - predicted
        results.append({
            "idx": i, "p1": p1, "p2": p2, "s1": s1, "s2": s2,
            "actual": actual, "pred": predicted, "surprise": surprise, "component": c1
        })

    if not results:
        print("No analyzable games found.\n")
        return

    over = sorted(results, key=lambda x: x["surprise"], reverse=True)
    under = sorted(results, key=lambda x: x["surprise"])

    top_k_in = input("\nHow many to show per list? (default 10): ").strip()
    try:
        k = int(top_k_in) if top_k_in else 10
        if k < 1:
            k = 10
    except ValueError:
        k = 10

    print("\n=== UPSET FINDER ===")
    print("(surprise = actual margin - predicted margin; positive = Player1 did better than expected)\n")

    print(f"--- TOP {k} OVERPERFORMANCES / UPSETS ---")
    for r in over[:k]:
        print(
            f"#{r['idx']}: {r['p1']} {r['s1']} - {r['p2']} {r['s2']} | "
            f"actual {r['actual']:+.0f}, pred {r['pred']:+.1f}, surprise {r['surprise']:+.1f} | comp {r['component']}"
        )

    print(f"\n--- TOP {k} UNDERPERFORMANCES ---")
    for r in under[:k]:
        print(
            f"#{r['idx']}: {r['p1']} {r['s1']} - {r['p2']} {r['s2']} | "
            f"actual {r['actual']:+.0f}, pred {r['pred']:+.1f}, surprise {r['surprise']:+.1f} | comp {r['component']}"
        )
    print()


def build_components_and_membership(games):
    graph = defaultdict(set)
    for g in games:
        p1, p2 = g["p1"], g["p2"]
        graph[p1].add(p2)
        graph[p2].add(p1)

    visited = set()
    components = []
    for node in graph.keys():
        if node in visited:
            continue
        q = deque([node])
        visited.add(node)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in graph[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        components.append(comp)

    components.sort(key=len, reverse=True)

    membership = {}
    for i, comp in enumerate(components, 1):
        for p in comp:
            membership[p] = i

    return components, membership, graph


def network_analysis(games, sample_n=10):
    if not games:
        print("No games entered yet.\n")
        return

    components, membership, graph = build_components_and_membership(games)

    print("\n=== NETWORK ANALYSIS ===")
    print(f"Total players: {len(graph)}")
    print(f"Total games: {len(games)}")
    print(f"Connected components: {len(components)}")

    print("\nComponent sizes (largest to smallest) + sample names:")
    for i, comp in enumerate(components, 1):
        sample = sorted(comp, key=str.lower)[:sample_n]
        sample_str = ", ".join(sample)
        more = f" (+{len(comp) - len(sample)} more)" if len(comp) > len(sample) else ""
        print(f"  Component {i}: {len(comp)} players | sample: {sample_str}{more}")

    print("\nType a player name to see which component they’re in.")
    print("Press Enter to return to the main menu.\n")

    while True:
        name = input("Player lookup: ").strip()
        if name == "":
            break

        if name in membership:
            cid = membership[name]
            comp_size = len(components[cid - 1])
            opps = len(graph[name])
            print(f"  {name} -> Component {cid} (size {comp_size}), opponents played: {opps}\n")
            continue

        lower_map = {p.lower(): p for p in membership.keys()}
        if name.lower() in lower_map:
            real = lower_map[name.lower()]
            cid = membership[real]
            comp_size = len(components[cid - 1])
            opps = len(graph[real])
            print(f"  {real} -> Component {cid} (size {comp_size}), opponents played: {opps}\n")
            continue

        print("  Player not found (check spelling).\n")


def shuffle_verification(games):
    if not games:
        print("No games entered yet.\n")
        return

    seed = 42
    rng = random.Random(seed)

    games_a = [dict(g) for g in games]
    games_b = [dict(g) for g in games]
    rng.shuffle(games_b)

    m_a = YT1v1Ranker()
    m_b = YT1v1Ranker()
    m_a.train(games_a)
    m_b.train(games_b)

    players = sorted(set(list(m_a.ratings.keys()) + list(m_b.ratings.keys())))
    abs_point_diffs = [abs((m_b.ratings[p] - m_a.ratings[p]) * 30.0) for p in players]
    max_diff_pts = max(abs_point_diffs) if abs_point_diffs else 0.0
    avg_diff_pts = (sum(abs_point_diffs) / len(abs_point_diffs)) if abs_point_diffs else 0.0

    print("\n=== SHUFFLE VERIFICATION (LEAST SQUARES) ===")
    print(f"Games: {len(games)} | seed={seed}")
    print(f"Max absolute rating difference: {max_diff_pts:.6f} points")
    print(f"Avg absolute rating difference: {avg_diff_pts:.6f} points")
    print("\n(Note: least squares is order-independent; any nonzero diffs should be ~floating-point tiny.)\n")


# ==============================
# EXPORT
# ==============================

def export_rankings_to_csv(model):
    if not model.ratings:
        print("No ratings yet. Train the model first.\n")
        return
    if not model.components:
        print("Component info not available. Train the model again.\n")
        return

    default = input("Export Component 1 only? (Y/n): ").strip().lower()
    if default in ("", "y", "yes"):
        components_to_export = [1]
    else:
        choice = input(f"Which component to export? (1-{len(model.components)}): ").strip()
        try:
            cid = int(choice)
        except ValueError:
            print("Not a valid number.\n")
            return
        if cid < 1 or cid > len(model.components):
            print("Invalid component number.\n")
            return
        components_to_export = [cid]

    also_all = input("Export ALL components into separate files too? (y/N): ").strip().lower()
    if also_all in ("y", "yes"):
        components_to_export = list(range(1, len(model.components) + 1))

    for cid in components_to_export:
        comp_players = set(model.components[cid - 1])
        comp_rankings = [(name, rating) for (name, rating) in model.rankings() if name in comp_players]

        filename = f"rankings_component{cid}.csv"
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "player", "rating_points_vs_avg", "rating_se_pts",
                "certainty_pct", "games_played", "component_id"
            ])
            for rank, (name, rating) in enumerate(comp_rankings, 1):
                pts = rating * 30.0
                se_pts = (model.rating_se.get(name, 0.0) or 0.0) * 30.0
                cert = _certainty_percent(pts, se_pts)
                writer.writerow([
                    rank,
                    name,
                    round(pts, 6),
                    round(se_pts, 6),
                    round(cert, 3),
                    model.games_count.get(name, 0),
                    cid
                ])

        print(f"Exported: {filename}")


# ==============================
# MAIN LOOP (reordered menu)
# ==============================

def main():
    games = load_games()
    print(f"Loaded {len(games)} saved games from {SAVE_FILE}.")

    model = YT1v1Ranker()

    while True:
        # Logical flow:
        # Data -> Train -> View -> Analysis -> Export -> Quit
        print("\n1. Add a game (duplicate warning enabled)")
        print("2. Undo last add (delete most recent game)")
        print("3. Edit a game")
        print("4. Delete a game (search/filter)")
        print("5. Train/update rankings (least squares)")
        print("6. Leaderboard (value-based tiers; Component 1 by default)")
        print("7. Player Card")
        print("8. Player Comparison")
        print("9. Predict matchup")
        print("10. Upset Finder")
        print("11. Network Analysis")
        print("12. Shuffle verification (order bias test)")
        print("13. Export rankings to CSV")
        print("14. Quit")

        choice = input("Select option: ").strip()

        if choice == "1":
            add_game(games)
        elif choice == "2":
            undo_last_add(games)
        elif choice == "3":
            edit_game(games)
        elif choice == "4":
            searchable_delete_game(games)
        elif choice == "5":
            if len(games) == 0:
                print("No games entered yet.\n")
                continue
            model.train(games)
            print("Model trained (least squares).\n")
        elif choice == "6":
            show_leaderboard_value_tiers(model, component_id=1)
            if model.components and len(model.components) > 1:
                ans = input(
                    f"View another component leaderboard? (enter number 2-{len(model.components)} or press Enter): "
                ).strip()
                if ans:
                    try:
                        cid = int(ans)
                        show_leaderboard_value_tiers(model, component_id=cid)
                    except ValueError:
                        print("Not a valid number.\n")
        elif choice == "7":
            player_card(model, games)
        elif choice == "8":
            player_comparison(model, games)
        elif choice == "9":
            predict_matchup(model)
        elif choice == "10":
            upset_finder(model, games)
        elif choice == "11":
            network_analysis(games, sample_n=10)
        elif choice == "12":
            shuffle_verification(games)
        elif choice == "13":
            export_rankings_to_csv(model)
        elif choice == "14":
            break
        else:
            print("Invalid option.\n")


if __name__ == "__main__":
    main()
