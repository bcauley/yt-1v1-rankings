import json
import os
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd

# IMPORTANT: this is your core program file in the repo
import yt_1v1_power_rankings_final as core


# -----------------------------
# Streamlit helpers / config
# -----------------------------
st.set_page_config(page_title="YT 1v1 Power Rankings", layout="wide")

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _load_games():
    # Use the same loader you already wrote.
    # It reads from core.SAVE_FILE (games.json) in the app's working dir.
    return core.load_games()

def _save_games(games):
    core.save_games(games)

def _ensure_session_state():
    if "games" not in st.session_state:
        st.session_state.games = _load_games()
    if "model" not in st.session_state:
        st.session_state.model = core.YT1v1Ranker()
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "last_action" not in st.session_state:
        st.session_state.last_action = None
    if "selected_component_index" not in st.session_state:
        st.session_state.selected_component_index = 0  # Component 1 default (index 0)

def _train():
    games = st.session_state.games
    if not games:
        st.session_state.trained = False
        return
    st.session_state.model.train(games)
    st.session_state.trained = True

def _components_summary(model):
    # model.components is expected to be list[list[player]]
    comps = getattr(model, "components", []) or []
    sizes = [len(c) for c in comps]
    total_players = sum(sizes)
    return comps, sizes, total_players

def _leaderboard_df(model, component_index=0):
    comps, _, _ = _components_summary(model)
    if not comps:
        return pd.DataFrame()

    component_index = max(0, min(component_index, len(comps) - 1))
    players_in_comp = set(comps[component_index])

    # model.rankings() returns list[(name, rating_internal)]
    rows = []
    for name, r_internal in model.rankings():
        if name not in players_in_comp:
            continue

        rating_pts = r_internal * 30.0
        se_internal = (model.rating_se.get(name, None) if hasattr(model, "rating_se") else None)
        se_pts = (se_internal * 30.0) if (se_internal is not None) else None

        # certainty function in your core expects (r_pts, se_pts)
        cert = core._certainty_percent(rating_pts, se_pts)

        gp = model.games_count.get(name, 0) if hasattr(model, "games_count") else None

        rows.append({
            "Player": name,
            "Rating (pts)": rating_pts,
            "SE (pts)": se_pts,
            "Certainty %": cert,
            "Games": gp
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Rating (pts)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df

def _tiers_from_thresholds(df, thresholds_pts):
    """
    thresholds_pts: list of descending thresholds. Example [10, 5, 0, -5]
    Tier assignment:
      >= 10 -> Tier A
      >= 5  -> Tier B
      >= 0  -> Tier C
      >= -5 -> Tier D
      else  -> Tier E
    """
    if df.empty:
        return df

    labels = []
    tier_names = []
    # Build tier names A, B, C...
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(len(thresholds_pts) + 1):
        tier_names.append(f"Tier {alphabet[i]}")

    def which_tier(val):
        for i, th in enumerate(thresholds_pts):
            if val >= th:
                return tier_names[i]
        return tier_names[len(thresholds_pts)]

    df2 = df.copy()
    df2["Tier"] = df2["Rating (pts)"].apply(which_tier)
    return df2

def _download_df_as_csv(df, filename):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {filename}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

def _render_warning_about_persistence():
    st.info(
        "Note: Streamlit Community Cloud can reset the server filesystem on redeploy or restart. "
        "If you want long-term persistence beyond a single running instance, use a database "
        "(SQLite/Postgres) or store games in Google Sheets. For now, games.json works for simple use."
    )

# -----------------------------
# App
# -----------------------------
_ensure_session_state()

st.title("YT 1v1 Power Rankings (Streamlit)")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Leaderboard", "Add Game", "Manage Games", "Train / Diagnostics", "Export"],
        index=0
    )

    st.divider()
    st.caption("Data file")
    st.code(getattr(core, "SAVE_FILE", "games.json"), language="text")
    if st.button("Reload games.json", use_container_width=True):
        st.session_state.games = _load_games()
        st.session_state.trained = False
        st.session_state.last_action = ("reload", _now_str())
        st.success("Reloaded games from file.")

    st.divider()
    st.caption("Quick stats")
    st.write(f"Games loaded: **{len(st.session_state.games)}**")

# Optional persistence note
_render_warning_about_persistence()

# Auto-train if not trained but games exist (keeps UX smooth)
if (not st.session_state.trained) and st.session_state.games:
    _train()

# -----------------------------
# Leaderboard page
# -----------------------------
if page == "Leaderboard":
    st.subheader("Leaderboard")

    model = st.session_state.model
    comps, sizes, total_players = _components_summary(model)

    if not st.session_state.games:
        st.warning("No games loaded. Add games first.")
        st.stop()

    if not comps:
        st.warning("Model has no components yet. Try training.")
        st.stop()

    colA, colB, colC = st.columns([1.2, 1.2, 2])

    with colA:
        st.write("**Component view**")
        # Default: component 1 (index 0)
        options = [f"Component {i+1} ({sizes[i]} players)" for i in range(len(sizes))]
        comp_idx = st.selectbox(
            "Select component",
            list(range(len(options))),
            format_func=lambda i: options[i],
            index=st.session_state.selected_component_index,
        )
        st.session_state.selected_component_index = comp_idx

    with colB:
        st.write("**Tier thresholds (points)**")
        st.caption("Example: 10, 5, 0, -5")
        thresholds_str = st.text_input("Thresholds (comma-separated)", value="10, 5, 0, -5")
        thresholds = []
        for part in thresholds_str.split(","):
            part = part.strip()
            if part:
                v = _safe_int(part, None)
                if v is not None:
                    thresholds.append(v)
        thresholds = sorted(list(set(thresholds)), reverse=True)

    with colC:
        st.write("**Display options**")
        show_se = st.checkbox("Show SE (pts)", value=True)
        show_cert = st.checkbox("Show Certainty %", value=True)
        show_games = st.checkbox("Show Games Played", value=True)
        decimals = st.slider("Rating decimals", 0, 2, 1)

    df = _leaderboard_df(model, component_index=comp_idx)
    if df.empty:
        st.warning("No players found in this component.")
        st.stop()

    df = _tiers_from_thresholds(df, thresholds)

    # Format
    df_show = df.copy()
    df_show["Rating (pts)"] = df_show["Rating (pts)"].map(lambda x: round(float(x), decimals))
    if "SE (pts)" in df_show and show_se:
        df_show["SE (pts)"] = df_show["SE (pts)"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    if "Certainty %" in df_show and show_cert:
        df_show["Certainty %"] = df_show["Certainty %"].map(lambda x: round(float(x), 1))
    if "Games" in df_show and show_games:
        df_show["Games"] = df_show["Games"].fillna(0).astype(int)

    cols = ["Rank", "Player", "Tier", "Rating (pts)"]
    if show_se: cols.append("SE (pts)")
    if show_cert: cols.append("Certainty %")
    if show_games: cols.append("Games")

    st.dataframe(df_show[cols], use_container_width=True, hide_index=True)

    st.caption(f"Total players across all components: {total_players} | Components: {len(comps)}")

# -----------------------------
# Add Game page
# -----------------------------
elif page == "Add Game":
    st.subheader("Add a Game")

    with st.form("add_game_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        with c1:
            p1 = st.text_input("Player 1 name")
        with c2:
            p2 = st.text_input("Player 2 name")
        with c3:
            s1 = st.number_input("P1 score", min_value=0, max_value=60, value=30, step=1)
        with c4:
            s2 = st.number_input("P2 score", min_value=0, max_value=60, value=20, step=1)

        submitted = st.form_submit_button("Add Game", use_container_width=True)

    if submitted:
        p1 = (p1 or "").strip()
        p2 = (p2 or "").strip()
        s1 = int(s1)
        s2 = int(s2)

        if not p1 or not p2:
            st.error("Both player names are required.")
            st.stop()
        if p1.lower() == p2.lower():
            st.error("Player 1 and Player 2 must be different.")
            st.stop()

        # Optional: warn about non-30 totals (you asked about this earlier)
        if (s1 != 30 and s2 != 30) and (max(s1, s2) != 30):
            st.warning("Heads up: Your format is games to 30. This score doesn't show either player reaching 30.")

        # Duplicate detection using your core helper (if present)
        dup_idx = None
        if hasattr(core, "find_duplicate_game_index"):
            dup_idx = core.find_duplicate_game_index(st.session_state.games, p1, p2, s1, s2)

        if dup_idx is not None:
            st.warning(f"Possible duplicate of existing game #{dup_idx+1}. Still added.")

        st.session_state.games.append({"p1": p1, "p2": p2, "s1": s1, "s2": s2})
        _save_games(st.session_state.games)

        st.session_state.last_action = ("add_game", _now_str())
        st.success(f"Saved: {p1} {s1} - {p2} {s2}")

        # retrain immediately for convenience
        _train()

# -----------------------------
# Manage Games page
# -----------------------------
elif page == "Manage Games":
    st.subheader("Manage Games (delete / undo)")

    games = st.session_state.games
    if not games:
        st.info("No games to manage yet.")
        st.stop()

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Undo last game", use_container_width=True):
            last = games[-1]
            games.pop()
            _save_games(games)
            st.session_state.last_action = ("undo_last", _now_str())
            st.success(f"Removed last: {last['p1']} {last['s1']} - {last['p2']} {last['s2']}")
            _train()

    with c2:
        st.write("")

    st.divider()

    st.write("### Delete a specific game")

    # Show games table
    df_games = pd.DataFrame(games)
    df_games.insert(0, "#", range(1, len(df_games) + 1))
    st.dataframe(df_games, use_container_width=True, hide_index=True)

    delete_idx = st.number_input("Game # to delete", min_value=1, max_value=len(games), value=1, step=1)
    if st.button("Delete selected game", type="primary", use_container_width=True):
        i = int(delete_idx) - 1
        g = games.pop(i)
        _save_games(games)
        st.session_state.last_action = ("delete_game", _now_str())
        st.success(f"Deleted: {g['p1']} {g['s1']} - {g['p2']} {g['s2']}")
        _train()

# -----------------------------
# Train / Diagnostics page
# -----------------------------
elif page == "Train / Diagnostics":
    st.subheader("Train / Diagnostics")

    if st.button("Train / Refresh now", use_container_width=True):
        _train()
        st.session_state.last_action = ("train", _now_str())
        st.success("Training complete.")

    model = st.session_state.model
    games = st.session_state.games

    if not games:
        st.warning("No games loaded.")
        st.stop()

    comps, sizes, total_players = _components_summary(model)
    st.write(f"**Total players:** {total_players}")
    st.write(f"**Components:** {len(comps)}")
    if sizes:
        st.write("**Component sizes:** " + ", ".join([f"{i+1}:{sizes[i]}" for i in range(len(sizes))]))

    # Structural sanity: bridges/isolates/etc if your core exposes checks
    st.divider()
    st.write("### Structural check")
    st.caption("This uses your model's graph structure if available.")

    if hasattr(core, "analyze_components"):
        # If your core has a function; if not, just show components summary
        info = core.analyze_components(st.session_state.games)
        st.write(info)
    else:
        st.info("Core doesn't expose `analyze_components()`. Showing component summary only.")

    st.divider()
    st.write("### Recent action")
    if st.session_state.last_action:
        st.write(st.session_state.last_action)
    else:
        st.write("(none)")

# -----------------------------
# Export page
# -----------------------------
elif page == "Export":
    st.subheader("Export")

    if not st.session_state.trained:
        st.warning("Train first (go to Train / Diagnostics).")
        st.stop()

    model = st.session_state.model
    comps, sizes, _ = _components_summary(model)

    comp_idx = st.selectbox(
        "Export which component?",
        list(range(len(sizes))),
        format_func=lambda i: f"Component {i+1} ({sizes[i]} players)",
        index=0
    )

    df = _leaderboard_df(model, component_index=comp_idx)
    if df.empty:
        st.warning("Nothing to export for this component.")
        st.stop()

    df = df.sort_values("Rating (pts)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    st.write("### CSV download")
    _download_df_as_csv(df, filename=f"rankings_component_{comp_idx+1}.csv")

    st.write("### Raw data download")
    st.download_button(
        label="Download games.json",
        data=json.dumps(st.session_state.games, indent=2).encode("utf-8"),
        file_name="games.json",
        mime="application/json",
        use_container_width=True
    )

