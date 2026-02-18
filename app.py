import streamlit as st
import yt_1v1_power_rankings_final as core

st.title("YT 1v1 Power Rankings")

games = core.load_games()
model = core.YT1v1Ranker()

if st.button("Train / Refresh Ratings"):
    model.train(games)

st.write(f"Total Games: {len(games)}")

if model.ratings:
    st.subheader("Leaderboard")
    for name, r in model.rankings():
        st.write(name, round(r*30, 1))
