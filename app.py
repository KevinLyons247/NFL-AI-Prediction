import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from Src.simulate_playoffs import (
    example_2025_playoff_field,
    load_team_strengths,
    matchup_win_prob,
    simulate_playoffs,
)


# Align with paths used in Src/main.py ("../data", "../figures")
DATA_PATH = "../data/nfl_games_features.csv"
FIG_PATH = "../figures/feature_correlations.png"


@st.cache_data
def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(
            f"Feature dataset not found at `{path}`. "
            "From the project root, run `python Src/main.py` first."
        )
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def cached_team_strengths(path: str):
    return load_team_strengths(path)


def compute_team_strengths_from_df(df: pd.DataFrame) -> dict:
    """
    Compute normalized win-rate strengths from a features dataframe.
    Expects columns: home_team, away_team, home_win
    """
    home = df[["home_team", "home_win"]].rename(
        columns={"home_team": "team", "home_win": "result"}
    )
    away = df[["away_team", "home_win"]].rename(
        columns={"away_team": "team", "home_win": "home_win_flag"}
    )
    away["result"] = 1 - away["home_win_flag"]
    away = away[["team", "result"]]

    long_df = pd.concat([home, away], ignore_index=True)
    strengths = long_df.groupby("team")["result"].mean().to_dict()
    mean_strength = float(np.mean(list(strengths.values()))) if strengths else 1.0
    return {t: (s / mean_strength) for t, s in strengths.items()}


def main() -> None:
    st.set_page_config(
        page_title="NFL Outcome Prediction & Super Bowl Simulator",
        page_icon="🏈",
        layout="wide",
    )

    st.title("NFL Outcome Prediction & Super Bowl Simulator")
    st.markdown(
        "Explore historical NFL games, model features, and simulated Super Bowl win "
        "probabilities based on team strength ratings."
    )

    df = load_features(DATA_PATH)
    if df.empty:
        return

    st.sidebar.header("Filters")
    scope = st.sidebar.radio("Scope", ["All seasons", "One season"], index=0)

    if scope == "One season":
        selected_season = st.sidebar.selectbox(
            "Season",
            sorted(df["season"].unique()),
            index=len(df["season"].unique()) - 1,
        )
        view_df = df[df["season"] == selected_season].copy()
    else:
        selected_season = None
        view_df = df.copy()

    tab_overview, tab_games, tab_features, tab_strengths, tab_superbowl, tab_season2026 = st.tabs(
        [
            "Overview",
            "Games",
            "Features",
            "Team Strengths",
            "Super Bowl Simulator",
            "2026 Season Simulator",
        ]
    )

    with tab_overview:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total games in dataset", len(df))
        with col2:
            label = "Games in scope" if scope == "All seasons" else "Games in selected season"
            st.metric(label, len(view_df))
        with col3:
            # Count unique teams based on both home and away in scope
            teams_in_season = pd.unique(
                pd.concat([view_df["home_team"], view_df["away_team"]], ignore_index=True)
            )
            label = "Unique teams (scope)" if scope == "All seasons" else "Unique teams (selected season)"
            st.metric(label, len(teams_in_season))

        st.markdown("#### What this app shows")
        st.markdown(
            "- **Games**: Browse historical matchups and outcomes.\n"
            "- **Features**: Inspect engineered model inputs and their distributions.\n"
            "- **Team strengths**: View normalized win-rate based ratings.\n"
            "- **Super Bowl simulator**: Run Monte Carlo simulations of the playoffs."
        )

        if os.path.exists(FIG_PATH):
            st.markdown("#### Feature correlation heatmap")
            st.image(FIG_PATH, caption="Feature correlations", use_column_width=True)
        else:
            st.info(
                f"No correlation figure found at `{FIG_PATH}`. "
                "Run `python Src/main.py` to generate it."
            )

    with tab_games:
        title = "Games (all seasons)" if scope == "All seasons" else f"Games for Season {selected_season}"
        st.subheader(title)
        st.dataframe(
            view_df[
                [
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "home_win",
                ]
            ],
            use_container_width=True,
            height=450,
        )

    with tab_features:
        st.subheader("Feature Distributions")
        numeric_cols = [
            c
            for c in view_df.columns
            if view_df[c].dtype != "object" and c not in {"season", "week"}
        ]
        if not numeric_cols:
            st.info("No numeric feature columns found for this season.")
        else:
            feature_to_plot = st.selectbox("Select feature", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(view_df[feature_to_plot].dropna(), bins=30, alpha=0.7)
            ax.set_xlabel(feature_to_plot)
            ax.set_ylabel("Count")
            scope_label = "all seasons" if scope == "All seasons" else str(selected_season)
            ax.set_title(f"Distribution of {feature_to_plot} ({scope_label})")
            st.pyplot(fig)

    with tab_strengths:
        st.subheader("Team Strength Ratings (normalized win rates)")
        strengths = compute_team_strengths_from_df(view_df)
        strength_items = sorted(strengths.items(), key=lambda x: -x[1])
        strength_df = pd.DataFrame(strength_items, columns=["team", "rating"])

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.bar_chart(
                strength_df.set_index("team")["rating"],
                height=450,
            )
        with col_right:
            st.dataframe(strength_df, use_container_width=True, height=450)

    with tab_superbowl:
        st.subheader("Super Bowl Win Probability Simulation")
        st.markdown(
            "Uses the team strength ratings and a simplified playoff bracket to "
            "estimate Super Bowl win probabilities via Monte Carlo simulation."
        )

        strengths = compute_team_strengths_from_df(view_df)
        n_sims = st.slider("Number of simulations", 1000, 20000, 5000, step=1000)

        if st.button("Run Super Bowl Simulation"):
            teams = example_2025_playoff_field(strengths)
            if not teams:
                st.error("No playoff teams found in strength ratings.")
                return

            probs = simulate_playoffs(teams, strengths, n_sims=n_sims, seed=42)
            prob_items = sorted(probs.items(), key=lambda x: -x[1])
            prob_df = pd.DataFrame(
                [(team, p * 100) for team, p in prob_items],
                columns=["team", "super_bowl_win_prob_%"],
            )

            st.bar_chart(
                prob_df.set_index("team")["super_bowl_win_prob_%"],
                height=450,
            )
            st.dataframe(prob_df, use_container_width=True)

    with tab_season2026:
        st.subheader("2026 Regular Season Game Predictions")
        st.markdown(
            "This section predicts every regular-season game for the 2026 season, "
            "given a schedule file, and summarizes expected wins per team."
        )

        schedule_path = "../data/season_2026_schedule.csv"
        if not os.path.exists(schedule_path):
            st.info(
                f"No 2026 schedule file found at `{schedule_path}`.\n\n"
                "Create a CSV with columns `week,home_team,away_team` for the 2026 "
                "regular season schedule and save it there to enable this view."
            )
        else:
            sched_df = pd.read_csv(schedule_path)
            required_cols = {"week", "home_team", "away_team"}
            if not required_cols.issubset(sched_df.columns):
                st.error(
                    "Schedule file must contain columns: week, home_team, away_team."
                )
            else:
                strengths = cached_team_strengths(DATA_PATH)

                def predict_row(row):
                    p_home = matchup_win_prob(row["home_team"], row["away_team"], strengths)
                    return pd.Series(
                        {
                            "home_win_prob": p_home,
                            "away_win_prob": 1.0 - p_home,
                            "predicted_winner": row["home_team"]
                            if p_home >= 0.5
                            else row["away_team"],
                        }
                    )

                preds = sched_df.apply(predict_row, axis=1)
                full_sched = pd.concat([sched_df, preds], axis=1)

                st.markdown("#### Game-by-game predictions")
                st.dataframe(
                    full_sched.sort_values(["week", "home_team"]),
                    use_container_width=True,
                    height=450,
                )

                # Expected wins per team across the full season
                team_expected_wins = {}
                for _, row in full_sched.iterrows():
                    home = row["home_team"]
                    away = row["away_team"]
                    p_home = row["home_win_prob"]
                    p_away = 1.0 - p_home

                    team_expected_wins[home] = team_expected_wins.get(home, 0.0) + p_home
                    team_expected_wins[away] = team_expected_wins.get(away, 0.0) + p_away

                standings = (
                    pd.DataFrame(
                        [(t, w) for t, w in team_expected_wins.items()],
                        columns=["team", "expected_wins"],
                    )
                    .sort_values("expected_wins", ascending=False)
                    .reset_index(drop=True)
                )

                st.markdown("#### Expected 2026 standings (by expected wins)")
                st.bar_chart(
                    standings.set_index("team")["expected_wins"],
                    height=450,
                )
                st.dataframe(standings, use_container_width=True)


if __name__ == "__main__":
    main()

