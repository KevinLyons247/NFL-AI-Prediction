import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PlayoffTeam:
    name: str
    conference: str  # "AFC" or "NFC"
    seed: int


def load_team_strengths(path: str = "../data/nfl_games_features.csv") -> Dict[str, float]:
    """
    Build a simple team strength rating from historical data.

    Rating = overall win rate across all games (home + away) in the dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature dataset not found at {path}. Run main.py first to generate features."
        )

    df = pd.read_csv(path)

    # Build long format with a result column for each team in each game.
    home = df[["home_team", "home_win"]].rename(
        columns={"home_team": "team", "home_win": "result"}
    )
    away = df[["away_team", "home_win"]].rename(
        columns={"away_team": "team", "home_win": "home_win_flag"}
    )
    away["result"] = 1 - away["home_win_flag"]
    away = away[["team", "result"]]

    long_df = pd.concat([home, away], ignore_index=True)

    team_strengths: Dict[str, float] = (
        long_df.groupby("team")["result"].mean().to_dict()
    )

    # Normalize so that an "average" team is ~1.0
    mean_strength = np.mean(list(team_strengths.values()))
    normalized = {t: s / mean_strength for t, s in team_strengths.items()}

    print("Computed team strength ratings (normalized win rates):")
    for team, rating in sorted(normalized.items(), key=lambda x: -x[1])[:10]:
        print(f"{team}: {rating:.3f}")

    return normalized


def matchup_win_prob(team_a: str, team_b: str, strengths: Dict[str, float]) -> float:
    """
    Return the probability that team_a beats team_b, based on relative strength.
    """
    ra = strengths.get(team_a, 1.0)
    rb = strengths.get(team_b, 1.0)
    # Simple Bradley-Terry style probability
    return ra / (ra + rb)


def simulate_series(
    teams: List[PlayoffTeam],
    strengths: Dict[str, float],
    rng: np.random.Generator,
) -> str:
    """
    Simulate an NFL-style playoff bracket for one season and return the Super Bowl winner.

    NOTE: This function uses a simplified bracket:
    - Fixed bracket based on seeds (1–7 per conference).
    - Single-elimination, neutral field (no explicit home-field advantage).
    """

    def run_round(bracket: List[PlayoffTeam]) -> List[PlayoffTeam]:
        winners: List[PlayoffTeam] = []
        # Pair highest seed with lowest, etc.
        sorted_teams = sorted(bracket, key=lambda t: t.seed)
        while len(sorted_teams) >= 2:
            high = sorted_teams[0]
            low = sorted_teams[-1]
            sorted_teams = sorted_teams[1:-1]
            p = matchup_win_prob(high.name, low.name, strengths)
            if rng.random() < p:
                winners.append(high)
            else:
                winners.append(low)
        return winners

    # Split by conference
    afc = [t for t in teams if t.conference == "AFC"]
    nfc = [t for t in teams if t.conference == "NFC"]

    # AFC bracket
    afc_round = afc
    while len(afc_round) > 1:
        afc_round = run_round(afc_round)
    afc_champ = afc_round[0]

    # NFC bracket
    nfc_round = nfc
    while len(nfc_round) > 1:
        nfc_round = run_round(nfc_round)
    nfc_champ = nfc_round[0]

    # Super Bowl
    p_afc = matchup_win_prob(afc_champ.name, nfc_champ.name, strengths)
    winner = afc_champ if rng.random() < p_afc else nfc_champ
    return winner.name


def simulate_playoffs(
    teams: List[PlayoffTeam],
    strengths: Dict[str, float],
    n_sims: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    winners = []
    for _ in range(n_sims):
        winners.append(simulate_series(teams, strengths, rng))

    counts = Counter(winners)
    probs = {team: counts[team] / n_sims for team in counts}
    return probs


def example_2025_playoff_field(strengths: Dict[str, float]) -> List[PlayoffTeam]:
    """
    Placeholder example playoff field for the upcoming season.

    Replace this with the actual seeds once the real bracket is known.
    """
    # Example names; adjust to real seeds/teams later.
    afc_teams = [
        "KC",  # Chiefs
        "BAL",  # Ravens
        "BUF",  # Bills
        "CIN",  # Bengals
        "MIA",  # Dolphins
        "NYJ",  # Jets
        "LAC",  # Chargers
    ]
    nfc_teams = [
        "SF",  # 49ers
        "PHI",  # Eagles
        "DAL",  # Cowboys
        "DET",  # Lions
        "GB",  # Packers
        "LA",  # Rams
        "SEA",  # Seahawks
    ]

    afc = [
        PlayoffTeam(name=team, conference="AFC", seed=i + 1)
        for i, team in enumerate(afc_teams)
        if team in strengths
    ]
    nfc = [
        PlayoffTeam(name=team, conference="NFC", seed=i + 1)
        for i, team in enumerate(nfc_teams)
        if team in strengths
    ]

    teams = afc + nfc
    if len(teams) < 14:
        print(
            "Warning: some example playoff teams were not found in training data "
            "and will be skipped."
        )
    return teams


def main() -> None:
    strengths = load_team_strengths("../data/nfl_games_features.csv")
    teams = example_2025_playoff_field(strengths)

    print("\nSimulating playoff bracket and Super Bowl winner probabilities...")
    probs = simulate_playoffs(teams, strengths, n_sims=10_000, seed=42)

    print("\nEstimated Super Bowl win probabilities (example upcoming season):")
    for team, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"{team}: {p*100:.1f}%")


if __name__ == "__main__":
    main()

