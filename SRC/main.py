# main.py
import nflreadpy as nfl
import pandas as pd
import os

print("Current working directory:", os.getcwd())
# Make sure the data folder exists
os.makedirs("../data", exist_ok=True)

# Seasons to pull
seasons = [2022, 2023, 2024]

all_games = []

for year in seasons:
    print(f"Downloading NFL schedule for season {year}...")
    sched = nfl.load_schedules(seasons=[year])
    df = sched.to_pandas()

    # Add home win flag and point difference
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['point_diff'] = df['home_score'] - df['away_score']

    all_games.append(df)

# Combine all seasons into one DataFrame
df_games = pd.concat(all_games, ignore_index=True)

# Keep only relevant columns
df_games = df_games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_win', 'point_diff']]

# Save to CSV
csv_path = "data/nfl_games.csv"
df_games.to_csv(csv_path, index=False)
print(f"CSV saved to {csv_path}")
print(df_games.head())
