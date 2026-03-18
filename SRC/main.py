import os
from typing import List, Tuple

import nflreadpy as nfl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def ensure_output_dirs() -> None:
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)


def load_raw_games(seasons: List[int]) -> pd.DataFrame:
    all_games: List[pd.DataFrame] = []
    for year in seasons:
        print(f"Downloading NFL schedule for season {year}...")
        sched = nfl.load_schedules(seasons=[year])
        df = sched.to_pandas()

        # Filter out games without final scores (future or in-progress games)
        df = df.dropna(subset=["home_score", "away_score"])

        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["point_diff"] = df["home_score"] - df["away_score"]
        all_games.append(df)

    games = pd.concat(all_games, ignore_index=True)
    print(f"Loaded {len(games)} games across {len(seasons)} seasons.")
    return games


def engineer_team_features(games: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """
    Create per-team rolling statistics to use as pre-game features.
    """
    feature_rows = []

    # Long format for both home and away to compute rolling stats
    home_stats = games[
        ["season", "week", "home_team", "home_score", "away_score", "home_win", "point_diff"]
    ].rename(
        columns={
            "home_team": "team",
            "home_score": "points_for",
            "away_score": "points_against",
            "home_win": "win",
        }
    )
    away_stats = games[
        ["season", "week", "away_team", "away_score", "home_score", "home_win", "point_diff"]
    ].rename(
        columns={
            "away_team": "team",
            "away_score": "points_for",
            "home_score": "points_against",
            "home_win": "home_win_flag",
        }
    )
    # For away team, win is the opposite of home_win
    away_stats["win"] = 1 - away_stats["home_win_flag"]
    away_stats = away_stats.drop(columns=["home_win_flag"])

    team_games = pd.concat([home_stats, away_stats], ignore_index=True)
    team_games = team_games.sort_values(["team", "season", "week"])

    rolling_features = []
    for team, group in team_games.groupby("team"):
        group = group.sort_values(["season", "week"])
        group["games_played"] = np.arange(len(group))
        group["rolling_points_for"] = (
            group["points_for"].shift().rolling(rolling_window, min_periods=1).mean()
        )
        group["rolling_points_against"] = (
            group["points_against"].shift().rolling(rolling_window, min_periods=1).mean()
        )
        group["rolling_win_rate"] = group["win"].shift().rolling(rolling_window, min_periods=1).mean()
        rolling_features.append(group)

    features_long = pd.concat(rolling_features, ignore_index=True)

    # Merge back to get per-game home/away feature set
    merged = games.copy()
    merged = merged.merge(
        features_long.add_prefix("home_"),
        left_on=["season", "week", "home_team"],
        right_on=["home_season", "home_week", "home_team"],
        how="left",
    )
    merged = merged.merge(
        features_long.add_prefix("away_"),
        left_on=["season", "week", "away_team"],
        right_on=["away_season", "away_week", "away_team"],
        how="left",
    )

    # Select and clean feature columns
    all_feature_cols = [
        "home_rolling_points_for",
        "home_rolling_points_against",
        "home_rolling_win_rate",
        "away_rolling_points_for",
        "away_rolling_points_against",
        "away_rolling_win_rate",
        "point_diff",
    ]

    available_features = [c for c in all_feature_cols if c in merged.columns]
    missing_features = [c for c in all_feature_cols if c not in merged.columns]

    if missing_features:
        print("Warning: the following engineered features were not found and will be skipped:")
        for col in missing_features:
            print(f"  - {col}")

    # Ensure target column exists
    if "home_win" not in merged.columns and {"home_score", "away_score"}.issubset(merged.columns):
        merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype(int)

    base_cols = ["season", "week", "home_team", "away_team", "home_win"]
    base_cols_available = [c for c in base_cols if c in merged.columns]
    missing_base = [c for c in base_cols if c not in merged.columns]
    if missing_base:
        print("Warning: the following base columns were not found and will be skipped:")
        for col in missing_base:
            print(f"  - {col}")

    model_df = merged[base_cols_available + available_features].dropna()
    print(f"Engineered feature dataset shape: {model_df.shape}")
    return model_df


def plot_feature_correlations(df: pd.DataFrame, feature_cols: List[str]) -> None:
    corr = df[feature_cols + ["home_win"]].corr()
    plt.figure(figsize=(10, 8))
    heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    fig_path = "../figures/feature_correlations.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved feature correlation heatmap to {fig_path}")


def build_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        "home_rolling_points_for",
        "home_rolling_points_against",
        "home_rolling_win_rate",
        "away_rolling_points_for",
        "away_rolling_points_against",
        "away_rolling_win_rate",
        "point_diff",
    ]
    X = df[feature_cols].copy()
    y = df["home_win"].astype(int)

    plot_feature_correlations(df, feature_cols)

    return X, y


def evaluate_models_kfold(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> None:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {"LogisticRegression": log_reg, "RandomForest": rf}

    for name, model in models.items():
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), start=1):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            accuracies.append(accuracy_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred))
            recalls.append(recall_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred))

            print(f"{name} - Fold {fold} classification report:")
            print(classification_report(y_val, y_pred))

        print(f"\n{name} - {n_splits}-fold cross-validation metrics:")
        print(f"Accuracy: {np.mean(accuracies):.3f}")
        print(f"Precision: {np.mean(precisions):.3f}")
        print(f"Recall: {np.mean(recalls):.3f}")
        print(f"F1-score: {np.mean(f1s):.3f}\n")

        # Confusion matrix on full dataset (for high-level view)
        model.fit(X_scaled, y)
        y_full_pred = model.predict(X_scaled)
        cm = confusion_matrix(y, y_full_pred)
        print(f"{name} - Confusion matrix on full dataset:")
        print(cm)


def main() -> None:
    print("Current working directory:", os.getcwd())
    ensure_output_dirs()

    # Example: pull last 10+ seasons
    seasons = list(range(2014, 2025))
    raw_games = load_raw_games(seasons)

    # Save raw subset for inspection
    raw_cols = [
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "point_diff",
    ]
    raw_csv_path = "../data/nfl_games_raw.csv"
    raw_games[raw_cols].to_csv(raw_csv_path, index=False)
    print(f"Raw games CSV saved to {raw_csv_path}")

    features_df = engineer_team_features(raw_games)
    features_csv_path = "../data/nfl_games_features.csv"
    features_df.to_csv(features_csv_path, index=False)
    print(f"Feature dataset CSV saved to {features_csv_path}")

    X, y = build_features_and_labels(features_df)
    evaluate_models_kfold(X, y, n_splits=5)


if __name__ == "__main__":
    main()