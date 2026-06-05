# NFL Outcome Prediction & Super Bowl Simulator

A Python project for exploring NFL game data, building engineered features, training simple prediction models, and simulating Super Bowl win probabilities with a Monte Carlo playoff simulator.

## Project Overview

This repo includes:

- `app.py` — a Streamlit dashboard for visualizing NFL games, features, team strength ratings, and simulated Super Bowl probabilities.
- `src/main.py` — data preparation, feature engineering, and model evaluation logic.
- `src/simulate_playoffs.py` — team strength rating calculation and simplified playoff simulation.
- `data/nfl_games.csv` — raw NFL game results used to generate feature datasets.

## Getting Started

### Requirements

Install the required Python packages in your environment:

```bash
python -m pip install pandas numpy streamlit matplotlib seaborn scikit-learn nflreadpy
```

If you prefer, create a virtual environment first:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy streamlit matplotlib seaborn scikit-learn nflreadpy
```

### Prepare the feature dataset

The Streamlit app expects a feature dataset at `data/nfl_games_features.csv`.

To generate that dataset from the raw game CSV, run:

```bash
python src/main.py
```

This script loads raw NFL schedule data through `nflreadpy`, engineers rolling team features, computes labels, and saves the processed dataset.

## Running the App

Start the Streamlit dashboard from the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Running the Playoff Simulator

The simplified Super Bowl simulator can be executed directly:

```bash
python src/simulate_playoffs.py
```

This script:

- loads team strength ratings from `data/nfl_games_features.csv`
- builds a placeholder 2025 playoff field
- simulates a bracket using relative team strengths
- prints estimated Super Bowl win probabilities

## What Each File Does

- `app.py`
  - loads preprocessed feature data
  - displays game information, feature distributions, and team strength ratings
  - includes a simulated Super Bowl probability tab

- `src/main.py`
  - downloads or reads NFL schedule data
  - computes per-team rolling stats and game features
  - plots feature correlations to `figures/feature_correlations.png`

- `src/simulate_playoffs.py`
  - computes normalized team strength ratings from historical wins
  - simulates a simplified playoff bracket using Monte Carlo

## Notes

- If `app.py` reports that `data/nfl_games_features.csv` is missing, make sure you run `python src/main.py` first.
- The playoff bracket is currently an example placeholder and should be updated with real playoff seedings when available.

## Project Structure

```text
NFL Project/
├── app.py
├── data/
│   └── nfl_games.csv
├── src/
│   ├── main.py
│   └── simulate_playoffs.py
└── README.md
```

Enjoy exploring NFL outcomes and simulating the road to the Super Bowl!"}