#!/usr/bin/env python3
"""
Train a RandomForestRegressor for bike rental prediction.
Generates synthetic training data based on realistic bikeshare patterns.
Saves model to artifacts/bikeshare_rf_model.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

np.random.seed(42)

# Feature columns matching the one-hot encoded schema
FEATURE_COLUMNS = [
    "temp", "humidity",
    "season_2", "season_3", "season_4",
    "month_2", "month_3", "month_4", "month_5", "month_6",
    "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",
    "hour_1", "hour_2", "hour_3", "hour_4", "hour_5", "hour_6",
    "hour_7", "hour_8", "hour_9", "hour_10", "hour_11", "hour_12",
    "hour_13", "hour_14", "hour_15", "hour_16", "hour_17", "hour_18",
    "hour_19", "hour_20", "hour_21", "hour_22", "hour_23",
    "holiday_1",
    "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
    "workingday_1",
    "weather_2", "weather_3", "weather_4",
]

N_SAMPLES = 5000

def generate_data():
    """Generate synthetic bikeshare data with realistic patterns."""
    data = []
    for _ in range(N_SAMPLES):
        # Random continuous features
        temp = np.random.uniform(0, 1)
        humidity = np.random.uniform(0, 1)

        # Random season (one-hot, base = season 1)
        season = np.random.choice([1, 2, 3, 4])
        season_oh = [1 if season == s else 0 for s in [2, 3, 4]]

        # Random month (one-hot, base = month 1)
        month = np.random.choice(range(1, 13))
        month_oh = [1 if month == m else 0 for m in range(2, 13)]

        # Random hour (one-hot, base = hour 0)
        hour = np.random.choice(range(0, 24))
        hour_oh = [1 if hour == h else 0 for h in range(1, 24)]

        # Binary features
        holiday = np.random.choice([0, 1], p=[0.95, 0.05])
        weekday = np.random.choice(range(0, 7))
        weekday_oh = [1 if weekday == w else 0 for w in range(1, 7)]
        workingday = 1 if weekday < 5 and not holiday else 0

        weather = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.12, 0.03])
        weather_oh = [1 if weather == w else 0 for w in [2, 3, 4]]

        # Simulate realistic rental count
        base = 50
        # Temperature effect (warm = more rentals)
        base += temp * 200
        # Humidity effect (high humidity = fewer rentals)
        base -= humidity * 80
        # Season effect
        season_bonus = {1: 0, 2: 60, 3: 40, 4: -20}
        base += season_bonus[season]
        # Hour effect (peaks at 8, 12, 17)
        hour_bonus = {8: 120, 9: 80, 12: 60, 17: 150, 18: 100, 7: 50, 16: 80, 19: 60}
        base += hour_bonus.get(hour, 0)
        # Weather penalty
        weather_penalty = {1: 0, 2: -30, 3: -80, 4: -150}
        base += weather_penalty[weather]
        # Working day commute boost
        if workingday and hour in [7, 8, 9, 17, 18]:
            base += 50
        # Holiday leisure boost midday
        if holiday and 10 <= hour <= 16:
            base += 40
        # Add noise
        cnt = max(0, int(base + np.random.normal(0, 30)))

        row = [temp, humidity] + season_oh + month_oh + hour_oh + [holiday] + weekday_oh + [workingday] + weather_oh
        data.append(row + [cnt])

    columns = FEATURE_COLUMNS + ["cnt"]
    return pd.DataFrame(data, columns=columns)


def main():
    print("Generating training data...")
    df = generate_data()
    X = df[FEATURE_COLUMNS].values
    y = df["cnt"].values

    print(f"Training set: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: {y.min()} to {y.max()}, mean={y.mean():.1f}")

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Quick evaluation
    train_score = model.score(X, y)
    print(f"Training R^2: {train_score:.4f}")

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/bikeshare_rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    # Test prediction
    sample = X[0:1]
    pred = model.predict(sample)[0]
    print(f"Sample prediction: {pred:.2f} (actual: {y[0]})")


if __name__ == "__main__":
    main()
