import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv("EmailAnalytics.csv")

# One-hot encode the treatment (3-arm)
df = pd.get_dummies(df, columns=["segment"], drop_first=False)

# Rename to simplify
df = df.rename(columns={
    "segment_No E-Mail": "T0",
    "segment_Mens E-Mail": "T1",
    "segment_Womens E-Mail": "T2"
})

# Covariates
X = df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]]
X = pd.get_dummies(X, drop_first=True)

# Add treatment dummies (NO control dummy)
X_with_T = X.join(df[["T1", "T2"]])

# Outcomes
outcomes = {
    "spend": df["spend"],
    "visit": df["visit"],
    "conversion": df["conversion"]
}

ATE_results = {}

for name, Y in outcomes.items():
    print(f"\n--- DIRECT METHOD ATE for outcome: {name} ---")

    # Fit the S-Learner
    model = GradientBoostingRegressor()
    model.fit(X_with_T, Y)

    # Counterfactual datasets
    X0 = X_with_T.copy(); X0["T1"] = 0; X0["T2"] = 0  # Control
    X1 = X_with_T.copy(); X1["T1"] = 1; X1["T2"] = 0  # Mens
    X2 = X_with_T.copy(); X2["T1"] = 0; X2["T2"] = 1  # Womens

    # Predict counterfactual outcomes
    Y0_hat = model.predict(X0)
    Y1_hat = model.predict(X1)
    Y2_hat = model.predict(X2)

    # Average treatment effects
    ATE_results[name] = {
        "Mens vs Control": np.mean(Y1_hat - Y0_hat),
        "Women vs Control": np.mean(Y2_hat - Y0_hat),
        "Mens vs Women":  np.mean(Y1_hat - Y2_hat)
    }

    print(ATE_results[name])
