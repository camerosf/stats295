import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("EmailAnalytics.csv")

# Treatment mapping
mapping = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
df["T"] = df["segment"].map(mapping)

# Covariates
feature_cols = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
X = pd.get_dummies(df[feature_cols], drop_first=True)
X_with_T = X.copy()
X_with_T["T"] = df["T"]

# Outcomes you want ATE for
outcomes = {
    "spend": df["spend"],
    "visit": df["visit"],
    "conversion": df["conversion"]
}

ATE_results = {}

for name, Y in outcomes.items():
    print(f"\n--- Estimating ATE for outcome: {name} ---")

    model = GradientBoostingRegressor()
    model.fit(X_with_T, Y)

    X0 = X_with_T.copy(); X0["T"] = 0
    X1 = X_with_T.copy(); X1["T"] = 1
    X2 = X_with_T.copy(); X2["T"] = 2

    Y0_hat = model.predict(X0)
    Y1_hat = model.predict(X1)
    Y2_hat = model.predict(X2)

    ATE_results[name] = {
        "Mens vs Control": np.mean(Y1_hat - Y0_hat),
        "Women vs Control": np.mean(Y2_hat - Y0_hat),
        "Mens vs Women":  np.mean(Y1_hat - Y2_hat)
    }

    print(ATE_results[name])
