import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def dr_ate(df, T_col, Y_col, X_cols):
    
    # Extract components
    T = df[T_col].values
    Y = df[Y_col].values
    X = df[X_cols].values
    
    n = len(df)

    # Propensity score model (logistic regression)

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, T)
    e_hat = ps_model.predict_proba(X)[:, 1]

    # clip extreme values
    e_hat = np.clip(e_hat, 0.01, 0.99)

    ps_mse = mean_squared_error(T, e_hat)

    # Outcome regression models (mu1, mu0)

    mu1_model = RandomForestRegressor(n_estimators=300, random_state=1)
    mu0_model = RandomForestRegressor(n_estimators=300, random_state=1)

    mu1_model.fit(X[T == 1], Y[T == 1])
    mu0_model.fit(X[T == 0], Y[T == 0])

    mu1_hat = mu1_model.predict(X)
    mu0_hat = mu0_model.predict(X)

    mu1_mse = mean_squared_error(Y[T == 1], mu1_hat[T == 1])
    mu0_mse = mean_squared_error(Y[T == 0], mu0_hat[T == 0])

    # Doubly Robust influence function

    dr_scores = (
        (T * (Y - mu1_hat)) / e_hat
        - ((1 - T) * (Y - mu0_hat)) / (1 - e_hat)
        + mu1_hat - mu0_hat
    )

    ate = np.mean(dr_scores)

    # Standard error (influence function variance)

    se = np.std(dr_scores, ddof=1) / np.sqrt(n)

    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se

    return {
        "ATE": ate,
        "SE": se,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "ps_mse": ps_mse,
        "mu1_mse": mu1_mse,
        "mu0_mse": mu0_mse
    }

df = pd.read_csv("EmailAnalytics.csv")

df_encoded = pd.get_dummies(df, drop_first=True)

df_encoded['T_mens'] = df['segment'].eq("Mens E-Mail").astype(int)
df_encoded['T_womens'] = df['segment'].eq("Womens E-Mail").astype(int)
df_encoded['T_control'] = df['segment'].eq("No E-Mail").astype(int)

X_cols = [col for col in df_encoded.columns 
          if col not in ["visit", "conversion", "spend", "segment", 
                         "T_mens", "T_womens", "T_control"]]

results = []

# --- Mens vs Control ---
df_mc = df_encoded[(df_encoded["T_mens"] == 1) | (df_encoded["T_control"] == 1)]

for outcome in ["visit", "conversion", "spend"]:
    res = dr_ate(df_mc, "T_mens", outcome, X_cols)
    results.append({
        "Comparison": "Mens vs Control",
        "Outcome": outcome.capitalize(),
        "ATE": res["ATE"],
        "SE": res["SE"],
        "CI_low": res["CI_low"],
        "CI_high": res["CI_high"],
        "ps_mse": res["ps_mse"],
        "mu1_mse": res["mu1_mse"],
        "mu0_mse": res["mu0_mse"]
    })

# --- Womens vs Control ---
df_wc = df_encoded[(df_encoded["T_womens"] == 1) | (df_encoded["T_control"] == 1)]

for outcome in ["visit", "conversion", "spend"]:
    res = dr_ate(df_wc, "T_womens", outcome, X_cols)
    results.append({
        "Comparison": "Womens vs Control",
        "Outcome": outcome.capitalize(),
        "ATE": res["ATE"],
        "SE": res["SE"],
        "CI_low": res["CI_low"],
        "CI_high": res["CI_high"],
        "ps_mse": res["ps_mse"],
        "mu1_mse": res["mu1_mse"],
        "mu0_mse": res["mu0_mse"]
    })

# --- Mens vs Women ---
df_mw = df_encoded[(df_encoded["T_mens"] == 1) | (df_encoded["T_womens"] == 1)].copy()
df_mw["T_mens_binary"] = df_mw["T_mens"]

for outcome in ["visit", "conversion", "spend"]:
    res = dr_ate(df_mw, "T_mens_binary", outcome, X_cols)
    results.append({
        "Comparison": "Mens vs Womens",
        "Outcome": outcome.capitalize(),
        "ATE": res["ATE"],
        "SE": res["SE"],
        "CI_low": res["CI_low"],
        "CI_high": res["CI_high"],
        "ps_mse": res["ps_mse"],
        "mu1_mse": res["mu1_mse"],
        "mu0_mse": res["mu0_mse"]
    })

# Convert to DataFrame
dr_table = pd.DataFrame(results)

print(dr_table)