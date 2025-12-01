import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

#Direct Method usingg GradientBoostingRegressor can use and learner
df = pd.read_csv("EmailAnalytics.csv")

mapping = {
    "No E-Mail": 0,
    "Mens E-Mail": 1,
    "Womens E-Mail": 2
}
df["T"] = df["segment"].map(mapping)

# Outcome variable
Y = df["spend"].values

# Covariates
feature_cols = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
X = pd.get_dummies(df[feature_cols], drop_first=True)

# Add treatment to covariates for model fitting
X_with_T = X.copy()
X_with_T["T"] = df["T"]

# Fit outcome regression model
model = GradientBoostingRegressor()
model.fit(X_with_T, Y)

# Copy covariates and set treatment levels for potential outcomes
X0 = X_with_T.copy()
X1 = X_with_T.copy()
X2 = X_with_T.copy()

X0["T"] = 0       # No Email
X1["T"] = 1       # Mens Email
X2["T"] = 2       # Womens Email

Y0_hat = model.predict(X0)   # Potential outcome under No Email
Y1_hat = model.predict(X1)   # Potential outcome under Mens Email
Y2_hat = model.predict(X2)   # Potential outcome under Womens Email


ATE_mens = np.mean(Y1_hat - Y0_hat)
print("ATE: Mens Email vs Control =", ATE_mens)

ATE_womens = np.mean(Y2_hat - Y0_hat)
print("ATE: Womens Email vs Control =", ATE_womens)

ATE_mens_vs_women = np.mean(Y1_hat - Y2_hat)
print("ATE: Mens Email vs Womens Email =", ATE_mens_vs_women)


#Using the Direct Method (outcome regression), we estimated the causal effect of each email campaign on customer spending.
#The Mens Email increased average spending by $0.58 relative to the control group, while the Womens Email increased spending by $0.39.
#The Mens Email outperformed the Womens Email by approximately $0.19 per customer.
#These results suggest that the Mens Email is the most effective campaign overall, generating the highest incremental revenue per customer.