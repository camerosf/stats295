import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv("EmailAnalytics.csv")

# map treatment
mapping = {
    "No E-Mail": 0,
    "Mens E-Mail": 1,
    "Womens E-Mail": 2
}
df["T"] = df["segment"].map(mapping)

# outcome variable (lowercase in your file)
Y = df["spend"].values

# covariates (all lowercase)
feature_cols = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]

# one-hot encode categorical vars
X = pd.get_dummies(df[feature_cols], drop_first=True)

# add treatment column
X_with_T = X.copy()
X_with_T["T"] = df["T"]

# fit model
model = GradientBoostingRegressor()
model.fit(X_with_T, Y)

# create copies for each treatment scenario
X0 = X_with_T.copy()
X1 = X_with_T.copy()
X2 = X_with_T.copy()

X0["T"] = 0     # No Email
X1["T"] = 1     # Mens Email
X2["T"] = 2     # Womens Email

Y0_hat = model.predict(X0)
Y1_hat = model.predict(X1)
Y2_hat = model.predict(X2)

# ATE estimates
ATE_mens = np.mean(Y1_hat - Y0_hat)
ATE_womens = np.mean(Y2_hat - Y0_hat)
ATE_mens_vs_women = np.mean(Y1_hat - Y2_hat)

print("ATE: Mens Email vs Control =", ATE_mens)
print("ATE: Womens Email vs Control =", ATE_womens)
print("ATE: Mens Email vs Womens Email =", ATE_mens_vs_women)

# CATE storage
df["CATE_mens"] = Y1_hat - Y0_hat
df["CATE_womens"] = Y2_hat - Y0_hat
df["CATE_mens_vs_womens"] = Y1_hat - Y2_hat

print(df[["CATE_mens", "CATE_womens", "CATE_mens_vs_womens"]].head())

print("Running diff-in-means only...")

ATE_mens_DM = df[df.T==1]["visit"].mean() - df[df.T==0]["visit"].mean()
ATE_womens_DM = df[df.T==2]["visit"].mean() - df[df.T==0]["visit"].mean()
ATE_m_vs_w_DM = df[df.T==1]["visit"].mean() - df[df.T==2]["visit"].mean()

print("Diff Means OK")