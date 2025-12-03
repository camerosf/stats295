import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


df = pd.read_csv("EmailAnalytics.csv")

# Treatment encodings for each comparison
df["T_mens"] = (df["segment"] == "Mens E-Mail").astype(int)
df["T_womens"] = (df["segment"] == "Womens E-Mail").astype(int)

# Filter dataset for Mens vs Womens (exclude control)
df_mw = df[df["segment"] != "No E-Mail"].copy()
df_mw["T_mvw"] = (df_mw["segment"] == "Mens E-Mail").astype(int)

Y = df["visit"].values
Y2 = df_mw["visit"].values

# Pre-treatment covariates
X_cols = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
X = pd.get_dummies(df[X_cols], drop_first=True)
X2 = X.loc[df_mw.index]   # align X for mens vs womens subset

# ============================
# 2. CAUSAL FOREST: MENS vs CONTROL
# ============================

T = df["T_mens"].values

forest_mens = CausalForestDML(
    model_t=RandomForestRegressor(),
    model_y=RandomForestRegressor(),
    n_estimators=1000,
    min_samples_leaf=50,
    random_state=0
)

forest_mens.fit(Y, T, X=X)
df["CATE_mens_vs_control"] = forest_mens.effect(X)

# ============================
# 3. CAUSAL FOREST: WOMENS vs CONTROL
# ============================

T = df["T_womens"].values

forest_womens = CausalForestDML(
    model_t=RandomForestRegressor(),
    model_y=RandomForestRegressor(),
    n_estimators=1000,
    min_samples_leaf=50,
    random_state=0
)

forest_womens.fit(Y, T, X=X)
df["CATE_womens_vs_control"] = forest_womens.effect(X)

# ============================
# 4. CAUSAL FOREST: MENS vs WOMENS (NO CONTROL)
# ===========================

T2 = df_mw["T_mvw"].values

forest_mvw = CausalForestDML(
    model_t=RandomForestRegressor(),
    model_y=RandomForestRegressor(),
    n_estimators=1000,
    min_samples_leaf=50,
    random_state=0
)

forest_mvw.fit(Y2, T2, X=X2)
df_mw["CATE_mens_vs_womens"] = forest_mvw.effect(X2)

# ============================
# 5. SUMMARY STATISTICS
# ============================

print("\n=== CATE Summary Statistics ===")
print("Mens vs Control:\n", df["CATE_mens_vs_control"].describe())
print("\nWomens vs Control:\n", df["CATE_womens_vs_control"].describe())
print("\nMens vs Womens:\n", df_mw["CATE_mens_vs_womens"].describe())

# ============================
# 6. CATE BY SEGMENT VARIABLES
# ============================

group_vars = ["history_segment", "newbie", "zip_code", "channel"]

for var in group_vars:
    print(f"\n===== CATE by {var} =====")
    print("Mens vs Control:\n", df.groupby(var)["CATE_mens_vs_control"].mean())
    print("Womens vs Control:\n", df.groupby(var)["CATE_womens_vs_control"].mean())

    if var in df_mw.columns:
        print("Mens vs Womens:\n", df_mw.groupby(var)["CATE_mens_vs_womens"].mean())



plt.figure(figsize=(8, 5))
plt.hist(df["CATE_mens_vs_control"], bins=40, alpha=0.6, label="Mens vs Control")
plt.hist(df["CATE_womens_vs_control"], bins=40, alpha=0.6, label="Womens vs Control")
plt.title("CATE Distribution")
plt.xlabel("Estimated Treatment Effect")
plt.ylabel("Number of Customers")
plt.legend()
plt.show()

# Histogram for Mens vs Womens
plt.figure(figsize=(8, 5))
plt.hist(df_mw["CATE_mens_vs_womens"], bins=40, alpha=0.7, color="purple")
plt.title("CATE Distribution: Mens vs Womens")
plt.xlabel("Estimated Treatment Effect")
plt.ylabel("Number of Customers")
plt.show()
