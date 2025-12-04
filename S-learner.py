from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("EmailAnalytics.csv")

df["T_mens"] = (df["segment"] == "Mens E-Mail").astype(int)
df["T_womens"] = (df["segment"] == "Womens E-Mail").astype(int)

df_mw = df[df["segment"] != "No E-Mail"].copy()
df_mw["T_mvw"] = (df_mw["segment"] == "Mens E-Mail").astype(int)

Y = df["visit"].values
Y2 = df_mw["visit"].values

X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]], drop_first=True)
X2 = X.loc[df_mw.index]

def s_learner(X,T,Y,base_model=None):
    if base_model is None:
        base_model = RandomForestRegressor(n_estimators=500, min_samples_leaf=20, random_state=0)

    X_with_T = X.copy()
    X_with_T["T"] = T

    base_model.fit(X_with_T, Y)

    X0 = X.copy()
    X0["T"] = 0
    mu0 = base_model.predict(X0)

    X1 = X.copy()
    X1["T"] = 1
    mu1 = base_model.predict(X1)

    cate = mu1 - mu0
    return cate,mu0,mu1

cate_mens_s, mu0_m, mu1_m = s_learner(X, df["T_mens"], Y)
df["S_CATE_mens_vs_control"] = cate_mens_s

cate_wm_s, wm0, wm1 = s_learner(X, df["T_womens"], Y)
df["S_CATE_womens_vs_control"] = cate_wm_s

cate_mvw_s, mvw0, mvw1 = s_learner(X2, df_mw["T_mvw"], Y2)
df_mw["S_CATE_mens_vs_womens"] = cate_mvw_s

print("\n=== S-Learner CATE Summary Statistics ===")
print("Mens vs Control:\n", df["S_CATE_mens_vs_control"].describe())
print("\nWomens vs Control:\n", df["S_CATE_womens_vs_control"].describe())
print("\nMens vs Womens:\n", df_mw["S_CATE_mens_vs_womens"].describe())

group_vars = ["history_segment", "channel", "newbie", "zip_code"]

for var in group_vars:
    print(f"\n===== S-Learner CATE by {var} =====")
    print("Mens vs Control:\n", df.groupby(var)["S_CATE_mens_vs_control"].mean())
    print("Womens vs Control:\n", df.groupby(var)["S_CATE_womens_vs_control"].mean())

    if var in df_mw.columns:
        print("Mens vs Womens:\n", df_mw.groupby(var)["S_CATE_mens_vs_womens"].mean())

plt.hist(df["S_CATE_mens_vs_control"], bins=40, alpha=0.6)
plt.title("S-Learner: Mens vs Control CATE")
plt.show()

plt.hist(df["S_CATE_womens_vs_control"], bins=40, alpha=0.6)
plt.title("S-Learner: Womens vs Control CATE")
plt.show()

plt.hist(df_mw["S_CATE_mens_vs_womens"], bins=40, alpha=0.6)
plt.title("S-Learner: Mens vs Womens CATE")
plt.show()