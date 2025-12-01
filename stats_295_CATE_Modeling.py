from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Load dataset
df = pd.read_csv("EmailAnalytics.csv")

# --- Treatment variable (binary): any email vs no email ---
df['treat'] = ((df['segment'] == "Mens E-Mail") | (df['segment'] == "Womens E-Mail")).astype(int)

# --- Outcome variable ---
Y = df['visit'].values     # or conversion, or spend

# --- CATE Covariates (pre-treatment only) ---
# Example: convert suburban / rural / urban into indicator variables
df['history_segment'] = df['history_segment'].astype(str)
df['zip_code'] = df['zip_code'].astype(str)
df['channel'] = df['channel'].astype(str)

X = df[['recency', 'history', 'newbie', 'channel', 'zip_code']]

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Treatment array
T = df['treat'].values

# Causal Forest Model
forest = CausalForestDML(
    model_t=RandomForestRegressor(),     # model for treatment
    model_y=RandomForestRegressor(),     # model for outcome
    n_estimators=1000,
    min_samples_leaf=50,
    random_state=0
)

# Fit the model
forest.fit(Y, T, X=X)

df['CATE'] = forest.effect(X)

print(df.groupby('history_segment')['CATE'].mean())
print(df.groupby('channel')['CATE'].mean())
print(df.groupby('newbie')['CATE'].mean())
print(df.groupby('zip_code')['CATE'].mean())

#shows by category increased chance of visiting site 
#Around 5-6% increase for mid range spenders while 7-9% increase for high spenders
#consistent difference across all channels or web/phone/multichannel
#newbie status aka new customer or not essentially no difference
#urban/suburban customers respond more than rural but barely more)

import matplotlib.pyplot as plt

plt.hist(df['CATE'], bins=40)
plt.xlabel("Estimated CATE")
plt.ylabel("Number of Customers")
plt.title("Distribution of Treatment Effects")
plt.show()

#graphs the overall difference between email and no email on going to site