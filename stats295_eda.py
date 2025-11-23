import pandas as pd
df = pd.read_csv("EmailAnalytics.csv")

missingness_count = df.isnull().sum()

email_type = df[['mens', 'womens']].sum()

visit_average = df['visit'].mean()
conversion_average = df['conversion'].mean()

spend_distribution = df[df['conversion'] == 1]['spend'].describe()

print(f"missingness_count: {missingness_count}")
#we see there is no missing values

print(f"email type: {email_type}")

print(f"Visit Average: {visit_average}")

print(f"Conversion Average: {conversion_average}")

print(f"spend distribution: {spend_distribution}")

#visit_rates = df.groupby('womens')['visit'].mean()
#print("No womens email visit rate:", visit_rates[0])
#print("Womens email visit rate:", visit_rates[1])

print(f"Conversion rate men: {df.groupby('mens')['conversion'].mean()}")
print(f"Conversion rate women: {df.groupby('womens')['conversion'].mean()}")

#percentage of sample
print("Percentage of sample")
print(df['segment'].value_counts(normalize=True) * 100)

#Recency
print("Recency mean and std")
print(df.groupby('segment')['recency'].mean())
print(df.groupby('segment')['recency'].std())

#historical spending
print("Historical spending category")
print(df.groupby('segment')['history_segment'].value_counts(normalize=True)*100)

#historical spending average
print("Historical spending mean and std")
print(df.groupby('segment')['history'].mean())
print(df.groupby('segment')['history'].std())

#Purchased a men's merchandise last year?
percent_by_segment = df.groupby('segment')['mens'].mean() * 100
print("Purchased a men's merchandise last year")
print(percent_by_segment)

#Purchased a women's merchandise last year?
percent_by_segment = df.groupby('segment')['womens'].mean() * 100
print("Purchased a women's merchandise last year")
print(percent_by_segment)

#regional area
print("Regional Area")
print(df.groupby('segment')['zip_code'].value_counts(normalize=True) * 100)

#new customer
print("New Customer")
print(df.groupby('segment')['newbie'].value_counts(normalize=True) * 100)

#site to purchase goods
print("Medium purchasing goods")
print(df.groupby('segment')['channel'].value_counts(normalize=True) * 100)

#visited site at least once following email campaign
print("visited site")
print(df.groupby('segment')['visit'].value_counts(normalize=True)*100)

#bought merchandise off site at least once following email campaign
print("purchase on site")
print(df.groupby('segment')['conversion'].value_counts(normalize=True)*100)

#average money spent given bought item
bought_item = df[df['conversion'] == 1]
print(bought_item.groupby('segment')['spend'].mean())


# Testing ECONML Pacakge and git stuff
mapping = {
    "Mens E-Mail": 1,
    "Womens E-Mail": 2,
    "No E-Mail": 0
}
df["T"] = df["segment"].map(mapping)

# --- Outcome variable (choose spend or conversion) ---
Y = df["spend"].values
T = df["T"].values

# --- Feature matrix ---
feature_cols = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
X = pd.get_dummies(df[feature_cols], drop_first=True).values

# ---- DR Learner ----
from econml.dr import DRLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

dr = DRLearner(
    model_regression=RandomForestRegressor(),
    model_propensity=LogisticRegression()
)

dr.fit(Y, T, X=X)
ATE = dr.ate(X=X)
print("===== ECONML RESULTS =====")
print(f"ATE (Doubly Robust): {ATE}")
