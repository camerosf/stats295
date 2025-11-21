import pandas as pd
print("Test - Nathan")
df = pd.read_csv("EmailAnalytics.csv")

#df.info()
#df.describe(include="all")

#categorical_cols = ["history_segment", "zip_code", "channel", "segment"]
#for col in categorical_cols:
#    df[col] = df[col].astype("category")

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

visit_rates = df.groupby('womens')['visit'].mean()
print("No womens email visit rate:", visit_rates[0])
print("Womens email visit rate:", visit_rates[1])

print(f"Conversion rate men: {df.groupby('mens')['conversion'].mean()}")

print(f"Conversion rate women: {df.groupby('womens')['conversion'].mean()}")

print(f"purchase mediums: {df["channel"].unique()}")

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
df['purchased_last_year_men'] = ((df['mens'] == 1)).astype(int)
percent_by_segment = df.groupby('segment')['purchased_last_year_men'].mean() * 100
print("Purchased a men's merchandise last year")
print(percent_by_segment)

#Purchased a women's merchandise last year?
df['purchased_last_year_women'] = ((df['womens'] == 1)).astype(int)
percent_by_segment = df.groupby('segment')['purchased_last_year_women'].mean() * 100
print("Purchased a women's merchandise last year")
print(percent_by_segment)