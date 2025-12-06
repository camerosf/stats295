import pandas as pd
from scipy.stats import f_oneway, chi2_contingency, kruskal

df = pd.read_csv("EmailAnalytics.csv")


#testing if baseline means differ across the 3 segments of mens, womens, no email
continuous_vars = ['recency', 'history']
binary_vars = ["newbie", "mens", "womens"]
categorical_vars = ['zip_code', 'channel', 'history_segment']

#continous variables
print("======== Continuous Variables ========")
for var in continuous_vars:
    groups = [df[df['segment'] == segment][var] for segment in df['segment'].unique()]
    f_stat, p_anova = f_oneway(*groups)
    h_stat, p_kruskal = kruskal(*groups)
    
    print(f"\n{var}")
    print(f" Anova p-value = {p_anova:.4f}")
    print(f" Kruskal-Wallis p-value = {p_kruskal:.4f}")
    print(f" Means by group:")
    print(df.groupby('segment')[var].mean())

print("======== Binary Variables ========")
#binary variables
for var in binary_vars:
    contingency = pd.crosstab(df['segment'], df[var])
    chi2, p, dof, expected = chi2_contingency(contingency)

    print(f"\n{var}")
    print(f" Chi-square p-value = {p:.4f}")
    print(df.groupby('segment')[var].mean()*100)

print("======== Categorial Variables ========")
for var in categorical_vars:
    contingency = pd.crosstab(df['segment'], df[var])
    chi2, p, dof, expected = chi2_contingency(contingency)

    print(f"\n{var}")
    print(f" Chi-square p-value = {p:.4f}")
    print(pd.crosstab(df['segment'], df[var], normalize='index')*100)
