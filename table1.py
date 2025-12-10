import pandas as pd
import numpy as np

df = pd.read_csv("EmailAnalytics.csv")

# Standardize segment naming
df['Segment'] = df['segment'].replace({
    'Mens E-Mail': 'Mens',
    'Womens E-Mail': 'Womens',
    'No E-Mail': 'Control'
})

# ---------------------------------------------------
# 1. Treatment group sizes
# ---------------------------------------------------
group_sizes = df['Segment'].value_counts()

# ---------------------------------------------------
# 2. Continuous variables (edit list if needed)
# ---------------------------------------------------
cont_vars = ['recency', 'history']

def get_cont_stats(var):
    return {
        g: f"{df.loc[df['Segment'] == g, var].mean():.1f} ({df.loc[df['Segment'] == g, var].std():.1f})"
        for g in ['Control', 'Mens', 'Womens']
    }

cont_results = {var: get_cont_stats(var) for var in cont_vars}

# ---------------------------------------------------
# 3. Categorical variables — includes ALL requested variables
# ---------------------------------------------------
cat_vars = ['history_segment', 'mens', 'womens', 'zip_code', 'newbie', 'channel']
cat_results = {}

for var in cat_vars:
    counts = pd.crosstab(df[var], df['Segment'])
    pct = pd.crosstab(df[var], df['Segment'], normalize='columns') * 100
    combined = counts.astype(str) + " (" + pct.round(1).astype(str) + "%)"
    cat_results[var] = combined

# ---------------------------------------------------
# 4. Build Table 1 rows
# ---------------------------------------------------
rows = []

# Number of customers
total = group_sizes['Control'] + group_sizes['Mens'] + group_sizes['Womens']
rows.append([
    "Number of Customers",
    f"{group_sizes['Control']} ({group_sizes['Control']/total*100:.1f}%)",
    f"{group_sizes['Mens']} ({group_sizes['Mens']/total*100:.1f}%)",
    f"{group_sizes['Womens']} ({group_sizes['Womens']/total*100:.1f}%)"
])

# Continuous variables
for var in cont_vars:
    rows.append([
        var.replace('_', ' ').capitalize() + " (mean (SD))",
        cont_results[var]['Control'],
        cont_results[var]['Mens'],
        cont_results[var]['Womens']
    ])

# Categorical variables
for var in cat_vars:
    # Header row for this variable
    rows.append([var.replace('_', ' ').capitalize(), "", "", ""])
    
    # Each level under it
    for level in cat_results[var].index:
        rows.append([
            f"  {level}",
            cat_results[var].loc[level, 'Control'],
            cat_results[var].loc[level, 'Mens'],
            cat_results[var].loc[level, 'Womens']
        ])

# ---------------------------------------------------
# 5. Convert to DataFrame
# ---------------------------------------------------
table1 = pd.DataFrame(rows, columns=["Characteristic", "Control", "Mens", "Womens"])

print(table1)

