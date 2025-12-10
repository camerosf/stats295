import pandas as pd
import numpy as np

df = pd.read_csv("EmailAnalytics.csv")

# Map common segment label variants to standardized treatment names.
def map_segment_to_treatment(s):
    if pd.isna(s):
        return np.nan
    low = str(s).lower()
    # check for women first (contains 'men' as substring)
    if 'women' in low or 'womens' in low or 'woman' in low:
        return 'Women'
    if 'mens' in low or 'men' in low or 'male' in low:
        return 'Mens'
    if 'no' in low and ('mail' in low or 'email' in low) or 'no e' in low or 'control' in low:
        return 'Control'
    return np.nan

# Create standardized treatment column
df['treatment'] = df.get('segment').apply(map_segment_to_treatment) if 'segment' in df.columns else np.nan

# Define masks based on the standardized treatment
mask_mens = df['treatment'] == 'Mens'
mask_womens = df['treatment'] == 'Women'
mask_control = df['treatment'] == 'Control'

def diff_in_means(df, outcome, mask_A, mask_B):
    mean_A = df.loc[mask_A, outcome].mean()
    mean_B = df.loc[mask_B, outcome].mean()
    return mean_A - mean_B

outcomes = ["spend", "visit", "conversion"]
results = {}

for outcome in outcomes:
    results[outcome] = {
        "Mens vs Control": diff_in_means(df, outcome, mask_mens, mask_control) if mask_mens.any() and mask_control.any() else float('nan'),
        "Womens vs Control": diff_in_means(df, outcome, mask_womens, mask_control) if mask_womens.any() and mask_control.any() else float('nan'),
        "Mens vs Women": diff_in_means(df, outcome, mask_mens, mask_womens) if mask_mens.any() and mask_womens.any() else float('nan')
    }

# Print results cleanly
for outcome, comps in results.items():
    print(f"\n--- ATE (Difference in Means) for outcome: {outcome} ---")
    for comp, val in comps.items():
        print(f"{comp}: {val:.6f}")
