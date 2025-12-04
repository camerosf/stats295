import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

# 1. Generate covariates (same structure as Hillstrom)
recency = np.random.randint(1, 13, n)
history = np.abs(np.random.normal(300, 150, n))

bins = [0, 100, 200, 350, 500, 750, 10000]
labels = [
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750+"
]
history_segment = pd.cut(history, bins=bins, labels=labels, right=False)

mens = np.random.binomial(1, 0.4, n)
womens = np.random.binomial(1, 0.6, n)
zip_code = np.random.choice(["Urban", "Suburban", "Rural"], n, p=[0.4, 0.4, 0.2])
channel = np.random.choice(["Web", "Phone"], n, p=[0.7, 0.3])
newbie = np.random.binomial(1, 0.2, n)

# 2. Confounded 3-arm treatment assignment
logit_m = 0.05*recency + 0.003*history + 0.5*mens - 0.3*newbie
logit_w = 0.04*recency + 0.004*history + 0.5*womens + 0.2*newbie

p_m = 1/(1 + np.exp(-logit_m))
p_w = 1/(1 + np.exp(-logit_w))

p_m = p_m / (p_m + p_w + 1)
p_w = p_w / (p_m + p_w + 1)
p_c = 1 - p_m - p_w

segment = np.random.choice(
    ["Mens E-Mail", "Womens E-Mail", "No E-Mail"],
    size=n,
    p=[p_m.mean(), p_w.mean(), p_c.mean()]
)

T = (
    (segment == "No E-Mail") * 0 +
    (segment == "Mens E-Mail") * 1 +
    (segment == "Womens E-Mail") * 2
)

# 3. True causal effects for the synthetic experiment
true_tau_mens_visit = 0.05
true_tau_womens_visit = 0.03
true_tau_mens_spend = 1.50
true_tau_womens_spend = 1.00

# 4. Visit (binary outcome)
base_visit_prob = (
    0.05
    + 0.01*(history/300)
    + 0.05*mens
    + 0.04*womens
    + 0.02*(zip_code=="Urban")
    - 0.03*newbie
)

visit_prob = (
    base_visit_prob
    + (T == 1) * true_tau_mens_visit
    + (T == 2) * true_tau_womens_visit
)

visit_prob = np.clip(visit_prob, 0, 1)
visit = np.random.binomial(1, visit_prob, n)

# 5. Conversion (binary outcome)
conv_prob = 0.2 * visit_prob
conversion = np.random.binomial(1, conv_prob, n)

# 6. Spend (continuous outcome)
baseline_spend = (
    0.3*recency
    + 0.01*history
    + 1.5*mens
    + 1.0*womens
    + 0.8*visit
    + np.random.normal(0, 1, n)
)

spend = (
    baseline_spend
    + (T == 1)*true_tau_mens_spend
    + (T == 2)*true_tau_womens_spend
)

# 7. Create final DataFrame
df_syn = pd.DataFrame({
    "recency": recency,
    "history_segment": history_segment,
    "history": history,
    "mens": mens,
    "womens": womens,
    "zip_code": zip_code,
    "newbie": newbie,
    "channel": channel,
    "segment": segment,
    "visit": visit,
    "conversion": conversion,
    "spend": spend,
    "T": T
})

print(df_syn.head())

df_syn.to_csv("Synthetic_data.csv", index=False)
print("Synthetic dataset saved as Synthetic_data.csv")