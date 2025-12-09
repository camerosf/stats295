import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

# ---------------------------------------------------
# 1. Generate covariates (Hillstrom-like)
# ---------------------------------------------------

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

zip_code = np.random.choice(["Urban", "Suburban", "Rural"], n, p=[0.4, 0.4, 0.2])
channel = np.random.choice(["Web", "Phone"], n, p=[0.7, 0.3])
newbie = np.random.binomial(1, 0.2, n)

# ---------------------------------------------------
# 2. 3-arm treatment with TRUE confounding
# ---------------------------------------------------

# latent logits (depend on confounders)
logit_m = 0.06*recency + 0.004*history - 0.3*newbie + 0.2*(zip_code=="Urban")
logit_w = 0.04*recency + 0.005*history + 0.3*newbie - 0.1*(zip_code=="Rural")

# convert to probabilities
p_m = 1/(1 + np.exp(-logit_m))
p_w = 1/(1 + np.exp(-logit_w))

# normalize to 3-arm probabilities
p_c = 1 - (p_m + p_w)
p_c = np.clip(p_c, 0.0001, 0.9999)

# renormalize so probabilities sum to 1
total = p_m + p_w + p_c
p_m /= total
p_w /= total
p_c /= total

# assign treatment INDIVIDUALLY (correct)
segment = np.array([
    np.random.choice(
        ["Mens E-Mail", "Womens E-Mail", "No E-Mail"],
        p=[p_m[i], p_w[i], p_c[i]]
    )
    for i in range(n)
])

# encode treatment 0/1/2
T = (
    (segment == "No E-Mail") * 0 +
    (segment == "Mens E-Mail") * 1 +
    (segment == "Womens E-Mail") * 2
)

# ---------------------------------------------------
# 3. TRUE treatment effects
# ---------------------------------------------------
true_tau_mens_visit = 0.05
true_tau_womens_visit = 0.03

true_tau_mens_spend = 1.50
true_tau_womens_spend = 1.00

# ---------------------------------------------------
# 4. Visit (binary)
# ---------------------------------------------------

base_visit_prob = (
    0.05
    + 0.01*(history/300)
    + 0.02*(zip_code=="Urban")
    + 0.03*(channel=="Web")
    - 0.04*newbie
)

visit_prob = (
    base_visit_prob
    + (T == 1)*true_tau_mens_visit
    + (T == 2)*true_tau_womens_visit
)

visit_prob = np.clip(visit_prob, 0, 1)
visit = np.random.binomial(1, visit_prob, n)

# ---------------------------------------------------
# 5. Conversion = scaled visit probability
# ---------------------------------------------------

conv_prob = 0.20 * visit_prob
conv_prob = np.clip(conv_prob, 0, 1)
conversion = np.random.binomial(1, conv_prob, n)

# ---------------------------------------------------
# 6. Spend (continuous)
# ---------------------------------------------------

baseline_spend = (
    0.3*recency
    + 0.01*history
    + 0.8*visit
    + np.random.normal(0, 1, n)
)

spend = (
    baseline_spend
    + (T == 1)*true_tau_mens_spend
    + (T == 2)*true_tau_womens_spend
)

# ---------------------------------------------------
# 7. Final dataset
# ---------------------------------------------------

df_syn = pd.DataFrame({
    "recency": recency,
    "history": history,
    "history_segment": history_segment,
    "zip_code": zip_code,
    "channel": channel,
    "newbie": newbie,
    "segment": segment,
    "T": T,
    "visit": visit,
    "conversion": conversion,
    "spend": spend,
    "true_tau_mens_visit": true_tau_mens_visit,
    "true_tau_womens_visit": true_tau_womens_visit,
    "true_tau_mens_spend": true_tau_mens_spend,
    "true_tau_womens_spend": true_tau_womens_spend
})

print(df_syn.head())
df_syn.to_csv("Synthetic_data_correct.csv", index=False)
print("Saved as Synthetic_data_correct.csv")
