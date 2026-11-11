# STAGE 2: load preprocessed data, scale, choose k, train K-Means, evaluate, save model.
# Note: clustering is unsupervised (no target column), so there is no train/test split.
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_PATH = "data/processed/mall_preprocessed.csv"
FEATURES = ["Age", "Income", "SpendingScore"]
N_CLUSTERS = 6          # chosen from elbow + silhouette (silhouette peaks at 6)

ACTIONS = {
    "Premium Big Spenders": "VIP loyalty tiers, early access to new collections, premium brands.",
    "Careful High-Earners": "Quality and value messaging, exclusive offers and personal shopping to raise spending.",
    "Low-Income Enthusiasts": "Student/youth discounts, trend-led flash sales, buy-now-pay-later offers.",
    "Budget Savers": "Coupons, bundle deals and essentials promotions; low-cost retention.",
    "Young Average Shoppers": "Social media campaigns, app rewards and experiences to lift spending.",
    "Older Average Shoppers": "Comfort, service and family offers; in-store events and loyalty points.",
}


def name_segment(age, income, score):
    income_level = "high" if income >= 70 else ("low" if income <= 40 else "mid")
    score_level = "high" if score >= 65 else ("low" if score <= 35 else "mid")
    if income_level == "high" and score_level == "high":
        return "Premium Big Spenders"
    if income_level == "high" and score_level == "low":
        return "Careful High-Earners"
    if income_level == "low" and score_level == "high":
        return "Low-Income Enthusiasts"
    if income_level == "low" and score_level == "low":
        return "Budget Savers"
    return "Young Average Shoppers" if age < 40 else "Older Average Shoppers"


df = pd.read_csv(DATA_PATH)
X = df[FEATURES]

# K-Means is distance based, so scale features first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow + silhouette for k = 2..10
ks = list(range(2, 11))
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))
    print("k =", k, "| inertia =", round(km.inertia_, 1), "| silhouette =", round(sils[-1], 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertias, marker="o")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sils, marker="o", color="green")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("k")
plt.tight_layout()
plt.savefig("app/static/elbow.png", dpi=110)
plt.close()

# Final model
model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)
print("Final silhouette:", round(silhouette_score(X_scaled, df["Cluster"]), 3))
print("Davies-Bouldin (lower is better):", round(davies_bouldin_score(X_scaled, df["Cluster"]), 3))

# Segment profiles in original units
profile = df.groupby("Cluster")[FEATURES].mean().round(1)
profile["Customers"] = df.groupby("Cluster").size()
segment_info = {}
for cluster_id, row in profile.iterrows():
    name = name_segment(row["Age"], row["Income"], row["SpendingScore"])
    segment_info[int(cluster_id)] = {"name": name, "action": ACTIONS[name]}
    profile.loc[cluster_id, "Segment"] = name
print(profile)

df["Segment"] = df["Cluster"].map(lambda c: segment_info[c]["name"])
df.to_csv("data/processed/mall_segments.csv", index=False)

# Cluster chart: Income vs Spending Score
plt.figure(figsize=(7, 5))
for name in df["Segment"].unique():
    part = df[df["Segment"] == name]
    plt.scatter(part["Income"], part["SpendingScore"], label=name, s=35)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("app/static/segments.png", dpi=110)
plt.close()

# Save with joblib
joblib.dump(model, "models/kmeans_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(segment_info, "models/segment_info.joblib")
print("Saved model, scaler and segment info in models/")
