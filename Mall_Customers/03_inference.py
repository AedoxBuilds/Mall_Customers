# STAGE 3: load saved model, create new data, predict segments.
import pandas as pd
import joblib

FEATURES = ["Age", "Income", "SpendingScore"]

model = joblib.load("models/kmeans_model.joblib")
scaler = joblib.load("models/scaler.joblib")
segment_info = joblib.load("models/segment_info.joblib")

# New customers (Age, Income in k$, SpendingScore 1-100)
new_data = pd.DataFrame(
    [[22, 20, 85], [45, 95, 15], [30, 90, 88], [58, 55, 50]],
    columns=FEATURES,
)

new_scaled = scaler.transform(new_data)
clusters = model.predict(new_scaled)

new_data["Cluster"] = clusters
new_data["Segment"] = [segment_info[int(c)]["name"] for c in clusters]
new_data["Marketing Action"] = [segment_info[int(c)]["action"] for c in clusters]
print(new_data.to_string(index=False))
