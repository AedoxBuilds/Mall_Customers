import pandas as pd
import joblib

FEATURES = ["Age", "Income", "SpendingScore"]

model = joblib.load("models/kmeans_model.joblib")
scaler = joblib.load("models/scaler.joblib")
segment_info = joblib.load("models/segment_info.joblib")


def predict_segment(age, income, score):
    data = pd.DataFrame([[age, income, score]], columns=FEATURES)
    cluster = int(model.predict(scaler.transform(data))[0])
    info = segment_info[cluster]
    return {"cluster": cluster, "segment": info["name"], "action": info["action"]}
