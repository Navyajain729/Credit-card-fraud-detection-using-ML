"""Flask backend: serves the UI and a JSON prediction API."""
import joblib, pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
bundle = joblib.load("model.joblib")
pipe, meta = bundle["pipeline"], bundle["meta"]
FEATURES = meta["features"]

REQUIRED_NUM = ["Age", "Credit_Limit_INR", "Transaction_Amount_INR", "Transaction_Hour",
                "Previous_Transactions_24h", "Average_Transaction_Amount_INR",
                "Account_Age_Months"]
REQUIRED_CAT = list(meta["categories"].keys())


def risk_reasons(r, prob):
    out = []
    avg = r["Average_Transaction_Amount_INR"]
    if avg and r["Transaction_Amount_INR"] > 3 * avg:
        out.append(f"Amount is {r['Transaction_Amount_INR'] / avg:.1f}x the customer's average")
    if r["Credit_Utilization"] > 0.25:
        out.append(f"Uses {r['Credit_Utilization'] * 100:.0f}% of the credit limit in one transaction")
    if r["Transaction_Hour"] < 5:
        out.append("Made late at night (00:00-05:00)")
    if r["International_Transaction"] == "Yes":
        out.append("International transaction")
    if r["Previous_Transactions_24h"] >= 8:
        out.append(f"{int(r['Previous_Transactions_24h'])} transactions in the last 24h")
    return out


@app.get("/")
def home():
    return render_template("home.html")


@app.get("/check")
def check():
    return render_template("check.html")


@app.get("/api/meta")
def get_meta():
    return jsonify({"categories": meta["categories"], "ranges": meta["ranges"],
                    "source": meta["source"]})


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    row, errors = {}, []
    for k in REQUIRED_NUM:
        try:
            row[k] = float(data[k])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{k} must be a number")
    for k in REQUIRED_CAT:
        v = data.get(k)
        if v not in meta["categories"][k]:
            errors.append(f"{k} must be one of {meta['categories'][k]}")
        row[k] = v
    if not errors:
        if row["Credit_Limit_INR"] <= 0:
            errors.append("Credit_Limit_INR must be > 0")
        if not 0 <= row["Transaction_Hour"] <= 23:
            errors.append("Transaction_Hour must be 0-23")
        if row["Transaction_Amount_INR"] < 0:
            errors.append("Transaction_Amount_INR must be >= 0")
    if errors:
        return jsonify({"errors": errors}), 400

    row["Credit_Utilization"] = round(row["Transaction_Amount_INR"] / row["Credit_Limit_INR"], 4)
    prob = float(pipe.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
    level = "HIGH" if prob >= .6 else "MEDIUM" if prob >= .3 else "LOW"
    return jsonify({"fraud_probability": round(prob, 4), "risk_level": level,
                    "is_fraud": prob >= .5, "credit_utilization": row["Credit_Utilization"],
                    "reasons": risk_reasons(row, prob)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
