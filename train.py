"""Train the fraud model (same idea as the notebook, but production-safe).

Usage:  python train.py [path/to/navya_project.csv]

Fixes vs. the notebook:
  * one encoder per categorical column (the notebook reused one LabelEncoder)
  * Person_ID dropped (an identifier, not a signal)
  * encoder + model saved together so the API encodes inputs identically
  * class_weight="balanced" because fraud is only 8% of rows
If the CSV is missing, synthetic data with the same schema is generated so the
app can be demoed. Replace it with your real CSV for real predictions.
"""
import sys, os, json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TARGET = "transaction type"
CATEGORICAL = ["Merchant_Category", "Transaction_Type", "Location",
               "Device_Type", "International_Transaction"]
NUMERIC = ["Age", "Credit_Limit_INR", "Transaction_Amount_INR", "Transaction_Hour",
           "Previous_Transactions_24h", "Average_Transaction_Amount_INR",
           "Account_Age_Months", "Credit_Utilization"]
FEATURES = ["Age", "Credit_Limit_INR", "Transaction_Amount_INR", "Merchant_Category",
            "Transaction_Type", "Location", "Device_Type", "Transaction_Hour",
            "Previous_Transactions_24h", "Average_Transaction_Amount_INR",
            "Account_Age_Months", "International_Transaction", "Credit_Utilization"]


def synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    cats = {
        "Merchant_Category": ["Fuel", "Electronics", "Entertainment", "Healthcare", "Travel",
                              "Dining", "Utilities", "Online Shopping", "Groceries", "Clothing"],
        "Transaction_Type": ["Online", "Contactless", "POS", "ATM"],
        "Location": ["Bengaluru", "Hyderabad", "Mumbai", "Ahmedabad", "Pune",
                     "Lucknow", "Chennai", "Delhi", "Kolkata", "Jaipur"],
        "Device_Type": ["Mobile", "POS Terminal", "Laptop", "ATM"],
    }
    df = pd.DataFrame({
        "Age": rng.integers(18, 76, n),
        "Credit_Limit_INR": (rng.gamma(4, 24000, n) // 100 * 100).clip(12400, 480700).astype(int),
        "Merchant_Category": rng.choice(cats["Merchant_Category"], n),
        "Transaction_Type": rng.choice(cats["Transaction_Type"], n),
        "Location": rng.choice(cats["Location"], n),
        "Device_Type": rng.choice(cats["Device_Type"], n),
        "Transaction_Hour": rng.integers(0, 24, n),
        "Previous_Transactions_24h": rng.poisson(4, n),
        "Average_Transaction_Amount_INR": rng.gamma(3, 1300, n).round(2),
        "Account_Age_Months": rng.integers(3, 181, n),
        "International_Transaction": rng.choice(["No", "Yes"], n, p=[.9, .1]),
    })
    df["Transaction_Amount_INR"] = (df["Average_Transaction_Amount_INR"] *
                                    rng.lognormal(0, .8, n)).round(2)
    risk = ((df["Transaction_Amount_INR"] > 3 * df["Average_Transaction_Amount_INR"]) * 1.5
            + (df["Transaction_Hour"] < 5) * 1.0
            + (df["International_Transaction"] == "Yes") * 1.2
            + (df["Transaction_Amount_INR"] / df["Credit_Limit_INR"] > .25) * 1.5)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-(risk - 4)))).astype(int)
    df["Credit_Utilization"] = (df["Transaction_Amount_INR"] / df["Credit_Limit_INR"]).round(4)
    return df


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/navya project.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        source = path
    else:
        print(f"!! {path} not found - training on SYNTHETIC data (demo only).")
        df = synthetic()
        source = "synthetic"

    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough")
    pipe = Pipeline([("pre", pre),
                     ("rf", RandomForestClassifier(n_estimators=300, random_state=42,
                                                   class_weight="balanced", n_jobs=-1))])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print("Accuracy:", accuracy_score(yte, pred))
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred))

    meta = {
        "source": source,
        "features": FEATURES,
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in
                   ["Age", "Credit_Limit_INR", "Transaction_Amount_INR",
                    "Average_Transaction_Amount_INR", "Account_Age_Months"]},
        "metrics": classification_report(yte, pred, output_dict=True),
    }
    joblib.dump({"pipeline": pipe, "meta": meta}, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
