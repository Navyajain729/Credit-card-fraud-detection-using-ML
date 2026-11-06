# Fraud Check app

Flask backend + web UI around your Random Forest fraud model.

## Run
```bash
pip install flask scikit-learn pandas numpy joblib
mkdir data && cp "navya project.csv" data/      # your real dataset
python train.py                                  # trains + writes model.joblib
python app.py                                    # home page at http://localhost:5000, checker at /check
```
Without the CSV, `train.py` trains on synthetic data so you can demo the app.

## API
`POST /api/predict` (JSON) with: Transaction_Amount_INR, Credit_Limit_INR,
Average_Transaction_Amount_INR, Transaction_Hour, Previous_Transactions_24h, Age,
Account_Age_Months, plus Merchant_Category, Transaction_Type, Location, Device_Type,
International_Transaction ("Yes"/"No").
Credit_Utilization is computed server-side (amount / limit, matches your data).

`GET /api/meta` returns the allowed category values.

## Changes from the notebook
- Separate encoder per column (notebook reused one LabelEncoder object).
- Person_ID removed from features.
- `class_weight="balanced"`: your recall on fraud was 0.56 (14 of 32 missed).
