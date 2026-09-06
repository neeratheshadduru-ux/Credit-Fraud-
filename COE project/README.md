# Credit Card Fraud Detection — Streamlit App

Converts the `Credit_Card_Fraud_dataset.ipynb` notebook into an interactive app with four tabs:

1. **Data Overview** — shape, fraud rate, sample rows, missing values
2. **EDA** — class balance, per-column distributions, correlation heatmap
3. **Model Evaluation** — Logistic Regression (`class_weight="balanced"`), classification report,
   confusion matrix, ROC curve, feature coefficients, with an adjustable decision threshold
4. **Predict a Transaction** — fill in a form describing one transaction and get a live fraud
   probability from the trained model

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Files

- `app.py` — the app
- `requirements.txt` — Python dependencies
- `credit_card_fraud_2026.csv` — bundled sample dataset (20,000 transactions, ~1.7% fraud).
  You can also upload your own CSV from the sidebar, as long as it has the same columns
  (`transaction_id`, `is_fraud`, and the same feature columns).

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (include `app.py`, `requirements.txt`, and the CSV).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick the repo/branch, and set the main file to `app.py`.
4. Click **Deploy** — it builds automatically from `requirements.txt`.

## Notes

- The model is retrained once per session and cached (`st.cache_resource`), so switching tabs
  or moving sliders doesn't retrain it — only uploading a new CSV does.
- The "Predict a Transaction" tab runs the new row through the exact same one-hot encoding +
  `StandardScaler` pipeline used during training, so results are consistent with the
  Model Evaluation tab.
