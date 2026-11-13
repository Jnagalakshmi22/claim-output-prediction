# Claim Outcome Prediction

Small end-to-end demo that synthesizes historical insurance claims, trains a classifier for `Paid`, `Partially_Paid`, and `Denied`, evaluates it on a stratified holdout, and serves predictions through FastAPI. The 500-row dataset is synthetic and for demonstration only; it is not suitable for real claims decisions.

## Run

```powershell
python -m pip install -r requirements.txt
python train.py
python -m uvicorn app:app --reload
```

Training writes `synthetic_claims.csv` and `claim_pipeline.joblib` in the project directory and prints per-class precision, recall, F1, support, and a confusion matrix. Open `http://localhost:8000/docs` to try the API. Run `python -m pytest -q` to execute tests. Train the model before starting the API; if training runs while the API is already running, the next health check or prediction automatically reloads the updated model artifact.

## Data preparation and model

The generator creates 500 rows with outcomes that are approximately 70% Paid, 20% Partially Paid, and 10% Denied. Labels are sampled probabilistically: payer, procedure, billed amount, provider, age, and diagnosis influence the chance of an outcome, but identical claim details can still have different outcomes. The generator's latent risk calculation is used only to create labels and is not included as a model feature. This provides a learnable but imperfect pattern rather than assigning outcomes deterministically from the model's own input score; it remains a simulation assumption, not a claim about real insurance behavior. An 80/20 split uses `stratify=y`. Payer/provider denial rates and payer mean billed amount are learned from training rows only. Training rows use leave-one-out aggregates so each row's own label does not enter its features; test and API requests use aggregates fitted over the full training split. Unseen categories use training-set global values. Engineered features include billed amount per patient age and payer-relative amount deviation.

Categorical fields are one-hot encoded, numeric fields standardized, and a regularized logistic regression is fitted. Its probabilities are not forced to be exactly 0 or 1 for familiar claims. The complete preprocessing/model pipeline is serialized as one joblib artifact.

## Holdout results

Run `python train.py` to print the reproducible 80/20 stratified holdout report (100 test claims). The synthetic data is probabilistic, so performance is expected to be imperfect.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Paid | 0.86 | 0.97 | 0.91 | 69 |
| Partially_Paid | 0.54 | 0.33 | 0.41 | 21 |
| Denied | 0.33 | 0.30 | 0.32 | 10 |

Confusion matrix (rows are actual; columns are predicted, in Paid / Partially_Paid / Denied order):

```text
[[67,  0, 2],
 [10,  7, 4],
 [ 1,  6, 3]]
```

Overall holdout accuracy was 0.77. Denied is weakest by F1 (0.32); it identified 3 of 10 denied claims. The minority-class results are unstable with only 10 denied examples. A next experiment would be to add more synthetic claims and compare results across multiple stratified splits.

## API

`POST /predict` accepts payer, provider, cpt_code, diagnosis_code, billed_amount, and patient_age. It returns `predicted_status` and a probability for each class. Missing fields, wrong types, blank strings, nonpositive amounts, and ages outside 0–125 return HTTP 400. `GET /health` reports whether the model loaded.

## Assumptions and limitations

Synthetic labels are generated from a probabilistic risk model and do not represent real payer behavior. Test metrics measure whether the pipeline can learn that synthetic pattern; they do not establish real-world claim prediction quality. The dataset has no patient identifiers, claim history, authorization evidence, or adjudication details, so it should not be used to approve or deny claims.
