"""
Backtesting a credit default model on LendingClub data
Jim Mott

Question: a model that looks good on a random train/test split, does it
still look good when it has to predict loans from a year it has never seen?

What this script does
1. Builds a 12-month default flag: the borrower stopped paying within the
   first 12 months after issue and the loan ended up charged off or seriously
   late. LendingClub does not record an exact default date, so the last
   payment date stands in for it.
2. Keeps only loans that had at least 12 months (plus a few months for the
   delinquency to show) before the data snapshot.
3. Uses only fields known at the time of application.
4. Scores XGBoost and logistic regression two ways:
   a. a random 80/20 split (the usual way)
   b. a year-by-year backtest. To predict loans issued in year Y, the model
      trains only on loans issued through year Y-2, because at the end of
      year Y-1 those are the only loans whose 12-month outcome is known.
5. Reports AUC and predicted vs. actual default rate for each test year.

Run in the Risk environment:
    python backtest_lendingclub.py lendingclub_sample_25000.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "lendingclub_sample_25000.csv"
LAST_ELIGIBLE_ISSUE = "2017-12-01"   # snapshot is March 2019, so 15+ months of history
TEST_YEARS = [2014, 2015, 2016, 2017]

# ---------------------------------------------------------------------------
# 1. Load and build the 12-month default flag
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False).copy()

issue = pd.to_datetime(df["issue_d"], format="%b-%Y")
last_pay = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y")
months_paying = ((last_pay.dt.year - issue.dt.year) * 12
                 + (last_pay.dt.month - issue.dt.month)).fillna(0)  # no payment = 0

bad_status = df["loan_status"].isin([
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
])
df["default_12m"] = (bad_status & (months_paying <= 12)).astype(int)
df["issue_year"] = issue.dt.year

df = df[issue <= LAST_ELIGIBLE_ISSUE].copy()

# ---------------------------------------------------------------------------
# 2. Application-time features only (nothing recorded after the loan was issued)
# ---------------------------------------------------------------------------
df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
df["emp_years"] = (df["emp_length"].str.replace("< 1", "0", regex=False)
                   .str.extract(r"(\d+)").astype(float))
earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
df["credit_history_months"] = ((issue.loc[df.index].dt.year - earliest.dt.year) * 12
                               + (issue.loc[df.index].dt.month - earliest.dt.month))
grade_order = {g: i for i, g in enumerate("ABCDEFG")}
df["grade_num"] = df["grade"].map(grade_order)
df["sub_grade_num"] = df["grade_num"] * 5 + df["sub_grade"].str[1].astype(float)
df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

NUMERIC = [
    "loan_amnt", "term_months", "int_rate", "installment", "grade_num",
    "sub_grade_num", "emp_years", "annual_inc", "loan_to_income", "dti",
    "delinq_2yrs", "fico", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "credit_history_months",
]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "default_12m"


def make_logit():
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])


def make_xgb():
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])
    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        random_state=SEED, n_jobs=4,
    )
    return Pipeline([("pre", pre), ("model", model)])


MODELS = {"XGBoost": make_xgb, "Logistic regression": make_logit}

# ---------------------------------------------------------------------------
# 3. The usual way: random 80/20 split
# ---------------------------------------------------------------------------
print("Eligible loans:", len(df), "| 12-month defaults:", int(df[TARGET].sum()))
print("\nDefault rate by issue year")
print(df.groupby("issue_year")[TARGET].agg(loans="size", defaults="sum", rate="mean")
        .round(3).to_string())

train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
random_rows = []
for name, make in MODELS.items():
    m = make().fit(train[FEATURES], train[TARGET])
    p = m.predict_proba(test[FEATURES])[:, 1]
    random_rows.append({"model": name, "auc": roc_auc_score(test[TARGET], p),
                        "predicted_rate": p.mean(), "actual_rate": test[TARGET].mean()})
random_split = pd.DataFrame(random_rows)
print("\nRandom 80/20 split")
print(random_split.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. The backtest: train through Y-2, predict year Y
# ---------------------------------------------------------------------------
rows = []
for year in TEST_YEARS:
    tr = df[df["issue_year"] <= year - 2]
    te = df[df["issue_year"] == year]
    for name, make in MODELS.items():
        m = make().fit(tr[FEATURES], tr[TARGET])
        p = m.predict_proba(te[FEATURES])[:, 1]
        rows.append({
            "test_year": year, "model": name,
            "train_loans": len(tr), "train_defaults": int(tr[TARGET].sum()),
            "test_loans": len(te), "test_defaults": int(te[TARGET].sum()),
            "auc": roc_auc_score(te[TARGET], p),
            "predicted_rate": p.mean(), "actual_rate": te[TARGET].mean(),
        })
backtest = pd.DataFrame(rows)
print("\nYear-by-year backtest")
print(backtest.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Is the lower backtest AUC about time, or just smaller training sets?
#    Rerun the random split with training sets cut to the backtest sizes.
# ---------------------------------------------------------------------------
size_rows = []
for year in TEST_YEARS:
    n = int((df["issue_year"] <= year - 2).sum())
    aucs = []
    for s in range(5):
        tr, te = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=s)
        tr = tr.sample(n, random_state=s)
        m = make_xgb().fit(tr[FEATURES], tr[TARGET])
        aucs.append(roc_auc_score(te[TARGET], m.predict_proba(te[FEATURES])[:, 1]))
    size_rows.append({"test_year": year, "train_loans": n,
                      "random_split_auc_same_size": np.mean(aucs)})
size_check = pd.DataFrame(size_rows).merge(
    backtest[backtest["model"] == "XGBoost"][["test_year", "auc"]]
    .rename(columns={"auc": "backtest_auc"}), on="test_year")
print("\nXGBoost: random split with the same training size vs. backtest")
print(size_check.round(3).to_string(index=False))

random_split.to_csv("random_split_results.csv", index=False)
backtest.to_csv("backtest_results.csv", index=False)
print("\nSaved random_split_results.csv and backtest_results.csv")
