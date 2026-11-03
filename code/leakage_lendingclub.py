"""
Data leakage in credit data: a LendingClub example.

Trains the same XGBoost model twice on finished LendingClub loans:
  1. "Everything" model: every numeric field in the export, including fields
     recorded after the loan was issued (payments, recoveries, last FICO pull).
  2. "Honest" model: only what was known on the day the loan was issued.

Then shows a quieter leak: target-encoding a category (3-digit zip code)
using the whole dataset before splitting, versus using the training rows only.

Usage:  python leakage_lendingclub.py lendingclub_sample_25000.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SEED = 42
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "lendingclub_sample_25000.csv"

# ---------------------------------------------------------------------------
# 1. Finished loans only: repaid in full (0) or charged off (1)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)
good = df["loan_status"].isin(["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"])
bad = df["loan_status"].isin(["Charged Off", "Does not meet the credit policy. Status:Charged Off"])
df = df[good | bad].copy()
df["charged_off"] = bad[good | bad].astype(int)
TARGET = "charged_off"

issue = pd.to_datetime(df["issue_d"], format="%b-%Y")

# ---------------------------------------------------------------------------
# 2. Application-time features (same set as the backtesting article)
# ---------------------------------------------------------------------------
df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
df["emp_years"] = (df["emp_length"].str.replace("< 1", "0", regex=False)
                   .str.extract(r"(\d+)").astype(float))
earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
df["credit_history_months"] = ((issue.dt.year - earliest.dt.year) * 12
                               + (issue.dt.month - earliest.dt.month))
grade_order = {g: i for i, g in enumerate("ABCDEFG")}
df["grade_num"] = df["grade"].map(grade_order)
df["sub_grade_num"] = df["grade_num"] * 5 + df["sub_grade"].str[1].astype(float)
df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

AT_ISSUE = [
    "loan_amnt", "term_months", "int_rate", "installment", "grade_num",
    "sub_grade_num", "emp_years", "annual_inc", "loan_to_income", "dti",
    "delinq_2yrs", "fico", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "credit_history_months",
]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]

# ---------------------------------------------------------------------------
# 3. Fields recorded AFTER the loan was issued (the leaks)
# ---------------------------------------------------------------------------
last_pay = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
df["months_of_payments"] = ((last_pay.dt.year - issue.dt.year) * 12
                            + (last_pay.dt.month - issue.dt.month))
df["last_fico"] = (df["last_fico_range_low"] + df["last_fico_range_high"]) / 2
df["debt_settlement"] = (df["debt_settlement_flag"] == "Y").astype(int)
df["hardship"] = (df["hardship_flag"] == "Y").astype(int)

AFTER_ISSUE = [
    "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
    "months_of_payments", "last_fico", "debt_settlement", "hardship",
]


def make_xgb(numeric):
    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])
    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        random_state=SEED, n_jobs=4,
    )
    return Pipeline([("pre", pre), ("model", model)])


def score(y, p):
    return {"AUC": roc_auc_score(y, p),
            "Accuracy": accuracy_score(y, p >= 0.5),
            "Recall": recall_score(y, p >= 0.5)}


train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
print(f"Finished loans: {len(df):,} | charged off: {int(df[TARGET].sum()):,} "
      f"({df[TARGET].mean():.1%})")
print(f"Train {len(train):,} | Test {len(test):,}\n")

# ---------------------------------------------------------------------------
# 4. The "everything" model versus the honest model
# ---------------------------------------------------------------------------
results = {}
fitted = {}
for name, numeric in [("Everything in the export", AT_ISSUE + AFTER_ISSUE),
                      ("Known at issue only", AT_ISSUE)]:
    m = make_xgb(numeric).fit(train[numeric + CATEGORICAL], train[TARGET])
    p = m.predict_proba(test[numeric + CATEGORICAL])[:, 1]
    results[name] = score(test[TARGET], p)
    fitted[name] = (m, numeric)
print(pd.DataFrame(results).T.round(3).to_string(), "\n")

# Single-column check: how well does each after-issue field rank loans alone?
print("AUC of each after-issue field on its own (test set, direction-free):")
single = {}
for c in AFTER_ISSUE:
    x = test[c].fillna(test[c].median())
    a = roc_auc_score(test[TARGET], x)
    single[c] = max(a, 1 - a)
print(pd.Series(single).sort_values(ascending=False).round(3).to_string(), "\n")

# Feature importance (gain) for the everything model, numeric fields only
m, numeric = fitted["Everything in the export"]
gain = m.named_steps["model"].get_booster().get_score(importance_type="gain")
names = list(m.named_steps["pre"].get_feature_names_out())
imp = pd.Series({names[int(k[1:])].replace("num__", ""): v for k, v in gain.items()})
imp = (imp / imp.sum()).sort_values(ascending=False)
print("Top 10 features by share of total gain (everything model):")
print(imp.head(10).round(3).to_string(), "\n")

# ---------------------------------------------------------------------------
# 5. A quieter leak: target encoding before the split
# ---------------------------------------------------------------------------
df["zip3"] = df["zip_code"].astype(str).str[:3]
prior = train[TARGET].mean()


def target_encode(fit_rows, apply_rows, k=20):
    stats = fit_rows.groupby("zip3")[TARGET].agg(["sum", "count"])
    enc = (stats["sum"] + k * prior) / (stats["count"] + k)
    return apply_rows["zip3"].map(enc).fillna(prior).values


tr = train.assign(zip3=df.loc[train.index, "zip3"])
te = test.assign(zip3=df.loc[test.index, "zip3"])
everyone = pd.concat([tr, te])

# Leaky: encoding built from ALL rows, test outcomes included
tr_leak = tr.assign(zip_rate=target_encode(everyone, tr))
te_leak = te.assign(zip_rate=target_encode(everyone, te))
# Correct: encoding built from training rows only (out-of-fold on train)
oof = np.zeros(len(tr))
for a, b in KFold(5, shuffle=True, random_state=SEED).split(tr):
    oof[b] = target_encode(tr.iloc[a], tr.iloc[b])
tr_ok = tr.assign(zip_rate=oof)
te_ok = te.assign(zip_rate=target_encode(tr, te))

zip_results = {}
for name, a, b in [("Zip rate built from all rows", tr_leak, te_leak),
                   ("Zip rate built from training rows", tr_ok, te_ok)]:
    num = AT_ISSUE + ["zip_rate"]
    mm = make_xgb(num).fit(a[num + CATEGORICAL], a[TARGET])
    zip_results[name] = score(b[TARGET], mm.predict_proba(b[num + CATEGORICAL])[:, 1])
print(pd.DataFrame(zip_results).T.round(3).to_string())

# Save for charts
pd.DataFrame(results).T.to_csv("results_main.csv")
imp.to_csv("importance.csv", header=["share"])
pd.Series(single).to_csv("single_field_auc.csv", header=["auc"])
pd.DataFrame(zip_results).T.to_csv("results_zip.csv")
