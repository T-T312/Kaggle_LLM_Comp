"""
LLM Classification Finetuning (Kaggle) - stage 1 baseline.

Predicts P(A wins), P(B wins), P(tie) for a prompt + two LLM responses using
cheap hand-crafted text features + TF-IDF similarities and a LightGBM model.

Pipeline: load + parse -> EDA -> features -> 5-fold CV (with A/B swap augmentation
and swap TTA) -> retrain on full train -> predict test -> validate + write submission.csv.

Kept as a single file on purpose so it can be pasted into one Kaggle notebook cell
or pushed as a Kaggle script kernel without uploading extra modules.

Run locally:   python project/project.py            (full run)
               python project/project.py --debug    (small sample, quick)
"""
import glob
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
except ImportError:  # fall back to LogisticRegression (see make_model)
    lgb = None


# =============================================================================
# Config
# =============================================================================
ON_KAGGLE = os.path.exists("/kaggle/input")

if ON_KAGGLE:
    DATA_DIR = "/kaggle/input/llm-classification-finetuning"
    if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
        # Mount paths have changed over time; find the competition data wherever it is.
        hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
        if hits:
            DATA_DIR = os.path.dirname(hits[0])
    OUTPUT_DIR = "/kaggle/working"
else:
    # Project root = parent of project/. Use cwd if __file__ is missing (e.g. pasted into a notebook).
    try:
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        ROOT = os.getcwd()
    DATA_DIR = os.path.join(ROOT, "data")
    OUTPUT_DIR = os.path.join(ROOT, "model")

# Turn on with `--debug` or env DEBUG=1, or just set it to True here (e.g. in a notebook).
DEBUG = "--debug" in sys.argv or os.environ.get("DEBUG", "0") == "1"
DEBUG_ROWS = 5000

SEED = 42
N_FOLDS = 5
USE_TFIDF = True    # TF-IDF cosine similarities + SVD of (A - B); set False for a faster run
SVD_DIM = 32

TEXT_COLS = ["prompt", "response_a", "response_b"]
TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]
SWAP = [1, 0, 2]    # column/label permutation for swapping A <-> B (tie stays tie)


# =============================================================================
# Data loading / parsing
# =============================================================================
def parse_turns(cell):
    """Decode a JSON-list cell into a list of turns (str or None). Returns None if it can't be parsed."""
    if not isinstance(cell, str):
        return None
    try:
        turns = json.loads(cell)
    except ValueError:
        return None
    if not isinstance(turns, list):
        turns = [turns]
    return [t if t is None or isinstance(t, str) else str(t) for t in turns]


def load_data(name):
    """Read a CSV and replace each text column with plain text (turns joined by newlines).

    Adds <col>_bad (cell failed to parse; raw text kept), <col>_nulls (number of null turns)
    and, when the winner columns exist, `label`: 0 = A wins, 1 = B wins, 2 = tie.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, name))
    for col in TEXT_COLS:
        raw = df[col]
        turns = raw.map(parse_turns)
        df[f"{col}_bad"] = turns.isna().astype(int)
        df[f"{col}_nulls"] = turns.map(lambda t: 0 if t is None else sum(x is None for x in t))
        df[col] = [
            "\n".join(x for x in t if x is not None) if t is not None
            else (r if isinstance(r, str) else "")
            for t, r in zip(turns, raw)
        ]
    if set(TARGETS).issubset(df.columns):
        df["label"] = df[TARGETS].to_numpy().argmax(axis=1)
    return df


# =============================================================================
# EDA
# =============================================================================
def eda(df):
    print("\n=== EDA ===")
    print(f"rows: {len(df)}")

    names = {0: "A wins", 1: "B wins", 2: "tie"}
    balance = df["label"].map(names).value_counts(normalize=True)
    print("class balance:\n" + balance.round(4).to_string())

    lens = pd.DataFrame({
        "len_a": df["response_a"].str.len(),
        "len_b": df["response_b"].str.len(),
        "label": df["label"].map(names),
    })
    print("mean response length (chars) per class:\n" + lens.groupby("label").mean().round(0).to_string())

    decided = (df["label"] != 2) & (lens["len_a"] != lens["len_b"])
    longer_wins = ((lens["len_a"] > lens["len_b"]) == (df["label"] == 0))[decided].mean()
    print(f"longer response wins (non-tie, unequal length): {longer_wins:.4f}")

    for col in ["response_a", "response_b"]:
        print(f"{col}: unparseable={df[col + '_bad'].sum()}, "
              f"rows with null turns={(df[col + '_nulls'] > 0).sum()}, "
              f"empty text={(df[col].str.strip() == '').sum()}")


# =============================================================================
# Features
# =============================================================================
def text_stats(s):
    """Cheap per-text statistics (one row per text)."""
    return pd.DataFrame({
        "chars": s.str.len(),
        "words": s.str.count(r"\S+"),
        "newlines": s.str.count("\n"),
        "list_items": s.str.count(r"(?m)^\s*(?:[-*+]|\d+[.)])\s"),
        "headers": s.str.count(r"(?m)^\s*#{1,6}\s"),
        "code_blocks": s.str.count("```") // 2,
        "bold": s.str.count(r"\*\*[^*\n]+\*\*"),
    }).reset_index(drop=True)


def fit_tfidf(df):
    """Fit TF-IDF on all train texts, and an SVD on the (A - B) TF-IDF difference."""
    vec = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True, dtype=np.float32)
    vec.fit(pd.concat([df[c] for c in TEXT_COLS]))
    diff = vec.transform(df["response_a"]) - vec.transform(df["response_b"])
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED).fit(diff)
    return vec, svd


def build_parts(df, tfidf=None):
    """Compute feature blocks: prompt stats, per-response stats (a, b), order-free pair stats,
    and an antisymmetric block (flips sign on A/B swap). Kept separate so we can swap A and B."""
    a = text_stats(df["response_a"])
    b = text_stats(df["response_b"])
    a["nulls"] = df["response_a_nulls"].to_numpy()
    b["nulls"] = df["response_b_nulls"].to_numpy()
    parts = {"p": text_stats(df["prompt"]), "a": a, "b": b,
             "sym": pd.DataFrame(index=a.index), "anti": None}

    if tfidf is not None:
        vec, svd = tfidf
        tp, ta, tb = (vec.transform(df[c]) for c in TEXT_COLS)
        # Rows are L2-normalized, so the row-wise dot product is the cosine similarity.
        a["sim_prompt"] = np.asarray(tp.multiply(ta).sum(axis=1)).ravel()
        b["sim_prompt"] = np.asarray(tp.multiply(tb).sum(axis=1)).ravel()
        parts["sym"]["sim_ab"] = np.asarray(ta.multiply(tb).sum(axis=1)).ravel()
        parts["anti"] = svd.transform(ta - tb)  # linear, so swap(A, B) == -anti
    return parts


def assemble(parts, swap=False):
    """Turn feature blocks into the model matrix; swap=True builds it as if A and B were exchanged."""
    a, b = (parts["b"], parts["a"]) if swap else (parts["a"], parts["b"])
    blocks = [
        parts["p"].add_prefix("p_"),
        a.add_prefix("a_"),
        b.add_prefix("b_"),
        (a - b).add_prefix("diff_"),
        (np.log1p(a) - np.log1p(b)).add_prefix("logratio_"),
        parts["sym"],
    ]
    if parts["anti"] is not None:
        anti = -parts["anti"] if swap else parts["anti"]
        blocks.append(pd.DataFrame(anti, columns=[f"svd_{i}" for i in range(anti.shape[1])]))
    return pd.concat(blocks, axis=1)


# =============================================================================
# Model
# =============================================================================
def make_model(n_estimators=3000):
    if lgb is not None:
        return lgb.LGBMClassifier(
            objective="multiclass", n_estimators=n_estimators, learning_rate=0.03,
            num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1,
        )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def augment(X, Xs, y):
    """Swap augmentation: original rows + A/B-swapped copies with labels 0 <-> 1."""
    return pd.concat([X, Xs], ignore_index=True), np.concatenate([y, np.array(SWAP)[y]])


def predict_tta(model, X, Xs):
    """Average predictions on the original and the swapped rows (swapped A/B probs mapped back)."""
    return (model.predict_proba(X) + model.predict_proba(Xs)[:, SWAP]) / 2


def cross_validate(X, Xs, y):
    """Stratified K-fold CV. Both copies of a row stay in the same fold (augmentation happens per fold).
    Returns the number of boosting rounds to use for the final model."""
    print(f"\n=== {N_FOLDS}-fold CV ({'LightGBM' if lgb else 'LogisticRegression'}) ===")
    oof_plain = np.zeros((len(y), 3))
    oof_tta = np.zeros((len(y), 3))
    best_iters = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, y_tr = augment(X.iloc[tr], Xs.iloc[tr], y[tr])
        model = make_model()
        if lgb is not None:
            # Early stopping on the validation fold makes CV slightly optimistic; fine for a baseline.
            # lightgbm >= 4.7 renamed eval_set to eval_X/eval_y; Kaggle may still have an older 4.x.
            if "eval_X" in inspect.signature(model.fit).parameters:
                eval_kw = {"eval_X": X.iloc[va], "eval_y": y[va]}
            else:
                eval_kw = {"eval_set": [(X.iloc[va], y[va])]}
            model.fit(X_tr, y_tr, callbacks=[lgb.early_stopping(100, verbose=False)], **eval_kw)
            best_iters.append(model.best_iteration_)
        else:
            model.fit(X_tr, y_tr)
        oof_plain[va] = model.predict_proba(X.iloc[va])
        oof_tta[va] = predict_tta(model, X.iloc[va], Xs.iloc[va])
        print(f"fold {fold}: log loss {log_loss(y[va], oof_tta[va], labels=[0, 1, 2]):.4f}"
              + (f"  (best iter {best_iters[-1]})" if best_iters else ""))

    prior = np.bincount(y, minlength=3) / len(y)
    print(f"uniform 1/3 baseline log loss : {log_loss(y, np.full((len(y), 3), 1 / 3), labels=[0, 1, 2]):.4f}")
    print(f"class-prior baseline log loss : {log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2]):.4f}")
    print(f"CV log loss (no TTA)          : {log_loss(y, oof_plain, labels=[0, 1, 2]):.4f}")
    print(f"CV log loss (swap TTA)        : {log_loss(y, oof_tta, labels=[0, 1, 2]):.4f}")
    # The full model trains on 1/(1 - 1/K) times more data, so give it a few more rounds.
    return int(np.mean(best_iters) * 1.1) if best_iters else None


# =============================================================================
# Submission
# =============================================================================
def make_submission(test_ids, probs):
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    sub = pd.DataFrame(probs, columns=TARGETS)
    sub.insert(0, "id", np.asarray(test_ids))
    return sub


def validate_submission(sub, test):
    cols = ["id"] + TARGETS
    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
    if os.path.exists(sample_path):
        cols = list(pd.read_csv(sample_path, nrows=0).columns)
    assert list(sub.columns) == cols, f"bad columns: {list(sub.columns)} != {cols}"
    assert len(sub) == len(test), f"row count {len(sub)} != {len(test)}"
    assert (sub["id"].to_numpy() == test["id"].to_numpy()).all(), "ids don't match test.csv"
    assert not sub.isna().any().any(), "NaNs in submission"
    p = sub[TARGETS].to_numpy()
    assert ((p >= 0) & (p <= 1)).all(), "probabilities out of [0, 1]"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), "rows don't sum to 1"
    print(f"submission checks passed ({len(sub)} rows)")


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = time.time()
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"ON_KAGGLE={ON_KAGGLE} DEBUG={DEBUG} DATA_DIR={DATA_DIR} OUTPUT_DIR={OUTPUT_DIR}")

    train = load_data("train.csv")
    test = load_data("test.csv")
    if DEBUG:
        train = train.sample(n=min(DEBUG_ROWS, len(train)), random_state=SEED).reset_index(drop=True)
    print(f"train {train.shape}, test {test.shape}")

    eda(train)

    print("\nbuilding features...")
    tfidf = fit_tfidf(train) if USE_TFIDF else None
    train_parts = build_parts(train, tfidf)
    test_parts = build_parts(test, tfidf)
    X, Xs = assemble(train_parts), assemble(train_parts, swap=True)
    X_test, Xs_test = assemble(test_parts), assemble(test_parts, swap=True)
    y = train["label"].to_numpy()
    print(f"{X.shape[1]} features ({time.time() - t0:.0f}s)")

    n_rounds = cross_validate(X, Xs, y)

    print("\n=== full fit ===")
    X_all, y_all = augment(X, Xs, y)
    model = make_model(n_rounds) if n_rounds else make_model()
    model.fit(X_all, y_all)
    if lgb is not None:
        model.booster_.save_model(os.path.join(OUTPUT_DIR, "lgbm_baseline.txt"))

    sub = make_submission(test["id"], predict_tta(model, X_test, Xs_test))
    validate_submission(sub, test)
    path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(path, index=False)
    print(f"wrote {path}")
    print(sub.head().to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
