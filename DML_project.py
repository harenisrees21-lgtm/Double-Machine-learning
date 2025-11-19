"""
dml_step_by_step.py
Step-by-step, beginner-friendly implementation of the project requirements.

Run:
    python dml_step_by_step.py

Outputs saved to current folder:
 - dataset_with_cate.csv
 - cate_hist.png
 - cate_scatter.png
"""

# ---------------------------
# STEP 0: Imports & settings
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

SEED = 42
np.random.seed(SEED)

# ---------------------------
# STEP 1: Generate synthetic dataset
# ---------------------------
def generate_synthetic(N=5000, P=60, seed=SEED):
    """
    Produces:
      - DataFrame with columns x0..x{P-1}, T, Y, true_cate
      - true_cate_fn for reference (callable)
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(N, P))

    # 1. Define ground-truth CATE (nonlinear function of some x's)
    def true_cate_fn(row):
        # uses x0, x4, x5
        return 1.5 * np.tanh(row[0]) + 0.8 * (row[4] ** 2) - 0.6 * row[5]

    true_cates = np.apply_along_axis(true_cate_fn, 1, X)

    # 2. Propensity score (nonlinear in X)
    logits = 0.8 * np.sin(X[:, 0]) + 0.6 * X[:, 1] + 0.5 * (X[:, 4] ** 2) - 0.3 * X[:, 7]
    p_t = 1 / (1 + np.exp(-logits))
    T = rng.binomial(1, p_t)

    # 3. Baseline outcome
    baseline = 2.0 * np.tanh(X[:, 2]) + 0.5 * X[:, 10] + 0.3 * (X[:, 3] ** 2)
    noise = rng.normal(scale=1.0, size=N)

    # 4. Outcome Y = baseline + T * true_cate + noise
    Y = baseline + T * true_cates + noise

    # Build DataFrame
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(P)])
    df["T"] = T
    df["Y"] = Y
    df["p_t"] = p_t
    df["true_cate"] = true_cates

    return df, true_cate_fn

# ---------------------------
# STEP 2: Baseline OLS
# ---------------------------
def fit_ols(df):
    """
    Fit OLS: Y ~ T + X, return coefficient on T and full model.
    """
    X_cols = [c for c in df.columns if c.startswith("x")]
    design = sm.add_constant(pd.concat([df["T"], df[X_cols]], axis=1))
    model = sm.OLS(df["Y"], design).fit()
    coef_T = float(model.params["T"])
    return coef_T, model

# ---------------------------
# STEP 3: Manual DML (cross-fitting)
# ---------------------------
def manual_dml(df, n_splits=3, random_state=SEED):
    """
    Manual DML:
      - cross-fit nuisance models (E[Y|X] and E[T|X])
      - residualize and compute ATE
      - fit pseudo-outcome model to predict CATE
    Returns:
      - ate_estimate (float)
      - cate_pred (np.array, size N)
    """
    X_cols = [c for c in df.columns if c.startswith("x")]
    X = df[X_cols].values
    Y = df["Y"].values
    T = df["T"].values
    n = len(df)

    mu_hat = np.zeros(n)   # predictions of E[Y|X]
    pi_hat = np.zeros(n)   # predictions of P(T=1|X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold = 0
    for train_idx, test_idx in kf.split(X):
        fold += 1
        # strong learners for nuisance functions
        y_model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        t_model = RandomForestClassifier(n_estimators=200, random_state=random_state)

        y_model.fit(X[train_idx], Y[train_idx])
        t_model.fit(X[train_idx], T[train_idx])

        mu_hat[test_idx] = y_model.predict(X[test_idx])
        pi_hat[test_idx] = t_model.predict_proba(X[test_idx])[:, 1]

    # Residualize
    Y_tilde = Y - mu_hat
    T_tilde = T - pi_hat

    # ATE estimator (orthogonal)
    denom = (T_tilde ** 2).sum()
    ate_hat = float((T_tilde * Y_tilde).sum() / (denom + 1e-12))

    # Pseudo outcome for CATE: stable division
    pseudo = Y_tilde / (T_tilde + 1e-6)

    # Fit CATE model: X -> pseudo
    cate_model = RandomForestRegressor(n_estimators=300, random_state=random_state)
    cate_model.fit(X, pseudo)
    cate_pred = cate_model.predict(X)

    return ate_hat, cate_pred

# ---------------------------
# STEP 4: Run everything & save outputs
# ---------------------------
def run_pipeline():
    print("1) Generating dataset...")
    df, true_cate_fn = generate_synthetic(N=5000, P=60, seed=SEED)
    print("   Dataset shape:", df.shape)
    print("   Columns example:", df.columns[:8].tolist(), "...")

    print("\n2) Running baseline OLS...")
    ols_ate, ols_model = fit_ols(df)
    print(f"   OLS estimate of ATE (coef on T): {ols_ate:.4f}")

    print("\n3) Running manual DML (cross-fitting)... (may take a few minutes)")
    dml_ate, cate_pred = manual_dml(df, n_splits=3, random_state=SEED)
    print(f"   Manual DML estimate of ATE: {dml_ate:.4f}")

    # Add predicted CATE to dataframe for saving
    df["cate_manual"] = cate_pred

    # Evaluate CATE accuracy vs true
    mse_cate = mean_squared_error(df["true_cate"].values, cate_pred)
    print(f"\n4) CATE evaluation: MSE between predicted and true CATE = {mse_cate:.4f}")

    # Save CSV for submission / inspection
    out_csv = "dataset_with_cate.csv"
    df.to_csv(out_csv, index=False)
    print(f"   Saved dataset with CATE to '{out_csv}'")

    # ---------------------------
    # STEP 5: Plots
    # ---------------------------
    print("\n5) Creating plots...")
    # Histogram: true vs predicted CATE
    plt.figure(figsize=(8,5))
    plt.hist(df["true_cate"], bins=40, alpha=0.6, label="true_cate")
    plt.hist(df["cate_manual"], bins=40, alpha=0.5, label="cate_manual")
    plt.legend()
    plt.title("True CATE vs Predicted CATE (Manual DML)")
    plt.xlabel("CATE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("cate_hist.png", dpi=150)
    plt.close()
    print("   Saved 'cate_hist.png'")

    # Scatter: true vs predicted (sample)
    sample_idx = np.random.RandomState(SEED).choice(len(df), size=1000, replace=False)
    plt.figure(figsize=(6,6))
    plt.scatter(df["true_cate"].values[sample_idx], df["cate_manual"].values[sample_idx], alpha=0.4, s=10)
    plt.plot([df["true_cate"].min()-0.5, df["true_cate"].max()+0.5],
             [df["true_cate"].min()-0.5, df["true_cate"].max()+0.5],
             linestyle="--", color="k")
    plt.xlabel("True CATE")
    plt.ylabel("Predicted CATE (Manual)")
    plt.title("Predicted vs True CATE (sample of 1000)")
    plt.tight_layout()
    plt.savefig("cate_scatter.png", dpi=150)
    plt.close()
    print("   Saved 'cate_scatter.png'")

    # ---------------------------
    # STEP 6: Print 10 sample CATE values for report
    # ---------------------------
    print("\n6) Example 10 sample CATE values (for your report):")
    sample10 = np.random.RandomState(SEED).choice(len(df), size=10, replace=False)
    for i, idx in enumerate(sample10):
        true_v = df.loc[idx, "true_cate"]
        pred_v = df.loc[idx, "cate_manual"]
        print(f"   idx={idx:4d}  true={true_v:+0.4f}  predicted={pred_v:+0.4f}")

    # Print quick summary to paste into report
    print("\nQuick summary (copy into report):")
    print(f"   True ATE (mean true_cate) = {df['true_cate'].mean():.4f}")
    print(f"   OLS ATE = {ols_ate:.4f}  (naive - likely biased)")
    print(f"   Manual DML ATE = {dml_ate:.4f}  (preferred)")

    print("\nDONE. Files created: dataset_with_cate.csv, cate_hist.png, cate_scatter.png")

if __name__ == "__main__":
    run_pipeline()
