"""
BSVAR — Bayesian Structural VAR
=================================
Sections:
  1. Data simulation
  2. Lag selection (AIC/BIC)
  3. Reduced-form VAR estimation
  4. BVAR with Minnesota prior (Gibbs sampler)
  5. Structural identification (sign restrictions + block exogeneity)
  6. IRF computation
  7. FEVD computation
  8. Historical decomposition
  9. Output — plots + numeric arrays
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import invwishart
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# OUTPUT & DATA DIRECTORIES
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "Final_Data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

def out(filename):
    """Retourne le chemin complet vers le dossier de sortie."""
    return os.path.join(OUTPUT_DIR, filename)

np.random.seed(42)

# ============================================================
# SECTION 1 — REAL DATA LOADING
# ============================================================
def load_real_data():
    csv_path = os.path.join(DATA_DIR, "df_final.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier {csv_path} est introuvable.")
    
    df_raw = pd.read_csv(csv_path)
    
    # Mapping des colonnes vers le modèle
    # Variable order: [climate, inflation, rate, output, fx]
    mapping = {
        "RAIN_log": "climate",
        "IPC_log_sa": "inflation",
        "taux_directeur": "rate",
        "IPI_log_sa": "output",
        "TCER_log": "fx"
    }
    
    # Sélection et renommage
    df = df_raw[list(mapping.keys())].rename(columns=mapping)
    
    # STATIONNARISATION
    # Pour le taux directeur, on prend souvent la différence première (variation du taux)
    # pour être cohérent avec les autres variables en log-diff.
    df_diff = df.diff().dropna()
    
    print(f"Données chargées : {len(df_diff)} observations après différenciation.")
    return df_diff

df_model = load_real_data()
var_names = list(df_model.columns)
Y_raw = df_model.values
T, N = Y_raw.shape
P = 2 # Sera ajusté par BIC plus loin


# ============================================================
# SECTION 2 — LAG SELECTION (AIC / BIC)
# ============================================================
def build_lag_matrix(Y, p):
    T, N = Y.shape
    T_eff = T - p
    X = np.ones((T_eff, 1))
    for lag in range(1, p + 1):
        X = np.hstack([X, Y[p - lag: T - lag, :]])
    Y_dep = Y[p:, :]
    return Y_dep, X

def ols_var(Y, p):
    Y_dep, X = build_lag_matrix(Y, p)
    T_eff, k = X.shape
    N = Y_dep.shape[1]
    B = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    U = Y_dep - X @ B
    Sigma = U.T @ U / T_eff
    log_det = np.log(np.linalg.det(Sigma))
    npar = N * k
    aic = log_det + 2 * npar / T_eff
    bic = log_det + np.log(T_eff) * npar / T_eff
    return B, Sigma, U, aic, bic

print("\n--- Lag selection ---")
ic_results = []
for p_test in range(1, 6):
    _, _, _, aic, bic = ols_var(Y_raw, p_test)
    ic_results.append({"p": p_test, "AIC": round(aic, 4), "BIC": round(bic, 4)})
ic_df = pd.DataFrame(ic_results)
print(ic_df.to_string(index=False))

best_p = ic_df.loc[ic_df["BIC"].idxmin(), "p"]
print(f"Selected lag p = {best_p} (BIC)")
P = int(best_p)


# ============================================================
# SECTION 3 — REDUCED-FORM VAR (OLS)
# ============================================================
B_ols, Sigma_ols, U_ols, _, _ = ols_var(Y_raw, P)
Y_dep, X_mat = build_lag_matrix(Y_raw, P)
T_eff = Y_dep.shape[0]
k = X_mat.shape[1]

print("\n--- OLS VAR coefficients (B) ---")
print(pd.DataFrame(B_ols, columns=var_names).round(4))
print("\n--- OLS Sigma ---")
print(pd.DataFrame(Sigma_ols, index=var_names, columns=var_names).round(4))


# ============================================================
# SECTION 4 — BVAR (Minnesota Prior + Gibbs Sampler)
# ============================================================

def build_minnesota_prior(N, P, k, lambda1=0.2, lambda2=0.5, lambda3=1.0):
    """
    Minnesota prior on vec(B).
    Returns:
        B_prior : (k, N) prior mean
        V_prior : (k, k) diagonal prior variance (per equation)
    """
    B_prior = np.zeros((k, N))
    # AR(1) on own first lag → 1
    for i in range(N):
        B_prior[1 + i, i] = 1.0   # row 1+i = lag-1 of variable i

    # Diagonal variances
    diag_V = np.zeros(k)
    diag_V[0] = 1e6   # constant — diffuse

    sigma2 = np.diag(Sigma_ols)   # residual variances from OLS

    for lag in range(1, P + 1):
        for j in range(N):
            col_idx = 1 + (lag - 1) * N + j
            if lag == 1:
                diag_V[col_idx] = (lambda1 ** 2) / (lag ** 2)
            else:
                diag_V[col_idx] = (lambda1 * lambda2) ** 2 / (lag ** 2) * (sigma2[j] / sigma2[j])

    return B_prior, np.diag(diag_V)


def gibbs_bvar(Y_dep, X, N, P, k, n_draw=3000, n_burn=1000):
    """
    Gibbs sampler for BVAR with Normal-Wishart posterior.
    Returns posterior draws of (B, Sigma).
    """
    B_prior, V_prior = build_minnesota_prior(N, P, k)

    # Storage
    B_draws   = np.zeros((n_draw, k, N))
    Sig_draws = np.zeros((n_draw, N, N))

    # Initialise
    B_cur   = B_ols.copy()
    Sig_cur = Sigma_ols.copy()

    T_eff = Y_dep.shape[0]
    V_prior_inv = np.diag(1.0 / np.diag(V_prior))

    for draw in range(n_draw + n_burn):
        # --- Draw B | Sigma, Y ---
        Sig_inv = np.linalg.inv(Sig_cur)
        for eq in range(N):
            sig_ii_inv = Sig_inv[eq, eq]
            V_post_inv = V_prior_inv + sig_ii_inv * (X.T @ X)
            V_post     = np.linalg.inv(V_post_inv)
            b_post     = V_post @ (V_prior_inv @ B_prior[:, eq]
                                   + sig_ii_inv * X.T @ Y_dep[:, eq])
            B_cur[:, eq] = np.random.multivariate_normal(b_post, V_post)

        # --- Draw Sigma | B, Y ---
        U_cur = Y_dep - X @ B_cur
        S_post = U_cur.T @ U_cur + np.eye(N)   # IW scale (df=T_eff+N+1)
        df_post = T_eff + N + 1
        Sig_cur = invwishart.rvs(df=df_post, scale=S_post)

        if draw >= n_burn:
            idx = draw - n_burn
            B_draws[idx]   = B_cur
            Sig_draws[idx] = Sig_cur

        if (draw + 1) % 500 == 0:
            print(f"  Gibbs draw {draw+1}/{n_draw + n_burn}")

    return B_draws, Sig_draws


print("\n--- Gibbs Sampling (BVAR) ---")
N_DRAW = 2000
N_BURN = 500
B_draws, Sig_draws = gibbs_bvar(Y_dep, X_mat, N, P, k,
                                 n_draw=N_DRAW, n_burn=N_BURN)
print(f"Posterior draws: B {B_draws.shape}, Sigma {Sig_draws.shape}")

# Posterior summaries
B_post_mean = B_draws.mean(axis=0)
B_post_std  = B_draws.std(axis=0)
Sig_post_mean = Sig_draws.mean(axis=0)

print("\n--- Posterior mean B ---")
print(pd.DataFrame(B_post_mean, columns=var_names).round(4))
print("\n--- Posterior mean Sigma ---")
print(pd.DataFrame(Sig_post_mean, index=var_names, columns=var_names).round(4))


# ============================================================
# SECTION 5 — STRUCTURAL IDENTIFICATION
# ============================================================
# 5a. Sign restrictions via random rotation (Uhlig 2005)
# 5b. Block exogeneity: climate does not respond contemporaneously

# Sign restriction matrix:
# Rows = variables [climate, inflation, rate, output, fx]
# Cols = shocks    [climate, demand,   monetary, supply, fx_shock]
# +1 = positive, -1 = negative, 0 = unrestricted

SIGN_MATRIX = np.array([
    # clim  demand  monetary  supply  fx
    [  1,    0,       0,       0,      0 ],   # climate
    [  1,    1,      -1,       0,      0 ],   # inflation
    [  0,    0,       1,       0,      0 ],   # rate
    [ -1,    1,       0,       0,      0 ],   # output
    [  0,    0,       0,       0,      1 ],   # fx
], dtype=float)

def check_sign_restrictions(A0, sign_mat):
    """
    A0: (N, N) impact matrix — columns are structural shocks
    Returns True if all non-zero sign restrictions are satisfied.
    """
    for shock_idx in range(A0.shape[1]):
        for var_idx in range(A0.shape[0]):
            s = sign_mat[var_idx, shock_idx]
            if s == 0:
                continue
            if s * A0[var_idx, shock_idx] <= 0:
                return False
    return True


def block_exogeneity_constraint(A0, n_exog=1):
    """
    Enforce block exogeneity: first n_exog variables do not react to
    shocks of the endogenous block contemporaneously.
    Sets A0[0:n_exog, n_exog:] = 0 and renormalises.
    """
    A0_constrained = A0.copy()
    A0_constrained[:n_exog, n_exog:] = 0.0
    return A0_constrained


def random_rotation_identification(Sigma, sign_mat, n_exog=1,
                                   max_draws=50000, max_valid=500,
                                   seed=0):
    """
    Draw random orthogonal matrices Q, form A0 = chol(Sigma) @ Q,
    apply block constraint, check sign restrictions.
    Returns list of valid A0 matrices.
    """
    rng_loc = np.random.default_rng(seed)
    N = Sigma.shape[0]
    P_chol = np.linalg.cholesky(Sigma)   # lower triangular
    valid_A0s = []

    for _ in range(max_draws):
        if len(valid_A0s) >= max_valid:
            break
        # Random orthogonal matrix via QR of random normal matrix
        M = rng_loc.standard_normal((N, N))
        Q, _ = np.linalg.qr(M)
        # Normalise columns to have positive diagonal
        Q = Q * np.sign(np.diag(Q))

        A0 = P_chol @ Q   # impact matrix candidate

        # Apply block exogeneity
        A0 = block_exogeneity_constraint(A0, n_exog)

        # Check sign restrictions
        if check_sign_restrictions(A0, sign_mat):
            valid_A0s.append(A0.copy())

    return valid_A0s


print("\n--- Structural identification ---")
# Use posterior mean Sigma for identification
valid_A0_list = random_rotation_identification(
    Sig_post_mean, SIGN_MATRIX, n_exog=1,
    max_draws=100_000, max_valid=500, seed=123
)
print(f"Valid A0 draws: {len(valid_A0_list)} / 100000")

if len(valid_A0_list) == 0:
    # Fallback: use Cholesky (recursive) identification
    print("  WARNING: No valid rotations found. Using Cholesky fallback.")
    A0_mean = np.linalg.cholesky(Sig_post_mean)
    valid_A0_list = [A0_mean]

A0_arr = np.array(valid_A0_list)   # (n_valid, N, N)
print(f"A0 array shape: {A0_arr.shape}")
print("\n--- Mean A0 (impact matrix) ---")
print(pd.DataFrame(A0_arr.mean(axis=0),
                   index=var_names, columns=[f"s_{v}" for v in var_names]).round(4))


# ============================================================
# SECTION 6 — IMPULSE RESPONSE FUNCTIONS (IRFs)
# ============================================================
SHOCK_NAMES = ["climate", "demand", "monetary", "supply", "fx"]
H = 24   # horizon

def compute_var_companion(B, N, P):
    """Companion matrix from B (k×N) with constant in row 0."""
    B_coef = B[1:, :]   # drop constant → (N*P, N)
    comp = np.zeros((N * P, N * P))
    comp[:N, :] = B_coef.T
    if P > 1:
        comp[N:, :N*(P-1)] = np.eye(N * (P - 1))
    return comp

def compute_irf_draws(B_draws, A0_list, N, P, H):
    """
    For each (B, A0) draw, compute IRFs.
    Returns array (n_draws, H+1, N, N_shocks).
    """
    n_valid = len(A0_list)
    n_B     = len(B_draws)
    # Sample pairs uniformly
    n_draws = min(n_valid, n_B, 500)
    idx_B   = np.random.choice(n_B,     n_draws, replace=False)
    idx_A0  = np.random.choice(n_valid, n_draws, replace=False)

    IRF_all = np.zeros((n_draws, H + 1, N, N))

    for d in range(n_draws):
        B  = B_draws[idx_B[d]]
        A0 = A0_list[idx_A0[d]]
        C  = compute_var_companion(B, N, P)

        # MA representation
        e1 = np.zeros((N * P, N))
        e1[:N, :] = np.eye(N)
        MA = np.zeros((H + 1, N * P, N * P))
        MA[0] = np.eye(N * P)
        for h in range(1, H + 1):
            MA[h] = MA[h-1] @ C.T

        # IRF_d[h, var, shock] = (e1' MA[h] e1) A0
        for h in range(H + 1):
            phi_h = (e1.T @ MA[h] @ e1)   # N × N
            IRF_all[d, h, :, :] = phi_h @ A0

    return IRF_all


print("\n--- Computing IRFs ---")
IRF_draws = compute_irf_draws(B_draws, valid_A0_list, N, P, H)
print(f"IRF draws shape: {IRF_draws.shape}")

IRF_mean  = IRF_draws.mean(axis=0)
IRF_lower = np.percentile(IRF_draws, 16, axis=0)
IRF_upper = np.percentile(IRF_draws, 84, axis=0)

# Numeric output — IRF tables
print("\n--- IRF mean (monetary shock → all variables) ---")
shock_idx = SHOCK_NAMES.index("monetary")
irf_monetary = pd.DataFrame(IRF_mean[:, :, shock_idx],
                              columns=var_names)
irf_monetary.index.name = "horizon"
print(irf_monetary.round(4))


# ============================================================
# SECTION 7 — FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)
# ============================================================
def compute_fevd(IRF_draws, N, H):
    """
    FEVD[h, var, shock] = share of variance of var at horizon h
                          explained by shock.
    """
    n_draws = IRF_draws.shape[0]
    FEVD_all = np.zeros((n_draws, H + 1, N, N))

    for d in range(n_draws):
        for h in range(H + 1):
            # Cumulative squared IRFs up to h
            mse = np.zeros((N, N))   # mse[var, shock]
            for s in range(h + 1):
                mse += IRF_draws[d, s, :, :] ** 2
            total = mse.sum(axis=1, keepdims=True)
            total = np.where(total == 0, 1e-10, total)
            FEVD_all[d, h, :, :] = mse / total

    return FEVD_all


print("\n--- Computing FEVD ---")
FEVD_draws = compute_fevd(IRF_draws, N, H)
FEVD_mean  = FEVD_draws.mean(axis=0)
FEVD_lower = np.percentile(FEVD_draws, 16, axis=0)
FEVD_upper = np.percentile(FEVD_draws, 84, axis=0)

print("\n--- FEVD at horizon 12 ---")
fevd_h12 = pd.DataFrame(FEVD_mean[12, :, :],
                          index=var_names,
                          columns=[f"s_{s}" for s in SHOCK_NAMES])
print(fevd_h12.round(4))


# ============================================================
# SECTION 8 — HISTORICAL DECOMPOSITION
# ============================================================
def compute_historical_decomp(Y_dep, X_mat, B_post_mean, A0_mean, N, P, T_eff):
    """
    Decompose observed time series into structural shock contributions.
    Returns HD array (T_eff, N, N_shocks).
    """
    U_reduced = Y_dep - X_mat @ B_post_mean     # reduced-form residuals
    A0_inv    = np.linalg.inv(A0_mean)

    # Structural shocks
    eps_struct = U_reduced @ A0_inv.T            # (T_eff, N)

    C = compute_var_companion(B_post_mean, N, P)
    e1 = np.zeros((N * P, N))
    e1[:N, :] = np.eye(N)

    HD = np.zeros((T_eff, N, N))   # [time, var, shock]

    for t in range(T_eff):
        for shock in range(N):
            contrib = np.zeros(N)
            for s in range(t + 1):
                if t - s < 0:
                    continue
                # MA coefficient at horizon s
                MA_s = np.linalg.matrix_power(C.T, s)
                phi_s = e1.T @ MA_s @ e1
                a0_shock = A0_mean[:, shock]
                contrib += phi_s @ a0_shock * eps_struct[t - s, shock]
            HD[t, :, shock] = contrib

    return HD, eps_struct


print("\n--- Computing Historical Decomposition ---")
A0_mean_mat = A0_arr.mean(axis=0)
HD, eps_struct = compute_historical_decomp(
    Y_dep, X_mat, B_post_mean, A0_mean_mat, N, P, T_eff
)
print(f"HD shape: {HD.shape}")
print(f"Structural shocks shape: {eps_struct.shape}")

HD_df = {}
for vi, vname in enumerate(var_names):
    HD_df[vname] = pd.DataFrame(HD[:, vi, :],
                                  columns=[f"s_{s}" for s in SHOCK_NAMES])
print("\n--- HD[output] first 5 rows ---")
print(HD_df["output"].head(5).round(4))


# ============================================================
# SECTION 9 — PLOTS
# ============================================================
COLORS = {
    "climate":  "#2196F3",
    "demand":   "#4CAF50",
    "monetary": "#F44336",
    "supply":   "#FF9800",
    "fx":       "#9C27B0",
}

horizons = np.arange(H + 1)

# --- PLOT 1: IRFs ---
fig, axes = plt.subplots(N, N, figsize=(20, 16), sharex=True)
fig.suptitle("Impulse Response Functions — Posterior median ± 68% CI", fontsize=14)

for shock_i, shock_name in enumerate(SHOCK_NAMES):
    for var_i, var_name in enumerate(var_names):
        ax = axes[var_i, shock_i]
        ax.fill_between(horizons,
                         IRF_lower[:, var_i, shock_i],
                         IRF_upper[:, var_i, shock_i],
                         alpha=0.3, color=COLORS[shock_name])
        ax.plot(horizons, IRF_mean[:, var_i, shock_i],
                color=COLORS[shock_name], lw=1.8)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_title(f"{shock_name} → {var_name}", fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(out("irf_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: irf_plot.png")


# --- PLOT 2: FEVD stacked bar ---
fig, axes = plt.subplots(1, N, figsize=(22, 5))
fig.suptitle("Forecast Error Variance Decomposition — posterior mean", fontsize=13)

cmap = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
bar_horizons = [1, 4, 8, 12, 20, 24]

for vi, vname in enumerate(var_names):
    ax = axes[vi]
    bottom = np.zeros(len(bar_horizons))
    for si, sname in enumerate(SHOCK_NAMES):
        vals = [FEVD_mean[h, vi, si] for h in bar_horizons]
        ax.bar(range(len(bar_horizons)), vals, bottom=bottom,
               color=cmap[si], label=sname, alpha=0.85)
        bottom += np.array(vals)
    ax.set_title(vname, fontsize=10)
    ax.set_xticks(range(len(bar_horizons)))
    ax.set_xticklabels(bar_horizons, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("horizon", fontsize=8)

axes[0].set_ylabel("share", fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           ncol=N, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(out("fevd_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fevd_plot.png")


# --- PLOT 3: Historical decomposition (output variable) ---
vi_out = var_names.index("output")
fig, axes = plt.subplots(N + 1, 1, figsize=(16, 14), sharex=True)
fig.suptitle("Historical Decomposition — output", fontsize=13)

time_axis = np.arange(T_eff)
total_fitted = HD[:, vi_out, :].sum(axis=1)

for si, sname in enumerate(SHOCK_NAMES):
    ax = axes[si]
    ax.bar(time_axis, HD[:, vi_out, si], color=cmap[si], alpha=0.8,
           label=sname, width=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(sname, fontsize=8, rotation=0, labelpad=40)
    ax.tick_params(labelsize=7)

# Observed vs fitted
axes[N].plot(time_axis, Y_dep[:, vi_out], color="black", lw=1.2, label="observed")
axes[N].plot(time_axis, total_fitted, color="red", lw=1.0, ls="--", label="HD sum")
axes[N].legend(fontsize=7)
axes[N].set_xlabel("time", fontsize=9)
axes[N].tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(out("hd_output_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: hd_output_plot.png")


# --- PLOT 4: Posterior distributions of key parameters ---
fig, axes = plt.subplots(2, N, figsize=(20, 8))
fig.suptitle("Posterior distributions — diagonal B coefficients (lag 1)", fontsize=13)

for i, vname in enumerate(var_names):
    # Own lag-1 coefficient
    b_draws_i = B_draws[:, 1 + i, i]
    ax = axes[0, i]
    ax.hist(b_draws_i, bins=40, color="#2196F3", alpha=0.75, edgecolor="white")
    ax.axvline(b_draws_i.mean(), color="red", lw=1.5, label=f"mean={b_draws_i.mean():.3f}")
    ax.axvline(B_ols[1 + i, i], color="black", lw=1.2, ls="--", label=f"OLS={B_ols[1+i,i]:.3f}")
    ax.set_title(f"{vname} AR(1)", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Sigma diagonal
    sig_draws_i = Sig_draws[:, i, i]
    ax2 = axes[1, i]
    ax2.hist(sig_draws_i, bins=40, color="#FF9800", alpha=0.75, edgecolor="white")
    ax2.axvline(sig_draws_i.mean(), color="red", lw=1.5)
    ax2.set_title(f"σ²({vname})", fontsize=9)
    ax2.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(out("posterior_params_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: posterior_params_plot.png")


# ============================================================
# NUMERIC OUTPUT — EXPORT ARRAYS
# ============================================================

# IRF dataframes per shock
irf_tables = {}
for si, sname in enumerate(SHOCK_NAMES):
    df_irf = pd.DataFrame({
        "horizon": horizons,
    })
    for vi, vname in enumerate(var_names):
        df_irf[f"{vname}_mean"]  = IRF_mean[:, vi, si]
        df_irf[f"{vname}_lower"] = IRF_lower[:, vi, si]
        df_irf[f"{vname}_upper"] = IRF_upper[:, vi, si]
    irf_tables[sname] = df_irf

print("\n--- IRF table (climate shock) ---")
print(irf_tables["climate"].round(5).to_string(index=False))

# FEVD summary table at horizon 24
fevd_h24 = pd.DataFrame(FEVD_mean[24, :, :],
                          index=var_names,
                          columns=[f"s_{s}" for s in SHOCK_NAMES])
print("\n--- FEVD at H=24 ---")
print(fevd_h24.round(4))

# Save CSV outputs
irf_tables["monetary"].to_csv(out("irf_monetary.csv"), index=False)
fevd_h24.to_csv(out("fevd_h24.csv"))
for vi, vname in enumerate(var_names):
    HD_df[vname].to_csv(out(f"hd_{vname}.csv"), index=False)

print("\n--- All outputs saved ---")
print("Plots : irf_plot.png, fevd_plot.png, hd_output_plot.png, posterior_params_plot.png")
print("CSVs  : irf_monetary.csv, fevd_h24.csv, hd_*.csv")