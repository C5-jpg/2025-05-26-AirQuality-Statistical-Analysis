"""
Step 3: Statistical Testing & Regression Analysis
Comprehensive statistical inference for air quality data
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
from config import *

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(os.path.join(DATA_DIR, '数据集3_空气质量.xlsx'))

report = []
report.append("=" * 70)
report.append("  STATISTICAL ANALYSIS REPORT")
report.append("  Air Quality Dataset — PM2.5 Analysis")
report.append("=" * 70)

# ─── 1. Normality Tests (detailed) ───────────────────────────────────────────
report.append("\n" + "─" * 70)
report.append("  SECTION 1: NORMALITY ASSESSMENT")
report.append("─" * 70)
report.append("  H₀: The data follows a normal distribution")
report.append("  H₁: The data does not follow a normal distribution")
report.append("")

normality_results = []
for col in NUMERIC_COLS:
    data = df[col].dropna()
    sw_stat, sw_p = stats.shapiro(data)
    dag_stat, dag_p = stats.normaltest(data)  # D'Agostino-Pearson
    normality_results.append({
        'Variable': COL_NAMES.get(col, col),
        'Shapiro-Wilk W': round(sw_stat, 4),
        'Shapiro p-value': sw_p,
        "D'Agostino K²": round(dag_stat, 4),
        "D'Agostino p-value": dag_p,
        'Conclusion (α=0.05)': 'Normal' if sw_p > 0.05 else 'Non-normal'
    })
    report.append(f"  {COL_NAMES.get(col, col)}:")
    report.append(f"    Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}")
    report.append(f"    D'Agostino: K²={dag_stat:.4f}, p={dag_p:.4f}")
    report.append(f"    → {'Normal distribution' if sw_p > 0.05 else 'Non-normal distribution (use non-parametric tests)'}")
    report.append("")

norm_df = pd.DataFrame(normality_results)
norm_df.to_csv(os.path.join(OUTPUT_DIR, 'normality_detailed.csv'), index=False, encoding='utf-8-sig')

# ─── 2. Correlation Analysis ─────────────────────────────────────────────────
report.append("─" * 70)
report.append("  SECTION 2: CORRELATION ANALYSIS")
report.append("─" * 70)

report.append("\n  2.1 Pearson Correlation (PM2.5 vs Predictors)")
report.append("  H₀: ρ = 0 (no linear correlation)")
report.append("  H₁: ρ ≠ 0")
report.append("")

corr_results = []
for pred in PREDICTORS:
    r_pear, p_pear = stats.pearsonr(df[pred], df['pm25'])
    r_spear, p_spear = stats.spearmanr(df[pred], df['pm25'])
    n = len(df)
    # Fisher z-test for significance
    t_stat = r_pear * np.sqrt(n - 2) / np.sqrt(1 - r_pear**2)

    corr_results.append({
        'Predictor': COL_NAMES.get(pred, pred),
        'Pearson r': round(r_pear, 4),
        'Pearson p': p_pear,
        'Spearman ρ': round(r_spear, 4),
        'Spearman p': p_spear,
        't-statistic': round(t_stat, 4),
        'R²': round(r_pear**2, 4),
        'Strength': 'Strong' if abs(r_pear) > 0.5 else 'Moderate' if abs(r_pear) > 0.3 else 'Weak'
    })

    sig = '***' if p_pear < 0.001 else '**' if p_pear < 0.01 else '*' if p_pear < 0.05 else 'n.s.'
    report.append(f"  {COL_NAMES.get(pred, pred)}:")
    report.append(f"    Pearson r = {r_pear:.4f} (p = {p_pear:.4f}) {sig}")
    report.append(f"    Spearman ρ = {r_spear:.4f} (p = {p_spear:.4f})")
    report.append(f"    R² = {r_pear**2:.4f} | Interpretation: {r_pear**2*100:.1f}% variance explained")
    report.append("")

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(os.path.join(OUTPUT_DIR, 'correlation_analysis.csv'), index=False, encoding='utf-8-sig')

# ─── 3. Independent Samples Tests (Seasonal) ────────────────────────────────
report.append("─" * 70)
report.append("  SECTION 3: SEASONAL DIFFERENCE TESTS")
report.append("─" * 70)

# 3.1 One-way ANOVA
report.append("\n  3.1 One-way ANOVA (PM2.5 across seasons)")
report.append("  H₀: μ_spring = μ_summer = μ_autumn = μ_winter")
report.append("  H₁: At least one mean differs")
report.append("")

season_groups = [df[df['season'] == s]['pm25'].values for s in SEASON_ORDER]
f_stat, p_anova = stats.f_oneway(*season_groups)
report.append(f"  F({len(SEASON_ORDER)-1}, {len(df)-len(SEASON_ORDER)}) = {f_stat:.4f}")
report.append(f"  p-value = {p_anova:.6f}")
report.append(f"  → {'Reject H₀: Significant seasonal differences' if p_anova < 0.05 else 'Fail to reject H₀'}")
report.append("")

# Effect size (η²)
ss_between = sum(len(g) * (g.mean() - df['pm25'].mean())**2 for g in season_groups)
ss_total = sum((df['pm25'] - df['pm25'].mean())**2)
eta_sq = ss_between / ss_total
report.append(f"  Effect size: η² = {eta_sq:.4f} ({'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'})")
report.append("")

# 3.2 Kruskal-Wallis (non-parametric alternative)
h_stat, p_kw = stats.kruskal(*season_groups)
report.append(f"  3.2 Kruskal-Wallis Test (non-parametric)")
report.append(f"  H = {h_stat:.4f}, p = {p_kw:.6f}")
report.append(f"  → {'Significant' if p_kw < 0.05 else 'Not significant'}")
report.append("")

# 3.3 Post-hoc pairwise comparisons
report.append("  3.3 Pairwise Comparisons (Bonferroni-corrected)")
report.append("")

pairwise = []
season_pairs = list(combinations(SEASON_ORDER, 2))
n_comparisons = len(season_pairs)
for s1, s2 in season_pairs:
    d1 = df[df['season'] == s1]['pm25']
    d2 = df[df['season'] == s2]['pm25']

    # Welch's t-test
    t_stat, p_t = stats.ttest_ind(d1, d2, equal_var=False)
    p_bonf = min(p_t * n_comparisons, 1.0)

    # Cohen's d
    pooled_std = np.sqrt((d1.std()**2 + d2.std()**2) / 2)
    cohens_d = (d1.mean() - d2.mean()) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U
    u_stat, p_mw = stats.mannwhitneyu(d1, d2, alternative='two-sided')

    pairwise.append({
        'Comparison': f'{s1} vs {s2}',
        'Mean Diff': round(d1.mean() - d2.mean(), 2),
        "Welch's t": round(t_stat, 3),
        'p-value (raw)': p_t,
        'p-value (Bonf.)': p_bonf,
        "Cohen's d": round(cohens_d, 3),
        'U-statistic': u_stat,
        'M-W p-value': p_mw,
        'Significant': 'Yes' if p_bonf < 0.05 else 'No'
    })

    sig_bonf = '*' if p_bonf < 0.05 else 'n.s.'
    report.append(f"  {s1} vs {s2}:")
    report.append(f"    Mean diff = {d1.mean()-d2.mean():.2f}")
    report.append(f"    Welch's t = {t_stat:.3f}, p = {p_t:.4f}, Bonf. p = {p_bonf:.4f} {sig_bonf}")
    report.append(f"    Cohen's d = {cohens_d:.3f}")
    report.append(f"    Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.4f}")
    report.append("")

pair_df = pd.DataFrame(pairwise)
pair_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_comparisons.csv'), index=False, encoding='utf-8-sig')

# ─── 4. Multiple Regression Analysis ─────────────────────────────────────────
report.append("─" * 70)
report.append("  SECTION 4: MULTIPLE LINEAR REGRESSION")
report.append("─" * 70)
report.append("  Dependent Variable: PM2.5 (μg/m³)")
report.append("  Independent Variables: Temperature, Humidity, Wind Speed, Traffic Index")
report.append("")

# Manual OLS using numpy
X = df[PREDICTORS].values
y = df['pm25'].values
n = len(y)
k = len(PREDICTORS)

# Add intercept
X_with_intercept = np.column_stack([np.ones(n), X])

# OLS: β = (X'X)⁻¹X'y
XtX = X_with_intercept.T @ X_with_intercept
XtX_inv = np.linalg.inv(XtX)
beta = XtX_inv @ X_with_intercept.T @ y

# Predictions and residuals
y_pred = X_with_intercept @ beta
residuals = y - y_pred

# R²
SS_res = np.sum(residuals**2)
SS_tot = np.sum((y - y.mean())**2)
R2 = 1 - SS_res / SS_tot
R2_adj = 1 - (1 - R2) * (n - 1) / (n - k - 1)

# F-statistic
SS_reg = SS_tot - SS_res
F_stat = (SS_reg / k) / (SS_res / (n - k - 1))
F_pvalue = 1 - stats.f.cdf(F_stat, k, n - k - 1)

# Standard errors and t-tests
MSE = SS_res / (n - k - 1)
var_beta = MSE * XtX_inv
se_beta = np.sqrt(np.diag(var_beta))
t_stats = beta / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

# Confidence intervals
t_crit = stats.t.ppf(0.975, n - k - 1)
ci_lower = beta - t_crit * se_beta
ci_upper = beta + t_crit * se_beta

report.append("  4.1 Model Summary")
report.append(f"  R² = {R2:.4f}")
report.append(f"  Adjusted R² = {R2_adj:.4f}")
report.append(f"  F({k}, {n-k-1}) = {F_stat:.4f}")
report.append(f"  p-value (model) = {F_pvalue:.6f}")
report.append(f"  RMSE = {np.sqrt(MSE):.4f}")
report.append(f"  AIC = {n * np.log(SS_res/n) + 2*(k+1):.4f}")
report.append(f"  BIC = {n * np.log(SS_res/n) + (k+1)*np.log(n):.4f}")
report.append("")

report.append("  4.2 Coefficients")
report.append(f"  {'Variable':<20} {'β':>8} {'SE':>8} {'t':>8} {'p':>10} {'95% CI':>20}")
report.append("  " + "-" * 80)

var_names = ['Intercept'] + [COL_NAMES.get(p, p) for p in PREDICTORS]
reg_results = []
for i, name in enumerate(var_names):
    stars = '***' if p_values[i] < 0.001 else '**' if p_values[i] < 0.01 else '*' if p_values[i] < 0.05 else ''
    report.append(f"  {name:<20} {beta[i]:>8.4f} {se_beta[i]:>8.4f} {t_stats[i]:>8.4f} {p_values[i]:>10.6f} [{ci_lower[i]:.2f}, {ci_upper[i]:.2f}] {stars}")
    reg_results.append({
        'Variable': name,
        'Coefficient (β)': round(beta[i], 4),
        'Std. Error': round(se_beta[i], 4),
        't-statistic': round(t_stats[i], 4),
        'p-value': p_values[i],
        'CI 2.5%': round(ci_lower[i], 4),
        'CI 97.5%': round(ci_upper[i], 4),
        'Significance': stars if stars else 'n.s.'
    })
report.append("")

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_coefficients.csv'), index=False, encoding='utf-8-sig')

# ─── 5. VIF (Variance Inflation Factor) ──────────────────────────────────────
report.append("  4.3 Multicollinearity Diagnostics (VIF)")
report.append("")

vif_results = []
for i, pred in enumerate(PREDICTORS):
    other_preds = [PREDICTORS[j] for j in range(k) if j != i]
    X_other = df[other_preds].values
    X_other_int = np.column_stack([np.ones(n), X_other])
    y_vif = df[pred].values
    beta_vif = np.linalg.lstsq(X_other_int, y_vif, rcond=None)[0]
    y_pred_vif = X_other_int @ beta_vif
    SS_res_vif = np.sum((y_vif - y_pred_vif)**2)
    SS_tot_vif = np.sum((y_vif - y_vif.mean())**2)
    R2_vif = 1 - SS_res_vif / SS_tot_vif
    vif = 1 / (1 - R2_vif) if R2_vif < 1 else float('inf')
    vif_results.append({
        'Variable': COL_NAMES.get(pred, pred),
        'VIF': round(vif, 3),
        'Concern': 'High (>10)' if vif > 10 else 'Moderate (5-10)' if vif > 5 else 'Low (<5)'
    })
    report.append(f"  {COL_NAMES.get(pred, pred)}: VIF = {vif:.3f} ({'High' if vif > 10 else 'Moderate' if vif > 5 else 'Low'})")
report.append("")

vif_df = pd.DataFrame(vif_results)
vif_df.to_csv(os.path.join(OUTPUT_DIR, 'vif_analysis.csv'), index=False, encoding='utf-8-sig')

# ─── 6. Residual Diagnostics ─────────────────────────────────────────────────
report.append("  4.4 Residual Diagnostics")
report.append("")

# Normality of residuals
sw_res_stat, sw_res_p = stats.shapiro(residuals)
report.append(f"  Shapiro-Wilk test on residuals: W = {sw_res_stat:.4f}, p = {sw_res_p:.4f}")
report.append(f"  → Residuals are {'normally distributed' if sw_res_p > 0.05 else 'NOT normally distributed'}")

# Durbin-Watson
dw = np.sum(np.diff(residuals)**2) / SS_res
report.append(f"  Durbin-Watson statistic: {dw:.4f}")
report.append(f"  → {'No significant autocorrelation' if 1.5 < dw < 2.5 else 'Potential autocorrelation detected'}")

# Breusch-Pagan test (manual)
resid_sq = residuals**2
X_bp = np.column_stack([np.ones(n), X])
bp_beta = np.linalg.lstsq(X_bp, resid_sq, rcond=None)[0]
resid_sq_pred = X_bp @ bp_beta
SS_bp = np.sum((resid_sq_pred - resid_sq_pred.mean())**2)
BP_stat = n * SS_bp / np.sum((resid_sq - resid_sq.mean())**2)
BP_p = 1 - stats.chi2.cdf(BP_stat, k)
report.append(f"  Breusch-Pagan test: LM = {BP_stat:.4f}, p = {BP_p:.4f}")
report.append(f"  → {'Homoscedastic' if BP_p > 0.05 else 'Heteroscedastic'} residuals")
report.append("")

# ─── 7. Standardized Regression (Beta Coefficients) ──────────────────────────
report.append("  4.5 Standardized Coefficients (Beta Weights)")
report.append("")

X_std = (X - X.mean(axis=0)) / X.std(axis=0)
y_std = (y - y.mean()) / y.std()
beta_std = np.linalg.lstsq(X_std, y_std, rcond=None)[0]

for i, pred in enumerate(PREDICTORS):
    report.append(f"  {COL_NAMES.get(pred, pred)}: β_std = {beta_std[i]:.4f}")
report.append("")

std_results = pd.DataFrame({
    'Variable': [COL_NAMES.get(p, p) for p in PREDICTORS],
    'Standardized β': [round(b, 4) for b in beta_std],
    '|Standardized β|': [round(abs(b), 4) for b in beta_std]
}).sort_values('|Standardized β|', ascending=False)
std_results.to_csv(os.path.join(OUTPUT_DIR, 'standardized_coefficients.csv'), index=False, encoding='utf-8-sig')

# ─── 8. Levene's Test for Homogeneity of Variance ────────────────────────────
report.append("─" * 70)
report.append("  SECTION 5: HOMOGENEITY OF VARIANCE (LEVENE'S TEST)")
report.append("─" * 70)

for col in NUMERIC_COLS:
    groups = [df[df['season'] == s][col].values for s in SEASON_ORDER]
    lev_stat, lev_p = stats.levene(*groups)
    report.append(f"  {COL_NAMES.get(col, col)}: F = {lev_stat:.4f}, p = {lev_p:.4f} → {'Equal variances' if lev_p > 0.05 else 'Unequal variances'}")
report.append("")

# ─── Save Full Report ────────────────────────────────────────────────────────
report_text = "\n".join(report)
with open(os.path.join(DATA_DIR, 'outputs', 'reports', 'statistical_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print("\n" + "=" * 70)
print("  Report saved to outputs/reports/statistical_report.txt")
print("  Tables saved to outputs/tables/")
print("=" * 70)
