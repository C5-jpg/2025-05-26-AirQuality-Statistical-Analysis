"""
Step 1: Exploratory Data Analysis & Descriptive Statistics
Outputs: summary tables, basic overview
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from scipy import stats
from config import *

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(os.path.join(DATA_DIR, '数据集3_空气质量.xlsx'))
print("=" * 70)
print("  AIR QUALITY DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print(f"\nDataset: {df.shape[0]} observations × {df.shape[1]} variables")
print(f"No missing values: {df.isnull().sum().sum() == 0}")

# ─── 1. Overall Descriptive Statistics ────────────────────────────────────────
print("\n" + "─" * 70)
print("  1. OVERALL DESCRIPTIVE STATISTICS")
print("─" * 70)

desc = df[NUMERIC_COLS].describe().T
desc['skewness'] = df[NUMERIC_COLS].skew()
desc['kurtosis'] = df[NUMERIC_COLS].kurtosis()
desc['cv(%)'] = (desc['std'] / desc['mean'] * 100).round(2)
desc = desc[['count', 'mean', 'std', 'cv(%)', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']]
desc.columns = ['N', 'Mean', 'SD', 'CV(%)', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Skew', 'Kurt']
desc.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_statistics.csv'), encoding='utf-8-sig')
print(desc.round(3).to_string())

# ─── 2. Seasonal Descriptive Statistics ───────────────────────────────────────
print("\n" + "─" * 70)
print("  2. SEASONAL DESCRIPTIVE STATISTICS (PM2.5)")
print("─" * 70)

seasonal = df.groupby('season')['pm25'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
seasonal.columns = ['N', 'Mean', 'SD', 'Median', 'Min', 'Max']
seasonal = seasonal.reindex(SEASON_ORDER)
seasonal.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_pm25.csv'), encoding='utf-8-sig')
print(seasonal.round(2).to_string())

# ─── 3. Seasonal stats for all variables ─────────────────────────────────────
print("\n" + "─" * 70)
print("  3. SEASONAL STATISTICS (ALL VARIABLES)")
print("─" * 70)

for col in NUMERIC_COLS:
    s = df.groupby('season')[col].agg(['mean', 'std']).reindex(SEASON_ORDER)
    s.columns = [f'{col}_mean', f'{col}_sd']
    print(f"\n  {COL_NAMES.get(col, col)}:")
    print(s.round(2).to_string())

# ─── 4. Correlation Matrix ───────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  4. PEARSON CORRELATION MATRIX")
print("─" * 70)

corr = df[NUMERIC_COLS].corr(method='pearson')
pvals = pd.DataFrame(np.zeros_like(corr), index=corr.index, columns=corr.columns)
for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        if i != j:
            r, p = stats.pearsonr(df[NUMERIC_COLS[i]], df[NUMERIC_COLS[j]])
            pvals.iloc[i, j] = p

print(corr.round(3).to_string())
corr.to_csv(os.path.join(OUTPUT_DIR, 'pearson_correlation.csv'), encoding='utf-8-sig')
pvals.to_csv(os.path.join(OUTPUT_DIR, 'pearson_pvalues.csv'), encoding='utf-8-sig')

# ─── 5. Spearman Correlation ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  5. SPEARMAN RANK CORRELATION MATRIX")
print("─" * 70)

corr_sp = df[NUMERIC_COLS].corr(method='spearman')
print(corr_sp.round(3).to_string())
corr_sp.to_csv(os.path.join(OUTPUT_DIR, 'spearman_correlation.csv'), encoding='utf-8-sig')

# ─── 6. Normality Tests ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  6. NORMALITY TESTS")
print("─" * 70)

normality = []
for col in NUMERIC_COLS:
    stat_sw, p_sw = stats.shapiro(df[col])
    stat_ks, p_ks = stats.kstest(df[col], 'norm', args=(df[col].mean(), df[col].std()))
    normality.append({
        'Variable': COL_NAMES.get(col, col),
        'Shapiro-Wilk W': round(stat_sw, 4),
        'Shapiro p-value': f'{p_sw:.4f}' if p_sw >= 0.001 else '<0.001',
        'K-S statistic': round(stat_ks, 4),
        'K-S p-value': f'{p_ks:.4f}' if p_ks >= 0.001 else '<0.001',
        'Normal? (α=0.05)': 'Yes' if p_sw > 0.05 else 'No'
    })

norm_df = pd.DataFrame(normality)
norm_df.to_csv(os.path.join(OUTPUT_DIR, 'normality_tests.csv'), index=False, encoding='utf-8-sig')
print(norm_df.to_string(index=False))

# ─── 7. Outlier Detection (IQR method) ───────────────────────────────────────
print("\n" + "─" * 70)
print("  7. OUTLIER DETECTION (IQR Method)")
print("─" * 70)

for col in NUMERIC_COLS:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        print(f"\n  {COL_NAMES.get(col, col)}: {len(outliers)} outliers")
        print(f"    Range: [{lower:.2f}, {upper:.2f}]")
        print(f"    Outlier values: {outliers[col].tolist()}")
    else:
        print(f"\n  {COL_NAMES.get(col, col)}: No outliers detected")

print("\n" + "=" * 70)
print("  EDA COMPLETE — Tables saved to outputs/tables/")
print("=" * 70)
