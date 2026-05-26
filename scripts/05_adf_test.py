"""
Step 5: Augmented Dickey-Fuller (ADF) Test & Time Series Stationarity Analysis
Generates a comprehensive ADF report figure (PDF + PNG)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from config import *
import warnings
warnings.filterwarnings('ignore')

# Publication-quality axis styling helper
def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color='#333333')
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color='#333333')
    if title: ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#1a1a1a')
    ax.tick_params(colors='#555555', labelsize=9)
    ax.grid(axis='y', alpha=0.2, linestyle='--', color='#999999')

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(DATA_DIR, 'outputs', 'figures')
TABLE_DIR = os.path.join(DATA_DIR, 'outputs', 'tables')
REPORT_DIR = os.path.join(DATA_DIR, 'outputs', 'reports')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_excel(os.path.join(DATA_DIR, 'data', 'raw', '数据集3_空气质量.xlsx'))

# Use date_id as implicit time index
series_dict = {}
for col in NUMERIC_COLS:
    series_dict[col] = df[col].values

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: ADF Test Computation
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  AUGMENTED DICKEY-FULLER TEST RESULTS")
print("=" * 70)

adf_results = []
for col in NUMERIC_COLS:
    data = series_dict[col]

    # ADF test with different lag selections
    for criterion in ['AIC', 'BIC']:
        result = adfuller(data, autolag=criterion)

        adf_results.append({
            'Variable': COL_NAMES.get(col, col),
            'Criterion': criterion,
            'ADF Statistic': round(result[0], 4),
            'p-value': round(result[1], 6),
            'Used Lags': result[2],
            'N Obs': result[3],
            'Critical 1%': round(result[4]['1%'], 4),
            'Critical 5%': round(result[4]['5%'], 4),
            'Critical 10%': round(result[4]['10%'], 4),
            'Stationary (α=0.05)': 'Yes' if result[1] < 0.05 else 'No',
            'Stationary (α=0.01)': 'Yes' if result[1] < 0.01 else 'No',
        })

        sig_05 = '***' if result[1] < 0.01 else '**' if result[1] < 0.05 else '*' if result[1] < 0.1 else 'n.s.'
        print(f"\n  {COL_NAMES.get(col, col)} (lag selection: {criterion}):")
        print(f"    ADF Statistic = {result[0]:.4f}")
        print(f"    p-value       = {result[1]:.6f} {sig_05}")
        print(f"    Lags used     = {result[2]}")
        print(f"    Critical values: 1%={result[4]['1%']:.4f}, 5%={result[4]['5%']:.4f}, 10%={result[4]['10%']:.4f}")
        print(f"    → {'Stationary' if result[1] < 0.05 else 'NON-STATIONARY'} at α=0.05")

adf_df = pd.DataFrame(adf_results)
adf_df.to_csv(os.path.join(TABLE_DIR, 'adf_test_results.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: KPSS Test (complementary — H₀ is stationarity)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  KPSS TEST RESULTS (H₀: Stationary)")
print("=" * 70)

kpss_results = []
for col in NUMERIC_COLS:
    data = series_dict[col]
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(data, regression='c', nlags='auto')

    kpss_results.append({
        'Variable': COL_NAMES.get(col, col),
        'KPSS Statistic': round(kpss_stat, 4),
        'p-value': round(kpss_p, 6),
        'Lags used': kpss_lags,
        'Critical 1%': round(kpss_crit['1%'], 4),
        'Critical 5%': round(kpss_crit['5%'], 4),
        'Critical 10%': round(kpss_crit['10%'], 4),
        'Stationary (α=0.05)': 'Yes' if kpss_p > 0.05 else 'No',
    })

    print(f"\n  {COL_NAMES.get(col, col)}:")
    print(f"    KPSS Statistic = {kpss_stat:.4f}")
    print(f"    p-value        = {kpss_p:.6f}")
    print(f"    → {'Stationary' if kpss_p > 0.05 else 'NON-STATIONARY'} at α=0.05")

kpss_df = pd.DataFrame(kpss_results)
kpss_df.to_csv(os.path.join(TABLE_DIR, 'kpss_test_results.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: First Difference ADF Test (if needed)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FIRST-DIFFERENCE ADF TEST")
print("=" * 70)

diff_results = []
for col in NUMERIC_COLS:
    data_diff = np.diff(series_dict[col])
    result = adfuller(data_diff, autolag='AIC')

    diff_results.append({
        'Variable': f'Δ{COL_NAMES.get(col, col)}',
        'ADF Statistic': round(result[0], 4),
        'p-value': round(result[1], 6),
        'Used Lags': result[2],
        'Critical 5%': round(result[4]['5%'], 4),
        'Stationary (α=0.05)': 'Yes' if result[1] < 0.05 else 'No',
    })

    print(f"\n  Δ{COL_NAMES.get(col, col)}:")
    print(f"    ADF Statistic = {result[0]:.4f}, p = {result[1]:.6f}")
    print(f"    → {'Stationary' if result[1] < 0.05 else 'NON-STATIONARY'}")

diff_df = pd.DataFrame(diff_results)
diff_df.to_csv(os.path.join(TABLE_DIR, 'adf_first_diff_results.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 17: ADF Test Comprehensive Report (Publication Quality)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 17: ADF Test Comprehensive Report...")

fig = plt.figure(figsize=(16, 24))
outer_gs = gridspec.GridSpec(6, 1, hspace=0.4, height_ratios=[1.2, 1, 1, 1, 1, 1.8])

# ─── Panel A: Summary Table ──────────────────────────────────────────────────
ax_table = fig.add_subplot(outer_gs[0])
ax_table.axis('off')
ax_table.set_title('(a) Augmented Dickey-Fuller (ADF) Unit Root Test Summary',
                    fontsize=13, fontweight='bold', loc='left', pad=15)

# Build table data
table_data = []
col_labels = ['Variable', 'ADF Stat.', 'p-value', 'Lags', 'Crit. 1%', 'Crit. 5%', 'Crit. 10%', 'Result']
for col in NUMERIC_COLS:
    data = series_dict[col]
    result = adfuller(data, autolag='AIC')
    stationarity = 'Stationary ✓' if result[1] < 0.05 else 'Non-stationary ✗'
    table_data.append([
        COL_NAMES_SHORT[col],
        f'{result[0]:.3f}',
        f'{result[1]:.4f}' if result[1] >= 0.001 else '<0.001',
        str(result[2]),
        f'{result[4]["1%"]:.3f}',
        f'{result[4]["5%"]:.3f}',
        f'{result[4]["10%"]:.3f}',
        stationarity
    ])

table = ax_table.table(cellText=table_data, colLabels=col_labels,
                        loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

# Style table
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    if row == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
    else:
        cell.set_facecolor('#F8F9FA' if row % 2 == 0 else 'white')
    if col == 7 and row > 0:
        if '✓' in table_data[row-1][7]:
            cell.set_facecolor('#D5F5E3')
            cell.set_text_props(fontweight='bold', color='#1B7A3D')
        else:
            cell.set_facecolor('#FADBD8')
            cell.set_text_props(fontweight='bold', color='#C0392B')

# ─── Panels B-F: Time Series + ADF visualization per variable ────────────────
for var_idx, col in enumerate(NUMERIC_COLS):
    row_idx = var_idx + 1
    ax = fig.add_subplot(outer_gs[row_idx])

    data = series_dict[col]
    x = np.arange(len(data))
    color = PALETTE_MAIN[var_idx]

    # Time series line
    ax.plot(x, data, color=color, linewidth=1.2, alpha=0.8, zorder=3)
    ax.fill_between(x, data, alpha=0.1, color=color)

    # Rolling mean (window=7)
    if len(data) >= 7:
        rolling_mean = pd.Series(data).rolling(window=7, center=True).mean()
        ax.plot(x, rolling_mean, color='#E74C3C', linewidth=1.8, linestyle='--',
                label='7-pt Moving Avg', zorder=4, alpha=0.8)

    # Rolling std
    if len(data) >= 7:
        rolling_std = pd.Series(data).rolling(window=7, center=True).std()
        upper = rolling_mean + 1.96 * rolling_std
        lower = rolling_mean - 1.96 * rolling_std
        ax.fill_between(x, lower, upper, alpha=0.15, color='#E74C3C', label='95% CI Band')

    # Overall mean line
    ax.axhline(data.mean(), color='#2C3E50', linestyle=':', linewidth=1, alpha=0.5)

    # ADF annotation
    adf_result = adfuller(data, autolag='AIC')
    adf_stat = adf_result[0]
    adf_p = adf_result[1]
    is_stationary = adf_p < 0.05

    text_str = (f'ADF = {adf_stat:.3f}, p = {adf_p:.4f}\n'
                f'{"Stationary" if is_stationary else "Non-stationary"}')
    bbox_color = '#D5F5E3' if is_stationary else '#FADBD8'
    text_color = '#1B7A3D' if is_stationary else '#C0392B'
    ax.text(0.98, 0.95, text_str, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=bbox_color, alpha=0.9, edgecolor='#CCCCCC'),
            color=text_color, fontweight='bold')

    panel_letter = chr(98 + var_idx)  # b, c, d, e, f
    ax.set_title(f'({panel_letter}) {COL_NAMES.get(col, col)}',
                 fontsize=11, fontweight='bold', loc='left')
    ax.set_xlabel('Time Index', fontsize=9)
    if var_idx == 0:
        ax.legend(fontsize=8, loc='upper left')

    style_ax(ax)

fig.suptitle('Time Series Stationarity Analysis — Augmented Dickey-Fuller Test',
             fontsize=15, fontweight='bold', y=0.99)

plt.savefig(os.path.join(FIG_DIR, 'fig17_adf_test_report.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig17_adf_test_report.pdf'), bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 18: ADF Critical Value Comparison Chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 18: ADF Critical Value Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

adf_stats = []
crit_1 = []
crit_5 = []
crit_10 = []
labels = []

for col in NUMERIC_COLS:
    result = adfuller(series_dict[col], autolag='AIC')
    adf_stats.append(result[0])
    crit_1.append(result[4]['1%'])
    crit_5.append(result[4]['5%'])
    crit_10.append(result[4]['10%'])
    labels.append(COL_NAMES_SHORT[col])

x = np.arange(len(labels))
width = 0.2

bars_adf = ax.bar(x - 1.5*width, adf_stats, width, label='ADF Statistic',
                   color=PALETTE_MAIN[:len(labels)], alpha=0.85, edgecolor='#333333', linewidth=0.5)

# Critical value lines
for i in range(len(labels)):
    ax.plot([x[i]-2*width, x[i]+2*width], [crit_1[i], crit_1[i]],
            color='#E74C3C', linewidth=2, linestyle='-', alpha=0.7)
    ax.plot([x[i]-2*width, x[i]+2*width], [crit_5[i], crit_5[i]],
            color='#F39C12', linewidth=2, linestyle='--', alpha=0.7)
    ax.plot([x[i]-2*width, x[i]+2*width], [crit_10[i], crit_10[i]],
            color='#3498DB', linewidth=2, linestyle=':', alpha=0.7)

# Add value labels
for bar, val in zip(bars_adf, adf_stats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.3,
            f'{val:.2f}', ha='center', va='top', fontsize=8, fontweight='bold', color='white')

# Legend for critical values
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#E74C3C', linewidth=2, linestyle='-', label='Critical 1%'),
    Line2D([0], [0], color='#F39C12', linewidth=2, linestyle='--', label='Critical 5%'),
    Line2D([0], [0], color='#3498DB', linewidth=2, linestyle=':', label='Critical 10%'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Test Statistic / Critical Value', fontsize=11)
ax.set_title('ADF Test Statistics vs. Critical Values\n(Bar below critical line → Reject H₀ → Stationary)',
             fontsize=13, fontweight='bold')
style_ax(ax)
ax.axhline(0, color='#999999', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig18_adf_critical_values.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig18_adf_critical_values.pdf'), bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 19: ACF/PACF for PM2.5 + Seasonal Decomposition
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 19: ACF/PACF + Seasonal Decomposition...")

fig = plt.figure(figsize=(14, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

# Panel A: ACF
ax_acf = fig.add_subplot(gs[0, 0])
plot_acf(df['pm25'].values, lags=20, ax=ax_acf, color='#4C72B0', vlines_kwargs={'color': '#4C72B0'})
ax_acf.set_title('(a) ACF — PM2.5', fontsize=11, fontweight='bold', loc='left')
ax_acf.set_xlabel('Lag', fontsize=9)
ax_acf.set_ylabel('Autocorrelation', fontsize=9)
style_ax(ax_acf)

# Panel B: PACF
ax_pacf = fig.add_subplot(gs[0, 1])
plot_pacf(df['pm25'].values, lags=20, ax=ax_pacf, color='#DD8452', vlines_kwargs={'color': '#DD8452'})
ax_pacf.set_title('(b) PACF — PM2.5', fontsize=11, fontweight='bold', loc='left')
ax_pacf.set_xlabel('Lag', fontsize=9)
ax_pacf.set_ylabel('Partial Autocorrelation', fontsize=9)
style_ax(ax_pacf)

# Panel C: Seasonal Decomposition (using seasonal_decompose with period=4 for 4 seasons)
pm25_series = pd.Series(df['pm25'].values, index=pd.RangeIndex(len(df)))
decomposition = seasonal_decompose(pm25_series, model='additive', period=4)

# Panel C: Trend + Original
ax_trend = fig.add_subplot(gs[1, 0])
ax_trend.plot(pm25_series, color='#4C72B0', linewidth=1, alpha=0.5, label='Original')
ax_trend.plot(decomposition.trend, color='#E74C3C', linewidth=2, label='Trend')
ax_trend.set_title('(c) Trend Component', fontsize=11, fontweight='bold', loc='left')
ax_trend.legend(fontsize=8)
style_ax(ax_trend)

# Panel D: Seasonal
ax_seasonal = fig.add_subplot(gs[1, 1])
ax_seasonal.plot(decomposition.seasonal, color='#55A868', linewidth=1.5)
ax_seasonal.set_title('(d) Seasonal Component', fontsize=11, fontweight='bold', loc='left')
style_ax(ax_seasonal)

# Panel E: Residual
ax_resid = fig.add_subplot(gs[2, 0])
ax_resid.plot(decomposition.resid, color='#C44E52', linewidth=1)
ax_resid.axhline(0, color='#999999', linestyle='--', linewidth=0.8)
ax_resid.set_title('(e) Residual Component', fontsize=11, fontweight='bold', loc='left')
style_ax(ax_resid)

# Panel F: Combined Summary
ax_summary = fig.add_subplot(gs[2, 1])
ax_summary.axis('off')
ax_summary.set_title('(f) Stationarity Decision Matrix', fontsize=11, fontweight='bold', loc='left')

# Build decision matrix
decision_data = []
decision_labels = ['Variable', 'ADF', 'KPSS', '1st Diff', 'Verdict']
for col in NUMERIC_COLS:
    adf_r = adfuller(series_dict[col], autolag='AIC')
    kpss_r = kpss(series_dict[col], regression='c', nlags='auto')
    diff_r = adfuller(np.diff(series_dict[col]), autolag='AIC')

    adf_verdict = 'S' if adf_r[1] < 0.05 else 'NS'
    kpss_verdict = 'S' if kpss_r[1] > 0.05 else 'NS'
    diff_verdict = 'S' if diff_r[1] < 0.05 else 'NS'

    if adf_verdict == 'S' and kpss_verdict == 'S':
        final = 'Stationary'
    elif adf_verdict == 'NS' and kpss_verdict == 'NS':
        final = 'Non-stationary'
    else:
        final = 'Conflicting'

    decision_data.append([COL_NAMES_SHORT[col], adf_verdict, kpss_verdict, diff_verdict, final])

dt = ax_summary.table(cellText=decision_data, colLabels=decision_labels,
                       loc='center', cellLoc='center')
dt.auto_set_font_size(False)
dt.set_fontsize(9)
dt.scale(1.0, 1.8)
for (row, col_idx), cell in dt.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    if row == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F8F9FA' if row % 2 == 0 else 'white')
        if col_idx == 4 and row > 0:
            if decision_data[row-1][4] == 'Stationary':
                cell.set_facecolor('#D5F5E3')
            elif decision_data[row-1][4] == 'Non-stationary':
                cell.set_facecolor('#FADBD8')
            else:
                cell.set_facecolor('#FEF9E7')

fig.suptitle('Time Series Analysis — ACF/PACF & Seasonal Decomposition (PM2.5)',
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig(os.path.join(FIG_DIR, 'fig19_acf_pacf_decomposition.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig19_acf_pacf_decomposition.pdf'), bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Save text report
# ═══════════════════════════════════════════════════════════════════════════════
report = []
report.append("=" * 70)
report.append("  STATIONARITY ANALYSIS REPORT")
report.append("  Augmented Dickey-Fuller (ADF) & KPSS Tests")
report.append("=" * 70)
report.append("")
report.append("  METHODOLOGY")
report.append("  ──────────────────────────────────────────────────────────")
report.append("  ADF Test:")
report.append("    H₀: The series has a unit root (non-stationary)")
report.append("    H₁: The series is stationary")
report.append("    Decision: Reject H₀ if ADF statistic < critical value")
report.append("")
report.append("  KPSS Test:")
report.append("    H₀: The series is trend-stationary")
report.append("    H₁: The series has a unit root (non-stationary)")
report.append("    Decision: Reject H₀ if KPSS statistic > critical value")
report.append("")
report.append("  Interpretation Matrix:")
report.append("    ADF=Reject + KPSS=Not Reject → Stationary")
report.append("    ADF=Not Reject + KPSS=Reject → Non-stationary")
report.append("    ADF=Reject + KPSS=Reject     → Difference-stationary")
report.append("    ADF=Not Reject + KPSS=Not Reject → Weak stationarity")
report.append("")
report.append("─" * 70)
report.append("  RESULTS (lag selection: AIC)")
report.append("─" * 70)

for col in NUMERIC_COLS:
    adf_r = adfuller(series_dict[col], autolag='AIC')
    kpss_r = kpss(series_dict[col], regression='c', nlags='auto')

    report.append(f"\n  {COL_NAMES.get(col, col)}:")
    report.append(f"    ADF:  τ = {adf_r[0]:.4f}, p = {adf_r[1]:.6f}, lags = {adf_r[2]}")
    report.append(f"          Critical: 1%={adf_r[4]['1%']:.4f}, 5%={adf_r[4]['5%']:.4f}, 10%={adf_r[4]['10%']:.4f}")
    report.append(f"    KPSS: τ = {kpss_r[0]:.4f}, p = {kpss_r[1]:.6f}")
    report.append(f"    → {'Stationary' if adf_r[1]<0.05 and kpss_r[1]>0.05 else 'Non-stationary' if adf_r[1]>=0.05 and kpss_r[1]<=0.05 else 'Conflicting evidence'}")

with open(os.path.join(REPORT_DIR, 'stationarity_report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print('\n'.join(report))
print("\n" + "=" * 70)
print("  OUTPUTS:")
print("  fig17_adf_test_report.png/pdf — Full ADF report figure")
print("  fig18_adf_critical_values.png/pdf — Critical value comparison")
print("  fig19_acf_pacf_decomposition.png/pdf — ACF/PACF & decomposition")
print("  adf_test_results.csv / kpss_test_results.csv / adf_first_diff_results.csv")
print("  stationarity_report.txt")
print("=" * 70)
