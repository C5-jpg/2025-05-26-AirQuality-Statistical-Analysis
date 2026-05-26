"""
Step 4: Advanced Publication-Quality Visualizations
Nature/Science journal standard figures
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from scipy import stats
from scipy.interpolate import make_interp_spline
import seaborn as sns
from config import *
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(DATA_DIR, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_excel(os.path.join(DATA_DIR, 'data', 'raw', '数据集3_空气质量.xlsx'))

# ─── Helper: Publication-quality styling ─────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Violin + Box + Scatter (Raincloud-style) — Nature Method style
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 7: Raincloud Plots (PM2.5 by Season)...")

fig, ax = plt.subplots(figsize=(10, 5.5))

positions = np.arange(len(SEASON_ORDER))
all_data = [df[df['season'] == s]['pm25'].values for s in SEASON_ORDER]

# Violin (half)
parts = ax.violinplot(all_data, positions=positions, showmeans=False,
                       showmedians=False, showextrema=False, widths=0.7)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(SEASON_COLORS[SEASON_ORDER[i]])
    pc.set_alpha(0.3)
    # Keep only left half
    m = np.mean(pc.get_paths()[0].vertices[:, 0])
    pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)

# Box plot
bp = ax.boxplot(all_data, positions=positions, widths=0.15, patch_artist=True,
                showfliers=False,
                medianprops=dict(color='#2C3E50', linewidth=2),
                whiskerprops=dict(linewidth=1.2, color='#555555'),
                capprops=dict(linewidth=1.2, color='#555555'),
                boxprops=dict(linewidth=1.2))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(SEASON_COLORS[SEASON_ORDER[i]])
    patch.set_alpha(0.85)
    patch.set_edgecolor('#333333')

# Individual scatter points (jittered)
for i, season in enumerate(SEASON_ORDER):
    data = df[df['season'] == season]['pm25'].values
    jitter = np.random.normal(0, 0.06, len(data))
    ax.scatter(data * 0 + positions[i] + 0.25 + jitter, data,
               color=SEASON_COLORS[season], alpha=0.5, s=20,
               edgecolors='white', linewidth=0.3, zorder=3)

# Significance brackets
def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color='#333333')
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=8, color='#333333')

# Compute pairwise significance for display
from itertools import combinations
y_max = df['pm25'].max()
bracket_y = y_max + 5
for idx, (s1, s2) in enumerate([(0,3), (1,3), (0,1)]):  # Selected pairs
    d1 = df[df['season'] == SEASON_ORDER[s1]]['pm25']
    d2 = df[df['season'] == SEASON_ORDER[s2]]['pm25']
    _, p = stats.ttest_ind(d1, d2, equal_var=False)
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    add_sig_bracket(ax, s1, s2, bracket_y + idx*10, 2, stars)

ax.set_xticks(positions)
ax.set_xticklabels([f'{s} ({SEASON_NAMES[s]})' for s in SEASON_ORDER], fontsize=10)
style_ax(ax, ylabel='PM2.5 (μg/m³)', title='Seasonal PM2.5 Distribution with Pairwise Comparisons')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_raincloud_pm25_season.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Pair Plot (Seaborn Enhanced) — Science style
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 8: Pair Plot...")

g = sns.pairplot(df[NUMERIC_COLS + ['season']],
                 hue='season',
                 hue_order=SEASON_ORDER,
                 palette=SEASON_COLORS,
                 diag_kind='kde',
                 plot_kws={'alpha': 0.6, 's': 25, 'edgecolor': 'white', 'linewidth': 0.3},
                 diag_kws={'alpha': 0.5, 'linewidth': 1.5},
                 corner=False)

# Rename axis labels
for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        ax = g.axes[i][j]
        if i == len(NUMERIC_COLS) - 1:
            ax.set_xlabel(COL_NAMES_SHORT[NUMERIC_COLS[j]], fontsize=9)
        if j == 0:
            ax.set_ylabel(COL_NAMES_SHORT[NUMERIC_COLS[i]], fontsize=9)

g.figure.suptitle('Multivariate Relationship Matrix — Air Quality Dataset', y=1.02,
                   fontsize=14, fontweight='bold')
g._legend.set_title('Season')
plt.savefig(os.path.join(FIG_DIR, 'fig8_pairplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Regression Diagnostic Panel (4-panel)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 9: Regression Diagnostics...")

# Fit regression
X = df[PREDICTORS].values
y = df['pm25'].values
n = len(y)
X_int = np.column_stack([np.ones(n), X])
beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
y_pred = X_int @ beta
residuals = y - y_pred
SS_res = np.sum(residuals**2)
MSE = SS_res / (n - len(PREDICTORS) - 1)
leverage = np.diag(X_int @ np.linalg.inv(X_int.T @ X_int) @ X_int.T)
std_resid = residuals / np.sqrt(MSE * (1 - leverage))
cooks_d = (std_resid**2 * leverage) / (len(PREDICTORS) + 1) / (1 - leverage)**2

fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Panel A: Residuals vs Fitted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_pred, residuals, c='#4C72B0', alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
ax1.axhline(0, color='#E74C3C', linestyle='--', linewidth=1.2)
# LOESS-like smooth
from scipy.signal import savgol_filter
sorted_idx = np.argsort(y_pred)
smooth = savgol_filter(residuals[sorted_idx], min(15, len(residuals)//2*2-1), 3)
ax1.plot(y_pred[sorted_idx], smooth, color='#E74C3C', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Fitted Values', fontsize=10)
ax1.set_ylabel('Residuals', fontsize=10)
ax1.set_title('(a) Residuals vs Fitted', fontsize=11, fontweight='bold')
style_ax(ax1)

# Panel B: Q-Q Plot
ax2 = fig.add_subplot(gs[0, 1])
(theoretical_q, ordered_resid), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
ax2.scatter(theoretical_q, ordered_resid, c='#55A868', alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
ax2.plot(theoretical_q, slope * np.array(theoretical_q) + intercept, color='#E74C3C', linewidth=1.5, linestyle='--')
ax2.set_xlabel('Theoretical Quantiles', fontsize=10)
ax2.set_ylabel('Sample Quantiles', fontsize=10)
ax2.set_title('(b) Normal Q-Q Plot', fontsize=11, fontweight='bold')
style_ax(ax2)

# Panel C: Scale-Location
ax3 = fig.add_subplot(gs[1, 0])
abs_sqrt_resid = np.sqrt(np.abs(std_resid))
ax3.scatter(y_pred, abs_sqrt_resid, c='#DD8452', alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
sorted_idx2 = np.argsort(y_pred)
smooth2 = savgol_filter(abs_sqrt_resid[sorted_idx2], min(15, len(abs_sqrt_resid)//2*2-1), 3)
ax3.plot(y_pred[sorted_idx2], smooth2, color='#E74C3C', linewidth=1.5, alpha=0.8)
ax3.set_xlabel('Fitted Values', fontsize=10)
ax3.set_ylabel('√|Standardized Residuals|', fontsize=10)
ax3.set_title('(c) Scale-Location Plot', fontsize=11, fontweight='bold')
style_ax(ax3)

# Panel D: Cook's Distance
ax4 = fig.add_subplot(gs[1, 1])
colors_cook = ['#C44E52' if d > 4/n else '#8172B3' for d in cooks_d]
ax4.stem(range(n), cooks_d, linefmt='-', markerfmt='o', basefmt=' ')
for i in range(n):
    ax4.plot(i, cooks_d[i], 'o', color=colors_cook[i], markersize=4, alpha=0.7)
ax4.axhline(4/n, color='#E74C3C', linestyle='--', linewidth=1.2, label=f'4/n = {4/n:.4f}')
ax4.set_xlabel('Observation Index', fontsize=10)
ax4.set_ylabel("Cook's Distance", fontsize=10)
ax4.set_title("(d) Cook's Distance", fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
style_ax(ax4)

fig.suptitle('Regression Diagnostic Plots', fontsize=14, fontweight='bold', y=0.98)
plt.savefig(os.path.join(FIG_DIR, 'fig9_regression_diagnostics.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Coefficient Plot (Forest Plot style)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 10: Coefficient Forest Plot...")

fig, ax = plt.subplots(figsize=(8, 5))

var_beta = MSE * np.linalg.inv(X_int.T @ X_int)
se_beta = np.sqrt(np.diag(var_beta))
t_crit = stats.t.ppf(0.975, n - len(PREDICTORS) - 1)

var_labels = ['Intercept'] + [COL_NAMES.get(p, p) for p in PREDICTORS]
y_pos = np.arange(len(var_labels))

# Plot coefficients with 95% CI
colors_coef = ['#4C72B0' if i == 0 else '#DD8452' if beta[i] > 0 else '#55A868' for i in range(len(var_labels))]

for i in range(len(var_labels)):
    ax.plot([beta[i] - t_crit*se_beta[i], beta[i] + t_crit*se_beta[i]],
            [y_pos[i], y_pos[i]], color=colors_coef[i], linewidth=2.5, solid_capstyle='round')
    ax.plot(beta[i], y_pos[i], 'o', color=colors_coef[i], markersize=8,
            markeredgecolor='white', markeredgewidth=1.5, zorder=5)

ax.axvline(0, color='#999999', linestyle='--', linewidth=1, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(var_labels, fontsize=10)
ax.set_xlabel('Coefficient Value (95% CI)', fontsize=11)
ax.set_title('Regression Coefficients with 95% Confidence Intervals', fontsize=13, fontweight='bold')
style_ax(ax)

# Add significance annotations
for i in range(len(var_labels)):
    p_val = 2 * (1 - stats.t.cdf(np.abs(beta[i]/se_beta[i]), n - len(PREDICTORS) - 1))
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax.text(beta[i] + t_crit*se_beta[i] + 0.5, y_pos[i], sig, va='center', fontsize=9, color='#333333')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig10_coefficient_forest.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Radar / Spider Chart (Seasonal Profile)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 11: Seasonal Radar Profile...")

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Normalize each variable to [0, 1] for radar
categories = [COL_NAMES_SHORT[c] for c in NUMERIC_COLS]
N_cat = len(categories)
angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

for season in SEASON_ORDER:
    values = []
    season_data = df[df['season'] == season]
    for col in NUMERIC_COLS:
        # Normalize to 0-1
        val = (season_data[col].mean() - df[col].min()) / (df[col].max() - df[col].min())
        values.append(val)
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=f'{season} ({SEASON_NAMES[season]})',
            color=SEASON_COLORS[season], markersize=5)
    ax.fill(angles, values, alpha=0.15, color=SEASON_COLORS[season])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_title('Seasonal Multivariate Profile\n(Normalized Means)', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax.set_rlabel_position(30)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig11_seasonal_radar.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Partial Regression (Added Variable) Plots
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 12: Partial Regression Plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Added Variable (Partial Regression) Plots', fontsize=14, fontweight='bold', y=0.98)

for idx, pred in enumerate(PREDICTORS):
    ax = axes[idx // 2][idx % 2]

    # Regress y on all X except pred_i, get residuals
    other_preds = [PREDICTORS[j] for j in range(len(PREDICTORS)) if j != idx]
    X_other = np.column_stack([np.ones(n), df[other_preds].values])
    beta_y_other = np.linalg.lstsq(X_other, y, rcond=None)[0]
    resid_y = y - X_other @ beta_y_other

    # Regress pred_i on all other X, get residuals
    X_pred_other = np.column_stack([np.ones(n), df[other_preds].values])
    beta_pred_other = np.linalg.lstsq(X_pred_other, df[pred].values, rcond=None)[0]
    resid_pred = df[pred].values - X_pred_other @ beta_pred_other

    # Scatter of residualized variables
    ax.scatter(resid_pred, resid_y, c=SEASON_COLORS[df['season'].iloc[0]], alpha=0, s=1)
    for season in SEASON_ORDER:
        mask = df['season'] == season
        mask_idx = mask.values
        ax.scatter(resid_pred[mask_idx], resid_y[mask_idx],
                   c=SEASON_COLORS[season], alpha=0.6, s=25,
                   edgecolors='white', linewidth=0.3, label=SEASON_NAMES[season])

    # Fit line through partial regression
    slope = np.sum(resid_pred * resid_y) / np.sum(resid_pred**2)
    x_line = np.linspace(resid_pred.min(), resid_pred.max(), 100)
    ax.plot(x_line, slope * x_line, color='#E74C3C', linewidth=1.5, linestyle='--')

    # Partial correlation
    partial_r = np.corrcoef(resid_pred, resid_y)[0, 1]
    ax.text(0.05, 0.95, f'Partial r = {partial_r:.3f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    ax.set_xlabel(f'{COL_NAMES.get(pred, pred)} (adjusted)', fontsize=10)
    ax.set_ylabel('PM2.5 (adjusted)', fontsize=10)
    ax.set_title(f'({chr(97+idx)}) {COL_NAMES.get(pred, pred)}', fontsize=11, fontweight='bold')
    style_ax(ax)

axes[0][0].legend(fontsize=8, loc='lower right')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, 'fig12_partial_regression.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Observed vs Predicted with Season Coloring
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 13: Observed vs Predicted...")

fig, ax = plt.subplots(figsize=(7, 7))

for season in SEASON_ORDER:
    mask = df['season'] == season
    ax.scatter(y[mask.values], y_pred[mask.values],
               c=SEASON_COLORS[season], label=f'{season} ({SEASON_NAMES[season]})',
               alpha=0.7, s=40, edgecolors='white', linewidth=0.5, zorder=3)

# Perfect prediction line
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.2, alpha=0.6, label='Perfect Prediction')

# R² annotation
SS_tot = np.sum((y - y.mean())**2)
R2 = 1 - np.sum(residuals**2) / SS_tot
ax.text(0.05, 0.95, f'R² = {R2:.4f}\nAdjusted R² = {1 - (1-R2)*(n-1)/(n-len(PREDICTORS)-1):.4f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', alpha=0.9, edgecolor='#CCCCCC'))

ax.set_xlabel('Observed PM2.5 (μg/m³)', fontsize=11)
ax.set_ylabel('Predicted PM2.5 (μg/m³)', fontsize=11)
ax.set_title('Observed vs. Predicted PM2.5 Values', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
style_ax(ax)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig13_observed_vs_predicted.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Enhanced Correlation Matrix with Density & Significance
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 14: Enhanced Correlation Matrix...")

fig, axes = plt.subplots(len(NUMERIC_COLS), len(NUMERIC_COLS), figsize=(12, 12))

for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        ax = axes[i][j]
        ax.clear()

        if i == j:
            # Diagonal: distribution
            data = df[NUMERIC_COLS[i]]
            ax.hist(data, bins=12, color=PALETTE_MAIN[i], alpha=0.6, density=True, edgecolor='white')
            kde_x = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color=PALETTE_MAIN[i], linewidth=1.5)
            ax.set_ylabel('Density', fontsize=7)
            ax.text(0.5, 0.9, COL_NAMES_SHORT[NUMERIC_COLS[i]],
                    transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold')
        elif i < j:
            # Upper triangle: correlation coefficient
            r, p = stats.pearsonr(df[NUMERIC_COLS[i]], df[NUMERIC_COLS[j]])
            bg_color = plt.cm.RdBu_r((r + 1) / 2)
            ax.set_facecolor(bg_color)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])

            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            fontsize = 14 if abs(r) > 0.5 else 12
            fontweight = 'bold' if abs(r) > 0.5 else 'normal'
            ax.text(0.5, 0.55, f'{r:.3f}', ha='center', va='center',
                    fontsize=fontsize, fontweight=fontweight, color='#1a1a1a',
                    transform=ax.transAxes)
            if stars:
                ax.text(0.5, 0.35, stars, ha='center', va='center',
                        fontsize=10, color='#1a1a1a', transform=ax.transAxes)
        else:
            # Lower triangle: scatter plot
            ax.scatter(df[NUMERIC_COLS[j]], df[NUMERIC_COLS[i]],
                       c='#4C72B0', alpha=0.5, s=15, edgecolors='white', linewidth=0.2)
            # Regression line
            z = np.polyfit(df[NUMERIC_COLS[j]], df[NUMERIC_COLS[i]], 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(df[NUMERIC_COLS[j]].min(), df[NUMERIC_COLS[j]].max(), 100)
            ax.plot(x_range, p_line(x_range), color='#E74C3C', linewidth=1, linestyle='--')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(labelsize=7)

        if i == len(NUMERIC_COLS) - 1:
            ax.set_xlabel(COL_NAMES_SHORT[NUMERIC_COLS[j]], fontsize=8)
        else:
            ax.set_xticklabels([])
        if j == 0 and i != 0:
            ax.set_ylabel(COL_NAMES_SHORT[NUMERIC_COLS[i]], fontsize=8)
        elif j != 0:
            ax.set_yticklabels([])

fig.suptitle('Enhanced Correlation Matrix\n(Diagonal: Distributions | Lower: Scatter | Upper: r values)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig14_enhanced_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Seasonal PM2.5 Kernel Density Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 15: Seasonal PM2.5 KDE Comparison...")

fig, ax = plt.subplots(figsize=(9, 5))

for season in SEASON_ORDER:
    data = df[df['season'] == season]['pm25']
    kde_x = np.linspace(0, 180, 200)
    kde = stats.gaussian_kde(data)
    ax.fill_between(kde_x, kde(kde_x), alpha=0.2, color=SEASON_COLORS[season])
    ax.plot(kde_x, kde(kde_x), color=SEASON_COLORS[season], linewidth=2,
            label=f'{season} ({SEASON_NAMES[season]}): μ={data.mean():.1f}, σ={data.std():.1f}')

ax.axvline(25, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.8, label='WHO Guideline (25 μg/m³)')
ax.set_xlabel('PM2.5 (μg/m³)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Seasonal PM2.5 Kernel Density Estimation', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
style_ax(ax)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig15_seasonal_kde.png'), dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 16: Model Performance Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 16: Model Performance Dashboard...")

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 0.8], wspace=0.35)

# Panel A: Actual vs Predicted by Season
ax1 = fig.add_subplot(gs[0])
for season in SEASON_ORDER:
    mask = df['season'] == season
    idx = mask.values
    ax1.scatter(y[idx], y_pred[idx], c=SEASON_COLORS[season],
                alpha=0.7, s=35, edgecolors='white', linewidth=0.3,
                label=f'{season} ({SEASON_NAMES[season]})')
ax1.plot([min(y), max(y)], [min(y), max(y)], 'k--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Observed PM2.5', fontsize=10)
ax1.set_ylabel('Predicted PM2.5', fontsize=10)
ax1.set_title('(a) Prediction Accuracy', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
style_ax(ax1)

# Panel B: Residual Distribution
ax2 = fig.add_subplot(gs[1])
ax2.hist(residuals, bins=15, color='#4C72B0', alpha=0.6, density=True, edgecolor='white')
kde_x = np.linspace(residuals.min(), residuals.max(), 100)
kde = stats.gaussian_kde(residuals)
ax2.plot(kde_x, kde(kde_x), color='#E74C3C', linewidth=1.5)
ax2.axvline(0, color='#333333', linestyle='--', linewidth=1)
ax2.set_xlabel('Residuals', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('(b) Residual Distribution', fontsize=11, fontweight='bold')
style_ax(ax2)

# Panel C: Variable Importance
ax3 = fig.add_subplot(gs[2])
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
y_std = (y - y.mean()) / y.std()
beta_std = np.linalg.lstsq(X_std, y_std, rcond=None)[0]
sorted_idx = np.argsort(np.abs(beta_std))
colors_bar = [PALETTE_MAIN[i+1] for i in sorted_idx]
ax3.barh(range(len(PREDICTORS)), np.abs(beta_std[sorted_idx]),
         color=colors_bar, alpha=0.8, edgecolor='#333333', linewidth=0.5, height=0.5)
ax3.set_yticks(range(len(PREDICTORS)))
ax3.set_yticklabels([COL_NAMES_SHORT[PREDICTORS[i]] for i in sorted_idx], fontsize=9)
ax3.set_xlabel('|Standardized β|', fontsize=10)
ax3.set_title('(c) Variable Importance', fontsize=11, fontweight='bold')
style_ax(ax3)

fig.suptitle('Multiple Regression Model Performance Summary', fontsize=14, fontweight='bold', y=1.02)
plt.savefig(os.path.join(FIG_DIR, 'fig16_model_dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ All advanced figures (7-16) saved to outputs/figures/")
for i in range(7, 17):
    prefix = {
        7: 'raincloud_pm25_season', 8: 'pairplot', 9: 'regression_diagnostics',
        10: 'coefficient_forest', 11: 'seasonal_radar', 12: 'partial_regression',
        13: 'observed_vs_predicted', 14: 'enhanced_correlation',
        15: 'seasonal_kde', 16: 'model_dashboard'
    }[i]
    print(f"  fig{i}_{prefix}.png")
