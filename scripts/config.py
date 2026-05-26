"""
Publication-quality visualization configuration
Adopts Nature/Science journal style guidelines
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ─── Global Font & Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    # Font settings - use sans-serif for clarity
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,

    # Axes
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'axes.unicode_minus': False,

    # Ticks
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 4,
    'ytick.major.size': 4,

    # Lines
    'lines.linewidth': 1.2,
    'lines.markersize': 4,

    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',

    # Save
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # Figure
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ─── Color Palettes (Nature-inspired) ────────────────────────────────────────
# Main palette - colorblind-friendly, Nature-style
PALETTE_MAIN = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

# Seasonal palette
SEASON_COLORS = {
    '春': '#2ECC71',   # Spring - fresh green
    '夏': '#E74C3C',   # Summer - warm red
    '秋': '#F39C12',   # Autumn - golden orange
    '冬': '#3498DB',   # Winter - cool blue
}

SEASON_ORDER = ['春', '夏', '秋', '冬']

# Diverging palette for correlation
CORR_CMAP = 'RdBu_r'

# Sequential palette
SEQUENTIAL_CMAP = 'YlOrRd'

# ─── Figure Size Presets ─────────────────────────────────────────────────────
FIG_SINGLE = (6, 4)        # Single panel
FIG_WIDE = (10, 4)         # Wide single panel
FIG_DOUBLE = (10, 8)       # Double panel
FIG_TRIPLE = (12, 4)       # Triple horizontal
FIG_QUAD = (10, 10)        # 2x2 grid
FIG_LARGE = (14, 10)       # Large multi-panel

# ─── Column Display Names ────────────────────────────────────────────────────
COL_NAMES = {
    'pm25': 'PM2.5 (μg/m³)',
    'temperature': 'Temperature (°C)',
    'humidity': 'Humidity (%)',
    'wind_speed': 'Wind Speed (m/s)',
    'traffic_index': 'Traffic Index',
}

COL_NAMES_SHORT = {
    'pm25': 'PM2.5',
    'temperature': 'Temp',
    'humidity': 'Humidity',
    'wind_speed': 'Wind',
    'traffic_index': 'Traffic',
}

SEASON_NAMES = {
    '春': 'Spring',
    '夏': 'Summer',
    '秋': 'Autumn',
    '冬': 'Winter',
}

NUMERIC_COLS = ['pm25', 'temperature', 'humidity', 'wind_speed', 'traffic_index']
TARGET = 'pm25'
PREDICTORS = ['temperature', 'humidity', 'wind_speed', 'traffic_index']
