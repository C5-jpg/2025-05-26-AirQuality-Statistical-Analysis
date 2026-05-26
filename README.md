# 2025-05-26 Air Quality Statistical Analysis

> **Probability and Mathematical Statistics (2025-2026 Spring)** | Instructor: Prof. Liu Liqun
>
> A comprehensive statistical analysis and visualization project examining the relationship between PM2.5 air quality and meteorological/traffic factors, featuring publication-quality figures following Nature/Science journal guidelines.

---

## Project Overview

This project applies **descriptive statistics, inferential testing, correlation analysis, and multiple linear regression** to an air quality dataset containing 65 observations across four seasons. The goal is to identify key drivers of PM2.5 concentration and quantify their effects through rigorous statistical methods.

### Research Questions

1. What are the distributional characteristics of PM2.5 and its predictors across seasons?
2. Which meteorological and traffic factors significantly correlate with PM2.5 levels?
3. Are there statistically significant seasonal differences in air quality?
4. How well can a multiple regression model predict PM2.5 from available predictors?

---

## Dataset Description

| Variable | Description | Unit | Range |
|---|---|---|---|
| `pm25` | PM2.5 concentration (target) | μg/m³ | 5.0 – 154.4 |
| `temperature` | Ambient temperature | °C | -10.8 – 32.5 |
| `humidity` | Relative humidity | % | 20.0 – 89.7 |
| `wind_speed` | Wind speed | m/s | 0.5 – 7.78 |
| `traffic_index` | Traffic congestion index | — | 40 – 120 |
| `season` | Season (春/夏/秋/冬) | — | 4 categories |

- **65 observations**, no missing values
- Seasonal distribution: Spring (20), Summer (12), Autumn (19), Winter (14)

---

## Methodology

### 1. Exploratory Data Analysis
- Descriptive statistics (mean, SD, median, IQR, skewness, kurtosis)
- Distributional assessment via histograms with KDE overlay
- Outlier detection using the IQR method (1.5×IQR rule)
- Seasonal decomposition and comparison

### 2. Normality Testing
- **Shapiro-Wilk test** (primary, n < 50 per group)
- **D'Agostino-Pearson test** (supplementary)

### 3. Correlation Analysis
- **Pearson product-moment correlation** (linear relationships)
- **Spearman rank correlation** (monotonic relationships, robust to non-normality)
- Significance testing with p-values and R² interpretation

### 4. Seasonal Difference Analysis
- **One-way ANOVA** (parametric, with η² effect size)
- **Kruskal-Wallis H test** (non-parametric alternative)
- **Post-hoc pairwise comparisons** with Bonferroni correction
- **Welch's t-test** and **Mann-Whitney U** for individual pairs
- **Cohen's d** effect sizes for all pairwise contrasts
- **Levene's test** for homogeneity of variance

### 5. Multiple Linear Regression
- OLS estimation: PM2.5 = β₀ + β₁·Temp + β₂·Humidity + β₃·Wind + β₄·Traffic
- Model fit: R², Adjusted R², F-test, AIC, BIC, RMSE
- **Variance Inflation Factor (VIF)** for multicollinearity diagnostics
- **Standardized coefficients** for variable importance ranking
- **Residual diagnostics**: Shapiro-Wilk on residuals, Durbin-Watson, Breusch-Pagan

---

## Key Findings

### Descriptive Statistics
| Variable | Mean | SD | Skewness | Normality |
|---|---|---|---|---|
| PM2.5 | 49.47 | 38.76 | 1.077 | **Non-normal** (p < 0.001) |
| Temperature | 16.26 | 9.23 | -0.584 | Normal (p = 0.083) |
| Humidity | 57.30 | 15.29 | -0.107 | Normal (p = 0.789) |
| Wind Speed | 3.67 | 1.75 | 0.019 | Normal (p = 0.493) |
| Traffic Index | 70.34 | 16.71 | 0.630 | Normal (p = 0.089) |

### Seasonal PM2.5 (μg/m³)
| Season | N | Mean | SD | Median |
|---|---|---|---|---|
| Spring (春) | 20 | 32.92 | 21.13 | 34.50 |
| Summer (夏) | 12 | 17.77 | 9.92 | 19.55 |
| Autumn (秋) | 19 | 42.64 | 19.80 | 47.00 |
| **Winter (冬)** | **14** | **109.57** | **28.48** | **105.45** |

### Correlation with PM2.5
| Predictor | Pearson r | p-value | R² | Significance |
|---|---|---|---|---|
| Temperature | **-0.756** | < 0.001 | 57.1% | *** |
| Wind Speed | **-0.348** | 0.004 | 12.1% | ** |
| Humidity | 0.248 | 0.046 | 6.2% | * |
| Traffic Index | 0.209 | 0.095 | 4.4% | n.s. |

### ANOVA Results
- **F(3, 61) = 51.64, p < 0.001** — Highly significant seasonal differences
- **Effect size: η² = 0.72** — Large effect
- Kruskal-Wallis: **H = 39.12, p < 0.001** (non-parametric confirmation)

### Multiple Regression Model
```
PM2.5 = 74.44 - 3.09·Temperature + 0.39·Humidity - 7.28·WindSpeed + 0.42·Traffic
```

| Metric | Value |
|---|---|
| **R²** | **0.7452** |
| **Adjusted R²** | **0.7282** |
| F(4, 60) | 43.87 (p < 0.001) |
| RMSE | 20.21 μg/m³ |

| Predictor | β | SE | t | p | Significance |
|---|---|---|---|---|---|
| Intercept | 74.44 | 16.60 | 4.48 | < 0.001 | *** |
| Temperature | -3.09 | 0.27 | -11.24 | < 0.001 | *** |
| Humidity | 0.39 | 0.17 | 2.36 | 0.021 | * |
| Wind Speed | -7.28 | 1.46 | -4.98 | < 0.001 | *** |
| Traffic Index | 0.42 | 0.15 | 2.75 | 0.008 | ** |

### Model Diagnostics — All Passed
- Residual normality: Shapiro-Wilk W = 0.972, p = 0.148 ✓
- Autocorrelation: Durbin-Watson = 2.02 ✓
- Homoscedasticity: Breusch-Pagan p = 0.520 ✓
- Multicollinearity: All VIF < 1.1 ✓

### Variable Importance (Standardized β)
1. Temperature: |β| = 0.735 (dominant predictor)
2. Wind Speed: |β| = 0.328
3. Traffic Index: |β| = 0.179
4. Humidity: |β| = 0.156

---

## Visualization Gallery

### Basic Visualizations
| Figure | Description |
|---|---|
| **Fig 1** | Distribution of all variables (histogram + KDE + mean/median) |
| **Fig 2** | Seasonal box plots for all variables |
| **Fig 3** | Pearson correlation heatmap with significance stars |
| **Fig 4** | Scatter plots: PM2.5 vs each predictor with regression lines |
| **Fig 5** | Seasonal mean comparison with 95% confidence intervals |
| **Fig 6** | PM2.5 ranked distribution with WHO guideline |

### Advanced Visualizations
| Figure | Description |
|---|---|
| **Fig 7** | Raincloud plots — PM2.5 seasonal distribution with pairwise comparisons |
| **Fig 8** | Pair plot — full multivariate scatter matrix |
| **Fig 9** | Regression diagnostic panel (residuals, Q-Q, scale-location, Cook's D) |
| **Fig 10** | Coefficient forest plot with 95% CI |
| **Fig 11** | Seasonal radar profile (normalized) |
| **Fig 12** | Partial regression (added variable) plots |
| **Fig 13** | Observed vs. predicted PM2.5 |
| **Fig 14** | Enhanced correlation matrix (distribution + scatter + r-values) |
| **Fig 15** | Seasonal PM2.5 kernel density estimation |
| **Fig 16** | Model performance summary dashboard |

---

## Project Structure

```
├── README.md
├── 数据集3_空气质量.xlsx               # Raw dataset
├── scripts/
│   ├── config.py                       # Visualization config (Nature style)
│   ├── 01_eda_basic_stats.py           # EDA & descriptive statistics
│   ├── 02_basic_visualization.py       # Basic visualization suite (Fig 1-6)
│   ├── 03_statistical_tests.py         # Statistical tests & regression
│   └── 04_advanced_visualization.py    # Advanced visualizations (Fig 7-16)
└── outputs/
    ├── figures/                        # 16 publication-quality PNG figures
    │   ├── fig1_distributions.png
    │   ├── fig2_seasonal_boxplots.png
    │   ├── fig3_correlation_heatmap.png
    │   ├── fig4_scatter_pm25_predictors.png
    │   ├── fig5_seasonal_means.png
    │   ├── fig6_pm25_ranked.png
    │   ├── fig7_raincloud_pm25_season.png
    │   ├── fig8_pairplot.png
    │   ├── fig9_regression_diagnostics.png
    │   ├── fig10_coefficient_forest.png
    │   ├── fig11_seasonal_radar.png
    │   ├── fig12_partial_regression.png
    │   ├── fig13_observed_vs_predicted.png
    │   ├── fig14_enhanced_correlation.png
    │   ├── fig15_seasonal_kde.png
    │   └── fig16_model_dashboard.png
    ├── tables/                         # Statistical result tables (CSV)
    │   ├── descriptive_statistics.csv
    │   ├── seasonal_pm25.csv
    │   ├── pearson_correlation.csv
    │   ├── pearson_pvalues.csv
    │   ├── spearman_correlation.csv
    │   ├── normality_tests.csv
    │   ├── normality_detailed.csv
    │   ├── correlation_analysis.csv
    │   ├── pairwise_comparisons.csv
    │   ├── regression_coefficients.csv
    │   ├── standardized_coefficients.csv
    │   └── vif_analysis.csv
    └── reports/
        └── statistical_report.txt      # Full statistical analysis report
```

---

## Technical Details

- **Python 3.x** with pandas, numpy, scipy, matplotlib, seaborn
- **Visualization style**: Nature/Science journal guidelines — sans-serif fonts, colorblind-friendly palettes, 300 DPI
- **Statistical significance levels**: * p < 0.05, ** p < 0.01, *** p < 0.001
- **Multiple comparison correction**: Bonferroni adjustment
- All regression assumptions verified (normality, homoscedasticity, independence, no multicollinearity)

---

## Conclusion

1. **Temperature is the dominant predictor** of PM2.5 (|β_std| = 0.735), explaining 57.1% of variance alone
2. **Winter shows dramatically higher PM2.5** levels (109.6 μg/m³) compared to other seasons (17.8–42.6 μg/m³)
3. The **multiple regression model explains 72.8% of PM2.5 variance** (Adjusted R² = 0.728) and passes all diagnostic checks
4. All four predictors show low multicollinearity (VIF < 1.1), indicating independent contributions
5. The model meets all OLS assumptions: normally distributed residuals, homoscedasticity, no autocorrelation

---

## License

This project is for educational purposes as part of the Probability and Mathematical Statistics course presentation.

## Author

Student presentation project — Probability and Mathematical Statistics (2025-2026 Spring)
