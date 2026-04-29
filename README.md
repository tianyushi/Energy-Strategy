# WTI Oil Price Analysis — Feature Importance & Professional Trading Framework

A comprehensive analysis of WTI (West Texas Intermediate) crude oil swap prices, implementing professional trading features and statistical analysis to understand what drives oil price movements.

## 🎯 Project Overview

This project analyzes **3.7 million WTI swap price records** (2012-2026) to identify key drivers of oil price movements using:
- **56 engineered features** including 10 Tier 1 professional trading features
- **PCA decomposition** of the forward curve (Level, Slope, Curvature)
- **Statistical feature importance** analysis (correlation, mutual information)
- **Professional pricing framework** (Roll Yield, Carry, Convenience Yield)

**Goal**: Understand what factors influence WTI oil prices using proper statistical methods (not ML prediction).

---

## 📊 Key Findings

### **Top Drivers of Oil Prices**

1. **Volatility** (#1 driver)
   - Volatility_5: Highest mutual information score (0.153)
   - Recent price volatility is the strongest predictor

2. **Forward Curve Structure**
   - Spread_M1_M2: Strongest correlation (-0.192)
   - Curve slope and contango/backwardation matter

3. **Momentum & Technical Indicators**
   - Price_to_MA5, RSI_14, Momentum features
   - Short-term momentum drives returns

4. **PCA Components**
   - PC1 (Level): Explains 97.6% of curve variance
   - PC2 (Slope): Explains 2.3% (contango/backwardation)
   - PC3 (Curvature): Explains 0.1% (butterfly shapes)

### **Coverage**
- **Current**: ~60% of professional feature stack
- **Implemented**: Core pricing framework, PCA, convenience yield
- **Missing**: Fundamental data (EIA inventory, OPEC), macro factors, seasonality

---

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Configure Snowflake Connection**

Copy `.env.example` to `.env` and add your credentials:
```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
```

Register your public key on Snowflake:
```sql
ALTER USER <username> SET RSA_PUBLIC_KEY='<base64-body>';
```

Verify connection:
```bash
python -m src.verify_key
python -m src.snowflake_client
```

### 3. **Run Analysis**

Open the main notebook:
```bash
jupyter notebook notebooks/wti_swap_preprocessing_and_eda.ipynb
```

The notebook contains:
1. **Data Preprocessing** - Clean and prepare WTI swap data
2. **Exploratory Data Analysis** - Understand data structure and patterns
3. **Feature Engineering** (hidden) - Create 56 features including Tier 1
4. **Feature Importance Analysis** - Statistical analysis of price drivers

---

## 📁 Project Structure

```
Energy-Strategy/
├── notebooks/                  # Analysis notebooks
│   └── wti_swap_preprocessing_and_eda.ipynb  # Main analysis
│
├── src/                        # Source code
│   ├── snowflake_client.py    # Snowflake connection (key-pair auth)
│   ├── probe_snowflake.py     # Query utilities
│   ├── decode_pricedata.py    # Price data decoding logic
│   ├── parse_description.py   # Description parsing
│   └── verify_mappings.py     # Mapping verification
│
├── data/                       # Data files
│   ├── analysis/               # Analysis-ready datasets
│   │   ├── wti_swap_only.parquet          # Raw WTI data (3.7M rows)
│   │   ├── wti_swap_cleaned.parquet       # Cleaned data
│   │   └── wti_features.parquet           # 56 features
│   ├── decoded/                # Decoded reference data
│   ├── verified/               # Verified mappings
│   ├── wti_curve_wide.parquet  # Forward curve for PCA
│   └── wti_curve_long.parquet  # Forward curve (long format)
│
├── output/                     # Analysis results
│   ├── feature_analysis_summary.txt       # Summary report
│   ├── feature_correlations.csv           # Correlation analysis
│   ├── feature_mutual_information.csv     # Mutual information
│   ├── feature_category_importance.csv    # Category analysis
│   └── feature_significance.csv           # Statistical tests
│
├── docs/                       # Documentation
│   └── tier1_features_implementation.md   # Tier 1 features guide
│
├── api_key/                    # API credentials (gitignored)
│   ├── xren_private_key.p8
│   └── xren_public_key.pub
│
├── .env                        # Environment config (gitignored)
├── .env.example                # Config template
├── energy_model.yaml           # Model configuration
├── PROJECT_STRUCTURE.md        # Detailed structure guide
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🎓 Features Implemented

### **Tier 1 Professional Features (10 features)**

#### **Phase 1: Quick Wins (5 features)**
1. **Roll_Yield**: Profit/loss from rolling futures positions
   - Formula: `(M1_Price - M2_Price) / M1_Price`
   - Positive = Backwardation (bullish), Negative = Contango (bearish)

2. **Carry**: Total return including collateral yield
   - Formula: `Roll_Yield + Risk_Free_Rate`
   - Represents full economic return

3. **Spread_M2_M3**: Short-term curve structure
   - Captures near-term supply/demand dynamics

4. **Spread_M6_M12**: Long-term curve structure
   - Reflects longer-term market expectations

5. **Butterfly_M1_M2_M3**: Curve curvature
   - Formula: `(M1 + M3 - 2*M2) / 2`
   - Detects non-linear curve shapes

#### **Phase 2: PCA Decomposition (3 features)**
6. **PCA_Level** (PC1): Overall curve level
   - Explains ~97.6% of variance
   - Captures parallel shifts

7. **PCA_Slope** (PC2): Curve slope/tilt
   - Explains ~2.3% of variance
   - Captures steepening/flattening

8. **PCA_Curvature** (PC3): Butterfly shape
   - Explains ~0.1% of variance
   - Captures complex curve shapes

#### **Phase 3: Convenience Yield (2 features)**
9. **Convenience_Yield**: Implied value of holding physical oil
   - Formula: `r + c - (1/T) * ln(F/S)`
   - High = tight supply, Low = abundant supply

10. **Carry_Deviation**: Mispricing signal
    - Deviation from cost-of-carry model
    - Mean-reversion signal

### **Additional Features (46 features)**
- Price lags (1, 2, 3, 5 days)
- Return lags (1, 2, 3, 5 days)
- Moving averages (5, 10, 20, 50 days)
- Volatility (5, 10, 20 days)
- Momentum (5, 10, 20 days)
- RSI (14 days)
- Volume and Open Interest features
- Calendar spreads (M1-M2, M1-M3, M1-M6, M1-M12)
- Temporal features (day of week, month, quarter)

**Total: 56 features**

---

## 📈 Analysis Methods

### **1. Correlation Analysis**
- Pearson correlation for linear relationships
- Top feature: Spread_M1_M2 (correlation: -0.192)

### **2. Mutual Information**
- Captures non-linear relationships
- Top feature: Volatility_5 (MI score: 0.153)

### **3. Statistical Significance**
- P-value testing for correlations
- Identifies statistically significant relationships

### **4. Category Analysis**
- Groups features by type (Volatility, Momentum, Curve, etc.)
- Shows which categories matter most

### **5. PCA Visualization**
- Time series of PCA components
- Explained variance charts
- Component loadings
- PC1 vs PC2 scatter plots

---

## 🔧 Data Pipeline

### **1. Data Extraction**
```python
from src.snowflake_client import SnowflakeClient

with SnowflakeClient() as sf:
    df = sf.read_sql("SELECT * FROM PLATTS_MDV2.PRICEDATA WHERE ...")
```

### **2. Data Decoding**
```python
from src.decode_pricedata import decode_pricedata

decoded_df = decode_pricedata(raw_df, mappings)
```

### **3. Feature Engineering**
- Automated in notebook (hidden section)
- Creates 56 features from cleaned data
- Saves to `data/analysis/wti_features.parquet`

### **4. Analysis**
- Statistical feature importance
- Correlation and mutual information
- PCA decomposition and visualization
- Results saved to `output/`

---

## 📊 Key Results

### **Feature Importance Rankings**

#### **By Mutual Information (Non-linear)**
1. Volatility_5: 0.153
2. Volatility_10: 0.098
3. Price_to_MA5: 0.097
4. Momentum_5: 0.087
5. Price_to_MA20: 0.078

#### **By Correlation (Linear)**
1. Spread_M1_M2: -0.192
2. Spread_M1_M3: -0.150
3. Curve_Slope_3M: 0.150
4. RSI_14: 0.084
5. Is_Contango: 0.082

#### **By Category**
1. **Volatility**: Highest average MI (0.103)
2. **Returns/Momentum**: Second highest (0.050)
3. **Moving Averages**: Third (0.048)
4. **Forward Curve**: Important for structure (0.025)

### **PCA Results**
- **PC1 (Level)**: 97.6% of variance - parallel shifts
- **PC2 (Slope)**: 2.3% of variance - contango/backwardation
- **PC3 (Curvature)**: 0.1% of variance - butterfly shapes

---

## 🎯 Use Cases

### **For Traders**
- Understand key drivers of oil price movements
- Identify which features to monitor
- Use PCA to understand curve dynamics
- Roll yield and carry for position management

### **For Analysts**
- Statistical feature importance (not ML black box)
- Professional pricing framework implementation
- Comprehensive EDA of WTI swap data
- Reproducible analysis pipeline

### **For Researchers**
- Clean, documented codebase
- Professional feature engineering
- Statistical methods (correlation, MI, PCA)
- Extensible framework for additional features

---

## 🔮 Future Enhancements (Tier 2+)

### **Fundamental Data**
- EIA weekly inventory reports
- OPEC production data
- Global demand indicators
- Refinery utilization

### **Macro Factors**
- USD index (DXY)
- 10-year Treasury yields
- Global GDP growth
- Interest rate expectations

### **Seasonality**
- Monthly seasonal patterns
- Refinery maintenance cycles
- Weather-driven demand
- Holiday effects

### **Advanced Models**
- Kalman filter for hidden states
- Multi-factor stochastic models
- Machine learning on curve shapes
- Regime detection

---

## 📚 Documentation

- **PROJECT_STRUCTURE.md**: Detailed project structure
- **docs/tier1_features_implementation.md**: Tier 1 features guide
- **notebooks/wti_swap_preprocessing_and_eda.ipynb**: Main analysis with inline documentation
- **output/feature_analysis_summary.txt**: Analysis results summary

---

## 🛠️ Technical Details

### **Data**
- **Source**: Snowflake (Platts MDV2)
- **Size**: 3.7M rows (WTI swaps only)
- **Date Range**: 2012-06-22 to 2026-04-28
- **Symbols**: 301 unique (XNCW000-XNCW108 + calendar codes)
- **BATE Types**: Open, High, Low, Close, Volume, Open Interest

### **Technologies**
- **Python 3.12+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: PCA, mutual information
- **Matplotlib/Seaborn**: Visualization
- **Snowflake**: Data warehouse
- **Jupyter**: Interactive analysis

### **Performance**
- Preprocessing: ~1-2 minutes
- Feature engineering: ~2-3 minutes
- PCA decomposition: ~1 minute
- Feature importance: ~3-5 minutes
- Total runtime: ~10 minutes

---

## 🤝 Contributing

This is a research project. To extend:

1. **Add new features**: Edit feature engineering section in notebook
2. **Add new data sources**: Extend `src/` with new data loaders
3. **Add new analysis**: Add cells to notebook
4. **Update documentation**: Keep README and docs/ in sync

---

## 📝 Notes

### **Important Assumptions**
- Risk-free rate: 4% annually (3-month Treasury proxy)
- Storage cost: $10/bbl annually
- M1 (XNCW001) used as primary contract (most liquid)
- Next-day return as target variable

### **Data Quality**
- No missing values in cleaned data
- No invalid prices (≤0 or >$200/bbl)
- No duplicates
- All dates are business days

### **Limitations**
- Analysis based only on price/volume data
- Missing fundamental data (inventory, production)
- Missing macro factors (USD, rates, GDP)
- Correlations are generally weak (market efficiency)

---

## 📧 Contact

For questions about this analysis or to request access to the Snowflake data, contact the project maintainer.

---

## 📄 License

This project is for research and educational purposes.

---

**Last Updated**: 2026-04-29  
**Status**: ✅ Production Ready  
**Version**: 1.0
