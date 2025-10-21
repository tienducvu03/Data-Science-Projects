# E-Commerce Payment Value Prediction - Brazil

Machine learning regression project to predict payment values for e-commerce transactions using customer payment behavior and transaction characteristics.

## Project Overview

This project analyzes Brazilian e-commerce payment data to predict transaction values. The goal is to understand payment patterns and build models that can forecast payment amounts based on payment method, installment plans, and transaction sequences. This information can help with revenue forecasting, fraud detection, and customer segmentation.

## Dataset

**Source:** `/content/Brazil Ecommerce.xlsx`

**Size:** 103,886 payment transactions

**Target Variable:**
- **payment_value:** Transaction amount in Brazilian Real (BRL)
  - Range: 0 to 13,664.08 BRL
  - Mean: 154.10 BRL
  - Median: 100.00 BRL
  - Highly right-skewed distribution

**Features:** 4 predictor variables

### Transaction Identifiers
- **order_id:** Unique order identifier (99,440 unique values)
  - Note: Some orders have multiple payment records

### Payment Characteristics
- **payment_sequential:** Payment sequence number (1-29)
  - Indicates split payments or payment order
  - Mean: 1.09 (most orders have single payment)

- **payment_type:** Payment method (categorical)
  - credit_card: 76,795 (73.9%)
  - boleto: 19,784 (19.0%) - Brazilian payment slip
  - voucher: 5,775 (5.6%)
  - debit_card: 1,529 (1.5%)
  - not_defined: 3 (0.003%)

- **payment_installments:** Number of installments (0-24)
  - Mean: 2.85 installments
  - Mode: 1 (single payment)
  - Reflects Brazilian installment payment culture

**Correlation Analysis:**
- payment_installments ↔ payment_value: 0.33 (moderate positive)
- payment_sequential ↔ payment_value: -0.07 (weak negative)
- payment_sequential ↔ payment_installments: -0.09 (weak negative)

## Methodology

### 1. Data Exploration

**Initial Analysis:**
- No missing values detected
- All features numerical or categorical
- No data quality issues in cleaned dataset
- Descriptive statistics calculated for all numerical features
- Correlation matrix computed

### 2. Data Preprocessing

**Feature Engineering:**
- Dropped 'order_id' (high cardinality, customer identifier)
- One-hot encoded 'payment_type' (5 categories → 5 binary features)
- Kept numerical features: payment_sequential, payment_installments

**Pipeline Construction:**
- ColumnTransformer for automated preprocessing
- OneHotEncoder with handle_unknown='ignore' for new categories

**Data Splitting:**
- Training set: 83,108 samples (80%)
- Test set: 20,778 samples (20%)
- Final feature count: 7 features (2 numerical + 5 one-hot encoded)

### 3. Model Selection

Six regression algorithms evaluated:

1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization
4. **Decision Tree Regressor** - Non-linear patterns
5. **Random Forest Regressor** - Ensemble of trees
6. **Gradient Boosting Regressor** - Boosted ensemble

### 4. Hyperparameter Tuning

**RandomizedSearchCV** with 50 iterations and 5-fold CV:

**Random Forest Regressor:**
- n_estimators: [100, 200, 300, 400, 500]
- max_features: ['auto', 'sqrt', 'log2']
- max_depth: [10, 20, 30, 40, 50, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- bootstrap: [True, False]

**Gradient Boosting Regressor:**
- n_estimators: [100, 200, 300, 400, 500]
- learning_rate: [0.01, 0.05, 0.1, 0.15, 0.2]
- max_depth: [3, 4, 5, 6, 7]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- subsample: [0.7, 0.8, 0.9, 1.0]
- max_features: ['auto', 'sqrt', 'log2', None]

**Scoring Metric:** Negative Mean Squared Error

### 5. Evaluation Metrics
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (R-squared / Coefficient of Determination)

## Results

### Best Hyperparameters

**Random Forest:**
```python
{
    'n_estimators': 500,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'max_depth': None,
    'bootstrap': False
}
```

**Gradient Boosting:**
```python
{
    'subsample': 1.0,
    'n_estimators': 100,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': None,
    'max_depth': 3,
    'learning_rate': 0.15
}
```

### Model Performance

| Model | MSE | RMSE | R² Score |
|-------|-----|------|----------|
| **Tuned Random Forest** | 36,378.17 | 190.73 | 0.14 |
| **Tuned Gradient Boosting** | 36,377.59 | 190.73 | 0.14 |

**Performance Analysis:**
- Both models show nearly identical performance
- R² = 0.14 indicates models explain only 14% of variance
- RMSE = 190.73 BRL on mean value of 154.10 BRL
- High prediction error relative to target variable range

## Key Insights

### Model Performance Analysis

**Low R² Score (0.14) Interpretation:**
1. **Limited Feature Set:** Only 4 features available for prediction
2. **Missing Information:** Critical predictors not in dataset:
   - Product category/price
   - Customer demographics
   - Order history
   - Promotion/discount information
   - Seller information
   - Geographic location

3. **Payment Value Variability:** Transaction amounts vary widely (0-13,664 BRL)
4. **Payment Method Limitation:** Payment type alone insufficient to predict value

### Feature Correlation Insights

**Positive Correlation (0.33):** payment_installments ↔ payment_value
- Higher-value purchases tend to use more installments
- Brazilian installment payment culture for expensive items
- Customers spread larger payments over time

**Weak Negative Correlation (-0.07):** payment_sequential ↔ payment_value
- Multiple payment sequences slightly associated with lower values
- Split payments may indicate budget constraints

### Business Implications

**Current Model Limitations:**
- Not suitable for accurate payment value prediction
- Requires additional features for practical use
- Better suited for payment method classification than value prediction

**Potential Applications:**
1. **Payment Method Recommendation:** Suggest installment plans based on amount
2. **Fraud Detection:** Flag unusual payment_value given installments
3. **Customer Segmentation:** Group customers by payment behavior
4. **Revenue Estimation:** Broad category forecasting (not precise predictions)

## Visualizations

The notebook includes:
- **Model Comparison Bar Chart:** MSE and R² scores side-by-side
- **Performance Metrics:** Visual comparison of Random Forest vs. Gradient Boosting

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
```

Note: `openpyxl` required for reading Excel files.

### Execution Steps
1. Open `e_commerce.ipynb` in Jupyter Notebook or Google Colab
2. Upload dataset: `/content/Brazil Ecommerce.xlsx`
3. Run all cells in sequence
4. Review exploratory analysis, model training, and evaluation

### Google Colab
Click the "Open in Colab" badge at the top of the notebook.

## Recommendations for Improvement

### Data Enhancement

**Critical Missing Features:**
1. **Product Information:**
   - Product category (electronics, clothing, etc.)
   - Product price/value
   - Product weight/size
   - Number of items in order

2. **Customer Data:**
   - Customer location (city, state)
   - Customer lifetime value
   - Purchase frequency
   - Customer account age

3. **Order Details:**
   - Shipping cost
   - Discount amount
   - Order timestamp (day, month, seasonality)
   - Review scores

4. **Seller Information:**
   - Seller location
   - Seller rating
   - Seller category

### Feature Engineering

**Derived Features:**
1. **Payment Ratios:**
   - Average installment amount (payment_value / payment_installments)
   - Payment frequency indicator

2. **Categorical Bins:**
   - Low/Medium/High value categories
   - Installment ranges (1, 2-5, 6-12, 13+)

3. **Temporal Features:**
   - Day of week, month, quarter
   - Holiday indicators
   - Shopping season flags

4. **Aggregate Features:**
   - Customer average transaction value
   - Typical payment method per customer
   - Order count per customer

### Model Improvements

1. **Advanced Algorithms:**
   - XGBoost, LightGBM, CatBoost
   - Neural networks for non-linear patterns
   - Stacking ensemble methods

2. **Problem Reframing:**
   - **Classification:** Predict payment value buckets instead of exact amounts
   - **Clustering:** Segment transactions into payment behavior groups
   - **Multi-task Learning:** Jointly predict value and payment method

3. **Regularization:**
   - Experiment with different regularization strengths
   - Feature selection to remove noise
   - Dimensionality reduction (PCA)

## Limitations

1. **Low Predictive Power:** R² = 0.14 is insufficient for production use
2. **Missing Context:** No product or customer information
3. **Temporal Gaps:** No time-based features or trends
4. **Geographic Data:** Brazilian regional payment patterns not captured
5. **External Factors:** Economic conditions, competitor pricing unknown

## Future Work

### Short-term
1. **Join with Product Data:** Merge with product catalog for price information
2. **Customer History:** Aggregate past transaction patterns
3. **Feature Selection:** Remove payment_sequential if not useful
4. **Error Analysis:** Investigate high-error predictions

### Long-term
1. **End-to-End E-commerce Analysis:** Combine with customer, product, seller datasets
2. **Time Series Forecasting:** Predict payment trends over time
3. **Recommendation System:** Suggest payment plans based on customer profile
4. **Fraud Detection:** Identify anomalous payment patterns
5. **A/B Testing:** Test payment option presentation strategies

## Technical Details

**Notebook:** `e_commerce.ipynb`

**Libraries:**
- pandas - Data manipulation
- numpy - Numerical operations
- scikit-learn - Machine learning
- matplotlib - Visualization
- openpyxl - Excel file reading

**Computational Resources:**
- Training time: ~15-20 minutes (RandomizedSearchCV with 50 iterations)
- Memory usage: ~500 MB RAM
- Platform: Google Colab (CPU)

## Dataset Source

**Brazilian E-commerce Public Dataset by Olist**
- Real commercial data from Brazilian marketplace
- Anonymized and aggregated
- Covers orders from 2016-2018

## Files

- `e_commerce.ipynb` - Main analysis notebook
- `Brazil Ecommerce.xlsx` - Payment dataset (not included)

## Project Structure

```
e_commerce.ipynb
├── Load Data
│   └── Read Excel file
├── Explore Data
│   ├── Data types and missing values
│   ├── Descriptive statistics
│   ├── Correlation matrix
│   └── Value counts for categorical features
├── Preprocess Data
│   ├── Drop order_id
│   ├── One-hot encode payment_type
│   └── Train-test split
├── Train Models
│   ├── Linear Regression
│   ├── Ridge Regression
│   ├── Lasso Regression
│   ├── Decision Tree
│   ├── Random Forest
│   └── Gradient Boosting
├── Hyperparameter Tuning
│   ├── RandomizedSearchCV (Random Forest)
│   └── RandomizedSearchCV (Gradient Boosting)
├── Evaluate Models
│   ├── MSE, RMSE, R² calculation
│   └── Performance comparison
└── Visualize Results
    └── Bar chart comparison
```

## Key Takeaways

1. **Feature Importance:** Payment-related features alone insufficient for value prediction
2. **Brazilian Payment Culture:** High installment usage reflects local payment preferences
3. **Model Selection:** Complex models (RF, GB) perform similarly to simpler ones (limited features)
4. **Data Requirements:** Need product, customer, and temporal data for accurate predictions
5. **Alternative Applications:** Better suited for classification or clustering than regression

## Author

Tien Duc Vu

## License

Educational purposes

## Last Updated

October 2024

---

**Conclusion:** This project demonstrates the importance of feature availability in predictive modeling. While the models are well-implemented, the low R² score highlights that payment characteristics alone cannot accurately predict transaction values. The project serves as a valuable lesson in feature engineering and data collection requirements for e-commerce analytics.
