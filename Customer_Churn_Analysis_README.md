# Customer Churn Analysis - Telecommunications

Predictive analytics project to identify customers at risk of churning from a telecommunications company.

## Project Overview

This project uses machine learning classification algorithms to predict customer churn based on customer demographics, service usage patterns, and billing information. The goal is to help the business proactively identify at-risk customers and implement retention strategies.

## Dataset

**Source:** `../content/cleaned_telco_customer_churn.csv`

**Size:** 7,043 customer records

**Features:** 21 columns including:
- **Customer Demographics:** Gender, Senior Citizen status, Partner, Dependents
- **Account Information:** Tenure, Contract type, Payment method, Paperless billing
- **Service Details:** Phone service, Multiple lines, Internet service type
- **Add-on Services:** Online security, Online backup, Device protection, Tech support
- **Streaming Services:** Streaming TV, Streaming movies
- **Billing Information:** Monthly charges, Total charges
- **Target Variable:** Churn (Yes/No)

**Data Characteristics:**
- Clean dataset with no missing values
- Mix of categorical and numerical features
- After one-hot encoding: 7,088 features
- Training set: 5,634 samples
- Test set: 1,409 samples

## Methodology

### 1. Data Preprocessing
- **Categorical Encoding:** One-hot encoding for categorical features
- **Feature Scaling:** StandardScaler for numerical features (tenure, monthly charges, total charges)
- **Pipeline Creation:** ColumnTransformer for preprocessing automation
- **Train-Test Split:** 80-20 split with random_state=42

### 2. Model Selection
Three classification algorithms were evaluated:
- **Logistic Regression** (solver='liblinear')
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

### 3. Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation for:

**Logistic Regression:**
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- solver: ['liblinear', 'lbfgs']

**Gradient Boosting:**
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.1, 0.2]
- max_depth: [3, 4, 5]
- subsample: [0.8, 1.0]
- max_features: ['sqrt', 'log2', None]

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression (Untuned)** | 0.8219 | 0.6871 | 0.6005 | 0.6409 |
| **Random Forest (Untuned)** | 0.7942 | 0.6590 | 0.4611 | 0.5426 |
| **Gradient Boosting (Untuned)** | 0.8133 | 0.6950 | 0.5255 | 0.5985 |
| **Tuned Logistic Regression** | 0.8219 | 0.6871 | 0.6005 | 0.6409 |
| **Tuned Gradient Boosting** | 0.8169 | 0.6898 | 0.5603 | 0.6183 |

### Best Model: Tuned Logistic Regression
- **Best Parameters:** {'C': 1, 'solver': 'lbfgs'}
- **Accuracy:** 82.19%
- **F1-Score:** 0.6409

### Top 10 Predictive Features (Gradient Boosting)

1. **Contract_Month-to-month** (27.50%) - Strongest churn predictor
2. **tenure** (12.87%) - Length of customer relationship
3. **TotalCharges** (6.54%) - Cumulative billing amount
4. **MonthlyCharges** (6.51%) - Current billing rate
5. **OnlineSecurity_No** (6.06%) - Lack of online security service
6. **InternetService_Fiber optic** (6.04%) - Fiber internet customers
7. **TechSupport_No** (3.82%) - Lack of tech support
8. **PaymentMethod_Electronic check** (2.66%) - Payment method
9. **MultipleLines_No** (0.94%) - Single phone line
10. **Contract_Two year** (0.87%) - Long-term contract (negative correlation)

## Key Business Insights

### High-Risk Customer Profile
1. **Contract Type:** Month-to-month contracts show highest churn risk
2. **Customer Tenure:** New customers (low tenure) are more likely to churn
3. **Service Add-ons:** Customers without Online Security and Tech Support are at higher risk
4. **Internet Type:** Fiber optic internet customers churn more frequently
5. **Billing:** Higher monthly charges correlate with increased churn

### Recommendations
1. **Retention Strategy:** Focus on month-to-month contract customers
2. **Early Intervention:** Target customers in first 12 months
3. **Value-Add Services:** Promote Online Security and Tech Support packages
4. **Pricing Review:** Analyze fiber optic pricing and service quality
5. **Long-term Incentives:** Encourage migration to 1-year or 2-year contracts

## Visualizations

The notebook includes:
- **Feature Importance Bar Charts** (Gradient Boosting & Logistic Regression)
- **Model Performance Comparison**
- Feature importance filtering (excluding customerID features for interpretability)

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Execution Steps
1. Open the notebook in Jupyter or Google Colab
2. Ensure the dataset path is correct: `../content/cleaned_telco_customer_churn.csv`
3. Run all cells sequentially
4. Review model evaluation results and visualizations

### Google Colab
Click the "Open in Colab" badge at the top of the notebook for cloud execution.

## Model Deployment Considerations

### Next Steps for Production
1. **Model Serialization:** Save trained model with joblib/pickle
2. **API Development:** Create REST API for real-time predictions
3. **Monitoring:** Track model performance over time
4. **Retraining Pipeline:** Automate model updates with new data
5. **A/B Testing:** Test retention strategies on predicted high-risk customers

### Feature Engineering Ideas
- Customer lifetime value (CLV) calculation
- Service usage trends (increasing/decreasing)
- Payment history (late payments, disputes)
- Customer service interaction frequency
- Competitive pricing analysis

## Limitations

1. **Class Imbalance:** Dataset may have unbalanced churn/non-churn ratio
2. **Customer ID Overfitting:** One-hot encoding of customer IDs created sparse features
3. **Temporal Patterns:** No time-series analysis of customer behavior
4. **External Factors:** Competitor actions and market conditions not included
5. **Feature Engineering:** Limited derived features from raw data

## Future Improvements

1. **Advanced Models:** XGBoost, LightGBM, Neural Networks
2. **Class Balancing:** SMOTE or class weights for imbalanced data
3. **Feature Selection:** Remove customer ID features, add domain knowledge
4. **Ensemble Methods:** Stack multiple models for improved predictions
5. **Explainability:** SHAP values for individual prediction explanations
6. **Cost-Sensitive Learning:** Incorporate business costs of false positives/negatives

## Technical Details

**Notebook:** `Customer_churn_analysis.ipynb`

**Libraries Used:**
- pandas - Data manipulation
- numpy - Numerical operations
- scikit-learn - Machine learning algorithms
- matplotlib - Visualizations

**Computational Resources:**
- Training time: ~2-5 minutes (on Google Colab)
- Memory usage: < 1 GB RAM

## Files

- `Customer_churn_analysis.ipynb` - Main analysis notebook
- `cleaned_telco_customer_churn.csv` - Dataset (not included in repo)

## Author

Tien Duc Vu

## Last Updated

October 2024
