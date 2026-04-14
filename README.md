# Data Science Projects Portfolio

A collection of machine learning projects demonstrating various data science techniques including classification, regression, feature engineering, hyperparameter tuning, and data visualization.

## Projects Overview

### 1. Customer Churn Analysis
**File:** `Customer_churn_analysis.ipynb` | **[📖 Detailed Documentation](Customer_Churn_Analysis_README.md)**

Predicts customer churn for a telecommunications company using machine learning classification models.

**Key Features:**
- Binary classification (Churn vs. No Churn)
- Multiple model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Data preprocessing with one-hot encoding and StandardScaler

**Best Results:**
- Tuned Logistic Regression: 82.19% accuracy, F1-score: 0.6409
- Tuned Gradient Boosting: 81.69% accuracy, F1-score: 0.6183
- Top features: Contract type (Month-to-month), Tenure, Total Charges, Monthly Charges

**Business Insights:**
- Month-to-month contracts are the strongest predictor of churn
- Customers with fiber optic internet and no online security/tech support are at higher risk
- Lower tenure correlates with higher churn probability

---

### 2. Airline Customer Satisfaction Analysis
**File:** `airline_analysis.ipynb` | **[📖 Detailed Documentation](Airline_Analysis_README.md)**

Analyzes airline customer satisfaction using classification models to predict satisfied vs. dissatisfied customers.

**Key Features:**
- Binary classification (Satisfied vs. Dissatisfied)
- Comprehensive data preprocessing (imputation, encoding, scaling)
- Model comparison (Logistic Regression, Random Forest)
- GridSearchCV hyperparameter optimization
- Perfect model performance achieved

**Best Results:**
- Both models achieved 100% accuracy, precision, recall, and F1-score
- Confusion matrices show perfect classification on test set
- 25,976 test samples with 29 features

**Note:** Perfect performance may indicate potential data leakage or an overly simplistic problem requiring further investigation.

---

### 3. E-Commerce Payment Analysis
**File:** `e_commerce.ipynb` | **[📖 Detailed Documentation](E_Commerce_README.md)**

Predicts payment values for Brazilian e-commerce transactions using regression models.

**Key Features:**
- Regression task predicting payment amounts
- Multiple model comparison (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting)
- RandomizedSearchCV for efficient hyperparameter tuning
- Feature engineering with one-hot encoding for payment types

**Dataset:**
- 103,886 transactions
- Features: payment type, installments, sequential payments
- Target: payment value (range: $0 - $13,664)

**Best Results:**
- Random Forest: RMSE: 190.73, R²: 0.14
- Gradient Boosting: RMSE: 190.73, R²: 0.14
- Low R² suggests additional features needed for better predictions

---

### 4. Finance Mortgage Loan Default Prediction
**File:** `Finance_Mortgage_loan_analysist.ipynb` | **[📖 Detailed Documentation](Mortgage_Loan_Analysis_README.md)**

Predicts mortgage loan defaults using borrower financial information and loan characteristics.

**Key Features:**
- Binary classification (Default vs. No Default)
- Outlier handling using 99th percentile capping
- Feature scaling with StandardScaler
- GridSearchCV hyperparameter tuning
- Feature importance visualization

**Dataset:**
- 3,000 loan applications
- 12 features including income, assets, loan amount, property details
- Balanced target variable

**Best Results:**
- Random Forest Classifier: 99.83% accuracy, ROC AUC: 1.0000
- Logistic Regression: 84.33% accuracy, ROC AUC: 0.8673
- Near-perfect performance on test set (600 samples)

**Key Predictors:**
- Monthly income, loan amount, savings
- Years at address, checking account balance
- Number of units, purchase price

---

## Technologies Used

### Core Libraries
- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and tools

### Machine Learning Models
- Logistic Regression
- Random Forest (Classifier & Regressor)
- Gradient Boosting (Classifier & Regressor)
- Decision Tree Regressor
- Ridge & Lasso Regression
- Linear Regression

### Data Preprocessing
- StandardScaler - Feature scaling
- OneHotEncoder - Categorical encoding
- SimpleImputer - Missing value imputation
- ColumnTransformer - Preprocessing pipelines

### Model Optimization
- GridSearchCV - Exhaustive hyperparameter search
- RandomizedSearchCV - Random hyperparameter sampling
- Cross-validation (CV=5)

### Visualization
- **matplotlib** - Static visualizations
- **seaborn** - Statistical data visualization

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC AUC Score
- Confusion Matrix
- Mean Squared Error (MSE), Root MSE (RMSE)
- R-squared (R²)

---

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or Google Colab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tienducvu03/Data-Science-Projects.git
cd Data-Science-Projects
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter openpyxl
```

### Running the Notebooks

**Option 1: Local Jupyter**
```bash
jupyter notebook
```
Then open any `.ipynb` file from the browser interface.

**Option 2: Google Colab**
Click the "Open in Colab" badge at the top of each notebook to run in Google Colab.

---

## Project Structure

```
Data-Science-Projects/
│
├── Customer_churn_analysis.ipynb          # Telecom churn prediction
├── Customer_Churn_Analysis_README.md      # Detailed project documentation
│
├── airline_analysis.ipynb                 # Airline satisfaction analysis
├── Airline_Analysis_README.md             # Detailed project documentation
│
├── e_commerce.ipynb                       # E-commerce payment prediction
├── E_Commerce_README.md                   # Detailed project documentation
│
├── Finance_Mortgage_loan_analysist.ipynb  # Mortgage default prediction
├── Mortgage_Loan_Analysis_README.md       # Detailed project documentation
│
└── README.md                              # Main portfolio documentation
```

---

## Key Learnings

### Data Preprocessing
- Importance of handling missing values appropriately
- Effective use of one-hot encoding for categorical features
- Feature scaling improves model convergence and performance
- Outlier detection and treatment (capping at percentiles)

### Model Selection
- Ensemble methods (Random Forest, Gradient Boosting) generally outperform single models
- Hyperparameter tuning significantly impacts model performance
- Different problems require different evaluation metrics
- Cross-validation prevents overfitting

### Business Insights
- Feature importance reveals actionable business insights
- Month-to-month contracts drive customer churn
- Financial stability indicators are strong default predictors
- Payment behavior patterns predict transaction values

### Performance Considerations
- Perfect scores may indicate data leakage or overfitting
- Low R² values suggest need for additional features
- Balance between model complexity and interpretability
- Feature engineering can significantly improve predictions

---

## Future Improvements

1. **Feature Engineering**
   - Create interaction features
   - Polynomial features for non-linear relationships
   - Time-based features for temporal patterns

2. **Advanced Models**
   - XGBoost and LightGBM implementations
   - Neural networks for complex patterns
   - Ensemble stacking techniques

3. **Model Interpretability**
   - SHAP values for feature importance
   - LIME for local interpretability
   - Partial dependence plots

4. **Production Deployment**
   - Model serialization with joblib/pickle
   - REST API with Flask/FastAPI
   - Docker containerization
   - Monitoring and retraining pipelines

5. **Additional Analysis**
   - Investigate data leakage in perfect-scoring models
   - Time series analysis for temporal datasets
   - Cluster analysis for customer segmentation
   - A/B testing frameworks

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit pull requests.

---

## License

This project is open source and available for educational purposes.

---

## Contact

**Tien Duc Vu**
- GitHub: [@tienducvu03](https://github.com/tienducvu03)

---

## Acknowledgments

- Datasets sourced from various public repositories
- Developed using Google Colab for cloud computing resources
- Inspired by real-world business problems in telecom, airline, e-commerce, and finance industries

---

**Last Updated:** October 2024
