# Data Science Projects

A collection of data science and machine learning projects focusing on various domains including customer analytics, finance, e-commerce, and airline industry analysis.

## Projects Overview

### 1. Customer Churn Analysis
**File:** `Customer_churn_analysis.ipynb`

Predicts customer churn for a telecommunications company using machine learning classification models.

- **Dataset:** Telco Customer Churn dataset
- **Models:** Logistic Regression, Random Forest, Gradient Boosting
- **Key Features:** Contract type, tenure, monthly charges, internet service type
- **Best Model:** Tuned Logistic Regression (Accuracy: 0.8219, F1-Score: 0.6409)
- **Techniques:** Hyperparameter tuning with GridSearchCV, feature importance analysis

**Key Findings:**
- Month-to-month contracts are the strongest predictor of churn
- Customers with lower tenure and higher charges are more likely to churn
- Lack of online security and tech support services correlates with higher churn rates

---

### 2. E-commerce Payment Analysis
**File:** `e_commerce.ipynb`

Analyzes and predicts payment values for Brazilian e-commerce transactions.

- **Dataset:** Brazil E-commerce dataset (103,886 transactions)
- **Models:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting
- **Features:** Payment type, payment installments, payment sequential
- **Best Models:** Tuned Random Forest and Gradient Boosting (R²: 0.14)
- **Techniques:** RandomizedSearchCV for hyperparameter optimization

**Key Findings:**
- Credit card is the most common payment method (74%)
- Low R² indicates additional features needed for better prediction
- Payment installments show positive correlation with payment value (0.33)

---

### 3. Airline Customer Satisfaction
**File:** `airline_analysis.ipynb`

Classifies airline customer satisfaction based on flight and service features.

- **Dataset:** Invistico Airline dataset (129,880 records)
- **Models:** Logistic Regression, Random Forest
- **Features:** Flight distance, seat comfort, online boarding, cleanliness, delays, etc.
- **Performance:** Both models achieved perfect scores (Accuracy: 1.0)
- **Techniques:** GridSearchCV, comprehensive feature preprocessing

**Key Findings:**
- Perfect classification achieved on test set
- Strong predictive power from service quality features
- Model performance suggests potential data leakage or highly separable classes

---

### 4. Finance Mortgage Loan Default Prediction
**File:** `Finance_Mortgage_loan_analysist.ipynb`

Predicts mortgage loan defaults using borrower financial information.

- **Dataset:** Finance Mortgage Loan Application dataset (3,000 records)
- **Models:** Logistic Regression, Random Forest Classifier
- **Features:** Loan amount, purchase price, monthly income, assets (checking, savings, retirement funds)
- **Best Model:** Random Forest (Accuracy: 0.9983, ROC AUC: 1.0)
- **Techniques:** Outlier capping, feature scaling, hyperparameter tuning

**Key Findings:**
- Loan amount, purchase price, and monthly income are top predictors
- Random Forest significantly outperforms Logistic Regression
- Financial asset variables show importance in default prediction

---

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running Locally
1. Clone the repository:
```bash
git clone https://github.com/tienducvu03/Data-Science-Projects.git
cd Data-Science-Projects
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open any project notebook and run the cells sequentially

### Running on Google Colab
Each notebook includes a "Open in Colab" badge at the top. Click it to run the notebook directly in Google Colab.

**Note:** Update dataset file paths in the notebooks to match your data location.

---

## Project Structure

```
Data-Science-Projects/
├── Customer_churn_analysis.ipynb
├── e_commerce.ipynb
├── airline_analysis.ipynb
├── Finance_Mortgage_loan_analysist.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

All projects follow a standard data science workflow:

1. **Data Loading** - Import and initial exploration
2. **Data Preprocessing** - Handling missing values, encoding, scaling
3. **Exploratory Data Analysis** - Statistical analysis and visualization
4. **Feature Engineering** - Creating and selecting relevant features
5. **Model Training** - Multiple algorithm comparison
6. **Hyperparameter Tuning** - GridSearchCV/RandomizedSearchCV optimization
7. **Model Evaluation** - Performance metrics and validation
8. **Visualization** - Results presentation and insights

---

## Technologies Used

- **Python** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

---

## Key Learnings

- Importance of feature engineering and domain knowledge
- Impact of hyperparameter tuning on model performance
- Trade-offs between model complexity and interpretability
- Necessity of proper data preprocessing for optimal results
- Value of ensemble methods (Random Forest, Gradient Boosting) for tabular data

---

## Future Improvements

- Add deep learning models for comparison
- Implement cross-validation strategies
- Deploy models as web APIs using Flask/FastAPI
- Create interactive dashboards with Plotly/Streamlit
- Expand datasets and perform more comprehensive feature engineering
- Add model explainability tools (SHAP, LIME)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is open source and available under the MIT License.

---

## Contact

For questions or feedback, please open an issue in the repository.

---

## Acknowledgments

- Datasets sourced from publicly available data repositories
- Inspired by real-world business problems in various industries
