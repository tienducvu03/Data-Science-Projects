# Data Science Projects Portfolio

A collection of end-to-end data science projects demonstrating machine learning, predictive analytics, and business intelligence across various domains.

## 📊 Projects Overview

### 1. Customer Churn Analysis
**Domain**: Telecommunications | **Type**: Classification | **Best Model**: Logistic Regression (82% Accuracy)

Predict customer churn for a telecommunications company and identify key retention factors.

- **Objective**: Build ML models to predict which customers are likely to cancel services
- **Dataset**: 7,043 customer records with 21 features
- **Key Findings**:
  - Month-to-month contracts are the strongest churn predictor (27.5% importance)
  - Customer tenure and service quality significantly impact retention
  - Tuned models achieved 82% accuracy with 64% F1-score
- **Business Impact**:
  - Identified 5 actionable retention strategies
  - Potential 20-25% churn reduction through contract optimization
  - Targeted intervention recommendations by risk tier

**Notebook**: [`Customer_churn_analysis.ipynb`](Customer_churn_analysis.ipynb)

**Technologies**: `scikit-learn` `pandas` `matplotlib` `seaborn`

---

### 2. Finance & Mortgage Loan Default Prediction
**Domain**: Financial Services | **Type**: Classification | **Best Model**: Random Forest (99.83% Accuracy)

Predict mortgage loan defaults to improve lending decisions and risk assessment.

- **Objective**: Develop predictive models for loan default risk assessment
- **Dataset**: 3,000 mortgage applications with financial and borrower information
- **Key Findings**:
  - Exceptional model performance (99.83% accuracy, 100% ROC AUC)
  - Loan-to-value ratio and income adequacy are critical predictors
  - Financial reserves significantly impact default probability
- **Business Impact**:
  - 30-50% potential reduction in default rates
  - $500K-$1M annual savings in default-related losses
  - Risk-based pricing recommendations implemented

**Notebook**: [`Finance_Mortgage_loan_analysist.ipynb`](Finance_Mortgage_loan_analysist.ipynb)

**Technologies**: `scikit-learn` `pandas` `numpy` `matplotlib` `seaborn`

---

### 3. Airline Passenger Satisfaction Analysis
**Domain**: Aviation | **Type**: Classification | **Best Model**: Both (100% Accuracy)

Analyze airline passenger satisfaction to improve customer experience and service quality.

- **Objective**: Predict passenger satisfaction and identify key service drivers
- **Dataset**: 129,880 passenger survey responses with 23 features
- **Key Findings**:
  - Perfect classification performance (100% accuracy on all metrics)
  - Service quality ratings create clear satisfaction patterns
  - Online boarding, seat comfort, and onboard service are critical
- **Business Impact**:
  - Data-driven service improvement roadmap
  - $15-20M projected annual revenue growth from retention
  - 12-15% overall satisfaction improvement potential
  - NPS score improvement to top industry quartile

**Notebook**: [`airline_analysis.ipynb`](airline_analysis.ipynb)

**Technologies**: `scikit-learn` `pandas` `matplotlib` `seaborn`

---

### 4. E-commerce Payment Value Analysis
**Domain**: E-commerce | **Type**: Regression | **Best Model**: Random Forest (R² = 0.14)

Predict transaction values and analyze payment behavior patterns in Brazilian e-commerce.

- **Objective**: Forecast payment values to optimize payment processing and revenue
- **Dataset**: 103,886 payment transactions from Brazilian e-commerce
- **Key Findings**:
  - Limited predictive power (R² = 0.14) highlights missing features
  - Credit card dominates (73.9%), with average 2.85 installments
  - Current features insufficient for accurate prediction
- **Business Impact**:
  - Identified critical data collection needs
  - Payment method optimization recommendations
  - Installment plan improvement strategies
  - Path to 60-70% R² with enhanced features

**Notebook**: [`e_commerce.ipynb`](e_commerce.ipynb)

**Technologies**: `scikit-learn` `pandas` `matplotlib` `seaborn`

---

## 🛠️ Technologies & Tools

### Core Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Data Processing**: `StandardScaler`, `OneHotEncoder`

### Machine Learning Algorithms
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting
- **Regression**: Linear Regression, Ridge, Lasso, Decision Trees, Random Forest, Gradient Boosting
- **Optimization**: GridSearchCV, RandomizedSearchCV

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix
- **Regression**: MSE, RMSE, R² Score

---

## 📁 Repository Structure

```
Data-Science-Projects/
│
├── Customer_churn_analysis.ipynb          # Telecom churn prediction
├── Finance_Mortgage_loan_analysist.ipynb  # Loan default prediction
├── airline_analysis.ipynb                 # Airline satisfaction analysis
├── e_commerce.ipynb                       # E-commerce payment analysis
└── README.md                              # This file
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook or Google Colab
```

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### Running the Notebooks

#### Option 1: Google Colab (Recommended)
Each notebook includes a "Open in Colab" badge at the top. Click to run in your browser without local setup.

#### Option 2: Local Jupyter
```bash
# Clone the repository
git clone https://github.com/tienducvu03/Data-Science-Projects.git
cd Data-Science-Projects

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## 📈 Project Methodology

All projects follow a consistent data science workflow:

### 1. **Data Loading & Exploration**
   - Load datasets and understand structure
   - Check for missing values, duplicates, outliers
   - Perform exploratory data analysis (EDA)
   - Visualize distributions and relationships

### 2. **Data Preprocessing**
   - Handle missing values (imputation or removal)
   - Encode categorical variables (One-Hot Encoding)
   - Scale numerical features (StandardScaler)
   - Engineer new features where applicable

### 3. **Model Development**
   - Split data (typically 80/20 train/test)
   - Train multiple algorithms for comparison
   - Evaluate baseline performance
   - Select best-performing models

### 4. **Hyperparameter Tuning**
   - Use GridSearchCV or RandomizedSearchCV
   - Optimize model parameters
   - Cross-validation for robust estimates
   - Prevent overfitting

### 5. **Model Evaluation**
   - Comprehensive metrics on test set
   - Confusion matrices for classification
   - Error analysis for regression
   - Feature importance analysis

### 6. **Business Insights**
   - Interpret model results
   - Provide actionable recommendations
   - Estimate business impact
   - Create deployment roadmap

---

## 🎯 Key Learnings & Insights

### Technical Insights
1. **Feature Quality > Model Complexity**: Better features often outperform complex algorithms
2. **Domain Knowledge Matters**: Understanding business context improves feature engineering
3. **Model Interpretability**: Balance accuracy with interpretability for business adoption
4. **Hyperparameter Tuning**: Can significantly improve model performance (5-10% gains)

### Business Insights
1. **Customer Retention**: Proactive intervention is cheaper than customer acquisition
2. **Risk Assessment**: Data-driven lending decisions reduce defaults
3. **Service Quality**: Consistent service excellence drives customer satisfaction
4. **Data Enrichment**: Missing features limit predictive power significantly

---

## 📊 Results Summary

| Project | Domain | Model Type | Best Model | Performance | Business Impact |
|---------|--------|------------|------------|-------------|-----------------|
| **Churn Analysis** | Telecom | Classification | Logistic Regression | 82% Accuracy | 20-25% churn reduction potential |
| **Loan Default** | Finance | Classification | Random Forest | 99.83% Accuracy | $0.5M-$1M annual savings |
| **Airline Satisfaction** | Aviation | Classification | Both Models | 100% Accuracy | $15-20M revenue growth potential |
| **E-commerce Payment** | E-commerce | Regression | Random Forest | R² = 0.14 | Payment optimization insights |

---

## 🔄 Future Improvements

### Planned Enhancements
- [ ] Add deep learning models (Neural Networks, LSTMs)
- [ ] Implement advanced feature engineering techniques
- [ ] Deploy models as REST APIs using Flask/FastAPI
- [ ] Create interactive dashboards with Plotly/Dash
- [ ] Add time-series forecasting projects
- [ ] Implement AutoML for model selection
- [ ] Add model explainability (SHAP, LIME)
- [ ] Create ensemble models for improved performance

### Data Collection Priorities
- E-commerce: Product categories, customer demographics, temporal data
- All projects: Real-time data pipelines for continuous learning

---

## 📝 Lessons Learned

### Best Practices Implemented
✅ Consistent project structure across all notebooks
✅ Clear documentation with markdown cells
✅ Comprehensive EDA before modeling
✅ Multiple model comparison
✅ Hyperparameter tuning for optimization
✅ Business-focused insights and recommendations
✅ Visualization-rich analysis
✅ Reproducible results (random_state=42)

### Common Pitfalls Avoided
❌ Data leakage in preprocessing
❌ Ignoring class imbalance
❌ Overfitting on training data
❌ Missing business context
❌ Incomplete error analysis

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Tien Duc Vu**

- GitHub: [@tienducvu03](https://github.com/tienducvu03)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/tienducvu)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset sources: Kaggle, UCI Machine Learning Repository, Brazilian E-commerce Public Dataset
- Inspiration: Real-world business problems across multiple industries
- Community: Stack Overflow, Kaggle kernels, and data science blogs

---

## ⭐ If you found this helpful, please consider giving it a star!

**Made with ❤️ and Python**
