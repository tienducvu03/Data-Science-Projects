# Airline Customer Satisfaction Analysis

Machine learning classification project to predict airline customer satisfaction based on flight experience and service quality metrics.

## Project Overview

This project analyzes customer satisfaction for Invistico Airlines using various service quality indicators. The goal is to build predictive models that can identify satisfied vs. dissatisfied customers, enabling the airline to improve service delivery and customer experience.

## Dataset

**Source:** `/content/Invistico_Airline_Cleaned.csv`

**Size:** 129,880 customer records

**Target Variable:**
- **satisfaction:** Binary (satisfied / dissatisfied)

**Features:** 22 attributes including:

### Customer Demographics
- **Gender:** Male/Female
- **Customer Type:** Loyal Customer / Disloyal Customer
- **Age:** Customer age
- **Type of Travel:** Personal Travel / Business Travel
- **Class:** Eco / Business / Eco Plus

### Flight Information
- **Flight Distance:** Distance traveled in miles

### Service Quality Ratings (0-5 scale)
- **Seat comfort**
- **Departure/Arrival time convenient**
- **Food and drink**
- **Gate location**
- **Inflight wifi service**
- **Inflight entertainment**
- **Online support**
- **Ease of Online booking**
- **On-board service**
- **Leg room service**
- **Baggage handling**
- **Checkin service**
- **Cleanliness**
- **Online boarding**

### Operational Metrics
- **Departure Delay in Minutes**
- **Arrival Delay in Minutes**

**Data Quality:**
- Clean dataset (no missing values after preprocessing)
- All features are numerical (post-encoding)
- Balanced distribution of satisfaction levels

## Methodology

### 1. Data Preprocessing

**Missing Value Treatment:**
- Identified missing values in 'Arrival Delay in Minutes'
- Applied mean imputation using SimpleImputer

**Feature Engineering:**
- Categorical encoding with OneHotEncoder (Gender, Customer Type, Type of Travel, Class)
- Numerical scaling with StandardScaler
- Created preprocessing pipeline with ColumnTransformer

**Data Splitting:**
- Training set: 103,904 samples (80%)
- Test set: 25,976 samples (20%)
- Final feature count: 29 (after one-hot encoding)

### 2. Model Selection

Two classification algorithms evaluated:
1. **Logistic Regression** (solver='liblinear')
2. **Random Forest Classifier**

### 3. Hyperparameter Tuning

**GridSearchCV** with 5-fold cross-validation:

**Logistic Regression Parameters:**
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- penalty: ['l1', 'l2']
- scoring: 'accuracy'

**Random Forest Parameters:**
- n_estimators: [100, 200, 300]
- max_depth: [None, 10, 20, 30]
- min_samples_split: [2, 5, 10]
- scoring: 'accuracy'

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Tuned Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Tuned Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Best Hyperparameters

**Logistic Regression:**
- C: 0.001
- penalty: 'l1'

**Random Forest:**
- max_depth: None
- min_samples_split: 2
- n_estimators: 100

### Confusion Matrices

Both models achieved perfect classification on the test set:

```
[[11,675      0]
 [     0  14,301]]
```

- **True Negatives (Dissatisfied correctly predicted):** 11,675
- **True Positives (Satisfied correctly predicted):** 14,301
- **False Positives:** 0
- **False Negatives:** 0

## Analysis Notes

### Perfect Performance Investigation

The models achieved 100% accuracy on both training and test sets. This exceptional performance suggests:

**Possible Explanations:**
1. **Data Leakage:** Target variable may be directly derivable from features
2. **Strong Feature Correlation:** Service ratings may perfectly correlate with satisfaction
3. **Preprocessed Dataset:** The "Cleaned" dataset may already be optimized
4. **Simple Decision Boundary:** Satisfaction may be deterministically based on rating thresholds

**Recommended Next Steps:**
1. Investigate feature correlation with target variable
2. Test on truly unseen dataset from different time period
3. Perform rigorous cross-validation on original (uncleaned) data
4. Check for data collection methodology that might create perfect separation

## Visualizations

The notebook includes:
- **Model Performance Comparison Bar Chart** (Accuracy, Precision, Recall, F1-Score)
- Side-by-side visualization of both models
- Perfect score demonstration (all metrics = 1.00)

## Key Business Insights

### Service Quality Drivers
Based on the dataset structure, satisfaction likely depends on:
1. **Service Ratings:** Higher ratings across all 14 service dimensions
2. **Customer Type:** Loyal customers may have different satisfaction patterns
3. **Travel Class:** Business class vs. Economy experience differences
4. **Flight Operations:** Delay times impact satisfaction
5. **Travel Purpose:** Business vs. Personal travel expectations

### Recommendations for Airline Operations

1. **Service Excellence:** Maintain high standards across all 14 service dimensions
2. **Delay Management:** Minimize departure and arrival delays
3. **Customer Segmentation:** Tailor services for business vs. leisure travelers
4. **Loyalty Programs:** Focus on converting disloyal to loyal customers
5. **Class-Specific Improvements:** Address pain points in each travel class

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution Steps
1. Open `airline_analysis.ipynb` in Jupyter Notebook or Google Colab
2. Ensure dataset is available at `/content/Invistico_Airline_Cleaned.csv`
3. Run all cells sequentially
4. Review preprocessing steps, model training, and evaluation results

### Google Colab
Click the "Open in Colab" badge at the top of the notebook.

## Model Deployment

### Production Considerations

**Current State:**
- Models ready for deployment (perfect test performance)
- Simple preprocessing pipeline for new data
- Fast prediction time (milliseconds per record)

**Deployment Architecture:**
1. **Model Serialization:** Save trained models with joblib
2. **REST API:** Flask/FastAPI for real-time predictions
3. **Batch Processing:** Predict satisfaction for upcoming flights
4. **Feedback Loop:** Collect actual satisfaction to validate predictions

### Real-World Application

**Use Cases:**
1. **Proactive Service Recovery:** Identify potentially dissatisfied passengers before landing
2. **Resource Allocation:** Deploy staff to assist predicted dissatisfied customers
3. **Service Monitoring:** Track satisfaction trends across routes and time
4. **Pricing Strategy:** Adjust pricing based on predicted satisfaction levels

## Dataset Characteristics

**Strengths:**
- Large sample size (129,880 records)
- Comprehensive service quality metrics (14 dimensions)
- Clean data with minimal preprocessing needed
- Balanced target variable distribution

**Limitations:**
- Perfect model performance suggests potential data issues
- Static snapshot (no temporal analysis)
- Limited demographic features (only age, gender)
- No route-specific or aircraft-specific information

## Future Improvements

### Model Enhancement
1. **Feature Selection:** Identify most important service dimensions
2. **Deep Learning:** Neural networks for complex pattern recognition
3. **Ensemble Methods:** Stack multiple models for robustness
4. **Probabilistic Predictions:** Output confidence scores instead of binary predictions

### Feature Engineering
1. **Composite Scores:** Create aggregate service quality index
2. **Delay Categories:** Bin delays into severity levels
3. **Customer Tenure:** Time since becoming loyal customer
4. **Route Characteristics:** Domestic vs. international, distance categories
5. **Temporal Features:** Day of week, season, holiday periods

### Data Collection
1. **Additional Features:** Route details, aircraft type, crew ratings
2. **Longitudinal Data:** Track same customers over time
3. **Textual Feedback:** Analyze customer comments with NLP
4. **Competitor Comparison:** Benchmark against industry standards

### Validation
1. **Cross-Dataset Testing:** Validate on different airlines' data
2. **Temporal Validation:** Test on future time periods
3. **A/B Testing:** Compare predictions against actual outcomes
4. **Blind Test Set:** Holdout data not seen during any phase

## Technical Details

**Notebook:** `airline_analysis.ipynb`

**Libraries:**
- pandas 1.x - Data manipulation
- numpy 1.x - Numerical computing
- scikit-learn 1.x - Machine learning
- matplotlib 3.x - Visualization
- seaborn - Statistical plots

**Computational Requirements:**
- Training time: ~5-10 minutes (Google Colab free tier)
- Memory usage: ~500 MB RAM
- CPU-only (no GPU required)

## Files

- `airline_analysis.ipynb` - Main analysis notebook
- `Invistico_Airline_Cleaned.csv` - Dataset (not included)

## Project Structure

```
airline_analysis.ipynb
├── Data Loading
├── Data Preprocessing
│   ├── Missing value imputation
│   ├── Feature encoding
│   └── Feature scaling
├── Data Splitting
├── Model Training
│   ├── Logistic Regression
│   └── Random Forest
├── Hyperparameter Tuning
│   ├── GridSearchCV (Logistic)
│   └── GridSearchCV (Random Forest)
├── Model Evaluation
│   ├── Metrics calculation
│   └── Confusion matrices
└── Results Visualization
    └── Performance comparison charts
```

## Author

Tien Duc Vu

## License

Educational purposes

## Last Updated

October 2024

---

**Note:** The perfect model performance (100% accuracy) warrants further investigation to ensure real-world applicability. Consider this project as a demonstration of the machine learning workflow rather than a production-ready model without additional validation.
