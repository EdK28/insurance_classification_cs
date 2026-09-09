# Car Insurance Classification Case Study

A machine learning project to predict whether a customer will make a claim on their car insurance during the policy period using a Random Forest classifier.

## Project Overview

Insurance companies invest significant time and resources into optimizing their pricing and accurately estimating the likelihood that customers will make claims. This case study addresses that need by building a predictive model for **On the Road car insurance** to classify customers as likely or unlikely to file a claim.

The model is trained on customer data and evaluated using comprehensive metrics to determine its ability to predict insurance claims effectively.

## Dataset

The project uses the `car_insurance.csv` dataset containing 10,000 customer records with 18 columns:

| Column | Description |
|--------|-------------|
| `id` | Unique client identifier |
| `age` | Client's age: 0 (16-25), 1 (26-39), 2 (40-64), 3 (65+) |
| `gender` | Client's gender: 0 (Female), 1 (Male) |
| `driving_experience` | Years driving: 0 (0-9), 1 (10-19), 2 (20-29), 3 (30+) |
| `education` | Education level: 0 (None), 1 (High school), 2 (University) |
| `income` | Income level: 0 (Poverty), 1 (Working class), 2 (Middle class), 3 (Upper class) |
| `credit_score` | Credit score (0-1 range) |
| `vehicle_ownership` | Vehicle ownership: 0 (Financing), 1 (Owns vehicle) |
| `vehicle_year` | Year of registration: 0 (Before 2015), 1 (2015 or later) |
| `married` | Marital status: 0 (Not married), 1 (Married) |
| `children` | Number of children |
| `postal_code` | Client's postal code |
| `annual_mileage` | Miles driven annually |
| `vehicle_type` | Vehicle type: 0 (Sedan), 1 (Sports car) |
| `speeding_violations` | Number of speeding violations |
| `duis` | DUI count |
| `past_accidents` | Previous accidents count |
| `outcome` | **Target variable**: 0 (No claim), 1 (Made a claim) |

**Dataset Statistics:**
- **Total records:** 10,000
- **Target distribution:** ~31.3% claims, ~68.7% no claims
- **Missing values:** 
  - `credit_score`: 982 missing values (9.8%)
  - `annual_mileage`: 957 missing values (9.6%)

## Project Workflow

```
Explore → Prepare X/y → Preprocess → Split → Pipeline → Random Forest → Train → Predict → Evaluate → Conclusion
```

### Key Steps

1. **Import Required Libraries**
   - Data manipulation: pandas, numpy
   - Visualization: seaborn, matplotlib
   - ML tools: scikit-learn (preprocessing, pipeline, Random Forest, metrics)

2. **Load & Explore Dataset**
   - Display dataset shape, info, and statistics
   - Check for missing values
   - Visualize target distribution

3. **Feature Preparation**
   - Separate features (X) from target (y)
   - Remove non-predictive columns (id)
   - Identify feature types:
     - **Numerical:** credit_score, annual_mileage, speeding_violations, duis, past_accidents
     - **Categorical:** age, gender, driving_experience, education, income, vehicle_ownership, vehicle_year, married, children, postal_code, vehicle_type

4. **Data Preprocessing**
   - **Numerical features:**
     - Imputation: Median strategy for missing values
     - Scaling: RobustScaler for robustness to outliers
   - **Categorical features:**
     - Imputation: Constant strategy (fill with 'missing')
     - Encoding: One-Hot Encoding for categorical transformation
   - **Organization:** ColumnTransformer to apply transformations to respective feature groups

5. **Data Splitting**
   - Training set: 80%
   - Testing set: 20%
   - Random state: 67 (for reproducibility)

6. **Model Pipeline**
   - Combines preprocessing and classification in a single pipeline
   - **Classifier:** RandomForestClassifier with 100 estimators
   - Ensures consistent preprocessing on train/test data

7. **Training & Prediction**
   - Fit pipeline on training data
   - Generate predictions on test data

8. **Model Evaluation**
   - **Accuracy:** Overall prediction correctness
   - **Confusion Matrix:** True positives, true negatives, false positives, false negatives
   - **Precision:** Reliability of positive predictions
   - **Recall:** Coverage of actual positive cases
   - **F1-Score:** Harmonic mean balancing precision and recall
   - **Classification Report:** Comprehensive metrics summary

## Technical Implementation

### Libraries Used
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computing
- **matplotlib & seaborn:** Data visualization
- **scikit-learn:** Machine learning pipeline and algorithms
  - `Pipeline`: Combine preprocessing and modeling
  - `ColumnTransformer`: Apply different transformations to different feature types
  - `SimpleImputer`: Handle missing values
  - `StandardScaler` & `RobustScaler`: Feature scaling
  - `OneHotEncoder`: Categorical encoding
  - `train_test_split`: Data splitting
  - `RandomForestClassifier`: Classification model
  - `accuracy_score`, `confusion_matrix`, `classification_report`, `precision_score`, `recall_score`: Model evaluation

### Random Forest Classifier
The model uses Random Forest, an ensemble learning method that:
- Builds multiple decision trees during training
- Handles both numerical and categorical features
- Provides feature importance rankings
- Is resistant to overfitting through randomness and averaging
- Works well with mixed feature types

## Model Performance Insights

The model evaluates its effectiveness through:
- **Accuracy:** Percentage of correct predictions overall
- **Precision & Recall Trade-off:** Balance between avoiding false alarms and catching actual claims
- **F1-Score:** Combined performance metric for imbalanced datasets
- **Confusion Matrix:** Detailed breakdown of prediction categories

## How to Use

1. **Prepare your environment:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Ensure `car_insurance.csv` is in your working directory**

3. **Run the notebook** to:
   - Load and explore the data
   - Preprocess features
   - Train the Random Forest model
   - Evaluate performance

4. **Interpret results:**
   - Review the classification report for precision/recall metrics
   - Analyze the confusion matrix for error types
   - Consider the model's ability to predict insurance claims for business use

## Expected Outcomes

- **Data insights:** Understanding of customer characteristics affecting claim probability
- **Predictive model:** A trained Random Forest classifier ready for predictions
- **Performance metrics:** Comprehensive evaluation showing model effectiveness
- **Business value:** Ability to estimate claim likelihood for pricing and risk management

## Notes

- The notebook includes visualization of the target distribution showing the class imbalance
- Robust preprocessing handles real-world data quality issues
- Pipeline approach ensures reproducibility and prevents data leakage
- Model hyperparameters (n_estimators=100, random_state=67) are set for consistency

## Source

Dataset concepts and context derived from Accenture's research on machine learning in insurance: https://www.accenture.com/_acnmedia/pdf-84/accenture-machine-leaning-insurance.pdf

---

**Created by:** EdK28  
**Repository:** insurance_classification_cs
