# CHURNGUARD AI  
## Customer Churn Prediction & Retention Intelligence System

CHURNGUARD AI is an end-to-end Machine Learning project designed to predict whether a customer is likely to leave a service provider and convert that prediction into practical customer-retention insights.

The system analyzes historical customer information, identifies patterns associated with churn, trains and compares classification models, calculates churn probability, and classifies customers into different churn-risk levels.

---

# 1. Project Title

**CHURNGUARD AI — Customer Churn Prediction & Retention Intelligence System**

**Domain:** Artificial Intelligence & Machine Learning  
**Problem Type:** Binary Classification  
**Primary Language:** Python  
**Development Environment:** Google Colab / Jupyter Notebook  

---

# 2. Problem Statement

Customer churn refers to customers discontinuing or leaving a company's service.

For businesses operating on subscriptions or recurring customer relationships, losing existing customers can directly affect revenue and growth.

One of the major challenges is that companies may identify a customer as dissatisfied only after the customer has already decided to leave.

The goal of this project is therefore to build a Machine Learning system capable of analyzing historical customer information and predicting whether a customer is likely to:

- **Stay with the company**
- **Churn / leave the company**

Instead of only producing a binary prediction, CHURNGUARD AI also estimates the customer's probability of churn and converts it into an understandable risk category.

The complete workflow is:

```text
Raw Customer Data
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Preparation
        ↓
Train/Test Split
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Improvement
        ↓
Churn Probability
        ↓
Risk Classification
        ↓
Business Recommendation
```

---

# 3. Objective

The main objective of CHURNGUARD AI is to develop a Machine Learning classification system that can identify customers who have a higher probability of leaving a company.

The project aims to:

- Understand customer behavior from historical data.
- Identify important variables associated with customer churn.
- Clean and preprocess customer data.
- Perform meaningful Exploratory Data Analysis.
- Build multiple Machine Learning classification models.
- Compare models using appropriate evaluation metrics.
- Select the most suitable model for customer churn prediction.
- Improve the model through hyperparameter tuning and class balancing.
- Calculate churn probability for new customers.
- Convert churn probability into understandable risk levels.
- Identify influential churn-related features.
- Generate business-oriented retention recommendations.

---

# 4. Dataset Description

The project uses the **IBM Telco Customer Churn Dataset**.

The dataset contains historical information about telecommunications customers, including demographic information, services subscribed to, billing information, contract information, customer tenure, and whether the customer eventually churned.

### Target Variable

The target variable is:

```text
Churn
```

Possible values are:

```text
Yes → Customer left the company
No  → Customer stayed with the company
```

For Machine Learning, the target is converted into:

```text
Yes → 1
No  → 0
```

---

## Important Dataset Features

Some of the major features used in the project include:

| Feature | Description |
|---|---|
| `gender` | Gender of the customer |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `PhoneService` | Whether the customer uses phone service |
| `MultipleLines` | Whether the customer has multiple phone lines |
| `InternetService` | Type of internet service |
| `OnlineSecurity` | Whether online security is enabled |
| `OnlineBackup` | Whether online backup is enabled |
| `DeviceProtection` | Whether device protection is enabled |
| `TechSupport` | Whether technical support is subscribed |
| `StreamingTV` | Whether streaming TV is subscribed |
| `StreamingMovies` | Whether streaming movies are subscribed |
| `Contract` | Customer contract type |
| `PaperlessBilling` | Whether paperless billing is enabled |
| `PaymentMethod` | Customer payment method |
| `MonthlyCharges` | Monthly amount charged |
| `TotalCharges` | Total amount charged |
| `Churn` | Target variable indicating whether the customer left |

---

# 5. Technologies Used

The project was developed using the following technologies.

## Programming Language

```text
Python
```

## Development Environment

```text
Google Colab
Jupyter Notebook
```

## Libraries

### Data Manipulation

```python
pandas
numpy
```

### Data Visualization

```python
matplotlib
seaborn
```

### Machine Learning

```python
scikit-learn
```

### Model Saving

```python
joblib
```

### Optional Web Interface

```python
streamlit
```

---

# 6. Project Methodology

The project follows a complete Machine Learning lifecycle.

```text
1. Problem Understanding
2. Dataset Loading
3. Dataset Inspection
4. Data Cleaning
5. Exploratory Data Analysis
6. Feature and Target Preparation
7. Train/Test Split
8. Data Preprocessing
9. Logistic Regression Training
10. Random Forest Training
11. Model Comparison
12. Confusion Matrix Analysis
13. Model Improvement
14. Final Model Selection
15. Feature Importance Analysis
16. Churn Probability Prediction
17. Risk Classification
18. Customer Prediction Function
19. Model Saving
20. Business Interpretation
```

---

# 7. Data Preprocessing

Data preprocessing is one of the most important stages of the Machine Learning pipeline.

The following preprocessing operations were performed.

---

## 7.1 Converting TotalCharges

The `TotalCharges` feature may contain values stored as text.

It was converted into a numerical format using:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Invalid values were converted into missing values for further processing.

---

## 7.2 Removing Duplicate Records

Duplicate customer records were removed using:

```python
df = df.drop_duplicates()
```

This prevents repeated observations from unnecessarily influencing the Machine Learning model.

---

## 7.3 Removing Customer ID

The `customerID` feature was removed.

```python
df = df.drop(columns=["customerID"])
```

Customer ID is only an identifier and does not represent meaningful behavioral information for predicting churn.

---

## 7.4 Handling Missing Values

Missing values created during data conversion were handled before training the model.

```python
df = df.dropna()
```

---

## 7.5 Encoding Target Variable

The churn target was converted into binary numerical values.

```python
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})
```

---

## 7.6 Separating Numerical and Categorical Features

Features were separated into:

```text
Numerical Features
Categorical Features
```

This allows different preprocessing operations to be applied to each type.

---

## 7.7 Numerical Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

Standardization helps ensure numerical features exist on comparable scales.

---

## 7.8 Categorical Feature Encoding

Categorical variables were transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This converts categorical values into numerical representations that Machine Learning algorithms can process.

---

## 7.9 ColumnTransformer

The complete preprocessing process was combined using:

```python
ColumnTransformer
```

This ensures that numerical and categorical transformations are applied automatically inside the Machine Learning pipeline.

---

# 8. Train-Test Split

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The split used:

```text
80% → Training Data
20% → Testing Data
```

`stratify=y` was used to maintain approximately the same churn distribution in both training and testing datasets.

The testing dataset remained unseen during model training and was used for final model evaluation.

---

# 9. Exploratory Data Analysis

Exploratory Data Analysis was performed to better understand customer behavior and identify possible relationships between customer characteristics and churn.

The major visualizations include:

---

## 9.1 Churn Distribution

The first analysis examined the number of customers who stayed compared with the number who churned.

This helps identify whether the target classes are balanced or imbalanced.

```text
Churn = No
Churn = Yes
```

The churn distribution is important because significant class imbalance can influence model evaluation.

---

## 9.2 Contract Type vs Churn

Customer churn was compared across different contract types:

```text
Month-to-month
One year
Two year
```

This analysis helps determine whether customers with different contract commitments demonstrate different churn behavior.

---

## 9.3 Tenure vs Churn

Customer tenure was compared against churn.

```text
Tenure = Number of months the customer has remained with the company
```

This analysis helps determine whether newly acquired customers and long-term customers exhibit different churn patterns.

---

## 9.4 Monthly Charges vs Churn

Monthly customer charges were analyzed against churn behavior.

This helps identify whether customers paying different monthly amounts demonstrate different probabilities of leaving.

---

## 9.5 Internet Service vs Churn

Customer churn was analyzed across internet-service categories.

Examples include:

```text
DSL
Fiber Optic
No Internet Service
```

This analysis helps determine whether service type may be associated with customer retention.

---

## 9.6 Payment Method vs Churn

The relationship between payment method and churn was examined.

Payment methods include:

```text
Electronic Check
Mailed Check
Bank Transfer
Credit Card
```

The purpose is to identify whether certain customer billing behaviors are associated with higher churn.

---

# 10. Machine Learning Models Tested

Two primary Machine Learning algorithms were trained as required by the project.

An improved version of one model was also created.

---

## Model 1 — Logistic Regression

```text
Logistic Regression
```

Logistic Regression was used as the baseline classification model.

### Advantages

- Simple to implement.
- Fast to train.
- Suitable for binary classification.
- Relatively interpretable.
- Provides class probabilities.

The model predicts whether a customer belongs to either:

```text
Class 0 → Stay
Class 1 → Churn
```

---

## Model 2 — Random Forest Classifier

```text
Random Forest Classifier
```

Random Forest combines multiple decision trees and produces predictions based on the combined result of those trees.

### Advantages

- Handles nonlinear relationships.
- Works effectively with many features.
- Can capture complex relationships.
- Provides feature importance.
- Generally robust to noisy data.

---

# 11. Model Improvement

The Random Forest model was improved using:

```text
Hyperparameter Tuning
+
Class Weight Balancing
```

Hyperparameter tuning was performed using:

```python
GridSearchCV
```

Parameters evaluated included:

```text
Number of estimators
Maximum tree depth
Minimum samples required for splitting
Minimum samples required in leaf nodes
```

Class imbalance was addressed using:

```python
class_weight="balanced"
```

The objective of this stage was to improve the model's ability to correctly detect customers who are likely to churn.

---

# 12. Evaluation Metrics

The models were evaluated using several classification metrics.

---

## Accuracy

Accuracy represents the proportion of total predictions that were correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Precision

Precision measures how many customers predicted as churners actually churned.

```text
Precision =
True Positives /
(True Positives + False Positives)
```

---

## Recall

Recall measures how many actual churn customers were successfully identified.

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

Recall is particularly important in customer churn because a false negative means:

> The system predicts that a customer will stay when the customer actually leaves.

Such cases may cause businesses to miss opportunities to retain at-risk customers.

---

## F1 Score

F1-score combines precision and recall.

```text
F1 =
2 × Precision × Recall /
(Precision + Recall)
```

F1-score was used as an important metric when comparing the churn models.

---

## ROC-AUC

ROC-AUC evaluates how well the model separates churn customers from non-churn customers across different classification thresholds.

A higher ROC-AUC indicates stronger discrimination capability.

---

# 13. Model Evaluation Results

After training, the models are compared using the following metrics.

> Replace the `XX` values below with the results generated by the Colab notebook.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | XX | XX | XX | XX | XX |
| Random Forest | XX | XX | XX | XX | XX |
| Improved Random Forest | XX | XX | XX | XX | XX |

---

# 14. Confusion Matrix

A confusion matrix was used to better understand the final model's predictions.

The matrix contains:

```text
True Positive
True Negative
False Positive
False Negative
```

### True Positive

The customer actually churned and the model correctly predicted churn.

### True Negative

The customer stayed and the model correctly predicted that the customer would stay.

### False Positive

The model predicted churn, but the customer actually stayed.

### False Negative

The model predicted that the customer would stay, but the customer actually churned.

From a retention perspective, **false negatives can be especially important** because the business may fail to identify an actual high-risk customer.

---

# 15. Final Model

The final model is selected by comparing the performance of:

```text
Logistic Regression
Random Forest
Improved Random Forest
```

The final model is chosen primarily by considering:

```text
F1 Score
Recall
Precision
Accuracy
ROC-AUC
Business requirements
```

### Final Selected Model

```text
Model: XX
Accuracy: XX
Precision: XX
Recall: XX
F1 Score: XX
ROC-AUC: XX
```

Replace these values after executing the final notebook.

---

# 16. Feature Importance

The project also identifies features that have the greatest influence on customer churn predictions.

For Random Forest, feature importance is obtained using:

```python
feature_importances_
```

For Logistic Regression, coefficient magnitude can be used to understand feature influence.

The project automatically extracts and visualizes the most important features.

Example output:

```text
1. Feature A
2. Feature B
3. Feature C
4. Feature D
5. Feature E
```

The actual features should be taken from the final trained model rather than manually assumed.

---

# 17. Customer Churn Prediction

After selecting the final model, a prediction function was created.

The function accepts customer information and produces:

```text
Churn Prediction
Churn Probability
Risk Level
Suggested Business Action
```

Example structure:

```text
Customer Risk Assessment

Prediction:
Likely to Churn

Churn Probability:
78.4%

Risk Level:
HIGH

Suggested Action:
Prioritize the customer for retention outreach.
```

---

# 18. Churn Risk Classification

Instead of only producing:

```text
Churn = Yes
Churn = No
```

CHURNGUARD AI converts churn probability into risk categories.

### LOW Risk

```text
Probability < 30%
```

### MEDIUM Risk

```text
Probability = 30% – 60%
```

### HIGH Risk

```text
Probability > 60%
```

Example:

```text
Churn Probability = 18%
Risk Level = LOW
```

```text
Churn Probability = 47%
Risk Level = MEDIUM
```

```text
Churn Probability = 82%
Risk Level = HIGH
```

These thresholds can be adjusted depending on business requirements.

---

# 19. Business Insights

The Machine Learning model is not intended only to generate predictions.

The larger goal of the project is to convert those predictions into useful customer-retention intelligence.

Customers predicted as:

```text
HIGH RISK
```

could be prioritized for actions such as:

- Personalized retention campaigns.
- Customer-support outreach.
- Service-quality reviews.
- Special discounts.
- Contract upgrade incentives.
- Loyalty benefits.
- Targeted offers.
- Billing or payment assistance.

Customers classified as:

```text
MEDIUM RISK
```

could be monitored and proactively engaged.

Customers classified as:

```text
LOW RISK
```

may continue through normal engagement strategies.

---

# 20. Business Value

CHURNGUARD AI demonstrates how Machine Learning can support customer-retention decisions.

Instead of treating every customer equally, organizations can use predictive analytics to prioritize resources toward customers showing stronger churn risk.

The system can support:

```text
Customer Retention
Revenue Protection
Customer Segmentation
Targeted Marketing
Service Improvement
Customer Experience Management
```

---

# 21. Limitations

Although the system can identify statistical patterns associated with churn, several limitations must be considered.

### 1. Predictions Are Not Certainties

A high churn probability does not guarantee that a customer will leave.

Machine Learning predictions represent statistical likelihood rather than certainty.

---

### 2. Dataset-Specific Patterns

The model learns from historical patterns present in the Telco dataset.

Its performance may change when applied to another company or industry.

---

### 3. Limited Customer Context

The dataset does not include every factor that may influence customer decisions.

Factors such as:

```text
Customer satisfaction
Competitor offers
Service outages
Support experience
Economic conditions
Personal circumstances
```

may influence churn but are not fully represented.

---

### 4. Changing Customer Behavior

Customer behavior changes over time.

A model trained using historical data may eventually become less accurate if customer behavior or business conditions change.

---

### 5. Probability Thresholds

The thresholds used for LOW, MEDIUM, and HIGH risk are configurable and may not represent the optimal business thresholds for every organization.

---

# 22. Future Improvements

Several improvements could be introduced in future versions of CHURNGUARD AI.

---

## 22.1 Additional Machine Learning Models

More classification algorithms could be compared, including:

```text
XGBoost
LightGBM
Gradient Boosting
Support Vector Machines
CatBoost
```

---

## 22.2 Advanced Hyperparameter Optimization

Instead of only using basic GridSearchCV, future versions could use:

```text
RandomizedSearchCV
Optuna
Bayesian Optimization
```

---

## 22.3 Advanced Class-Imbalance Handling

Techniques such as:

```text
SMOTE
ADASYN
Undersampling
Cost-sensitive learning
```

could be evaluated.

---

## 22.4 Feature Engineering

Additional behavioral features could be created, such as:

```text
Average monthly spending
Tenure groups
Contract-risk indicators
Service-count features
Customer value groups
Service engagement score
```

---

## 22.5 Explainable AI

Explainability techniques such as:

```text
SHAP
LIME
```

could provide customer-level explanations showing why a particular customer received a high churn probability.

---

## 22.6 Dynamic Risk Thresholds

Instead of fixed:

```text
30%
60%
```

thresholds, the system could determine optimal thresholds based on business cost and retention capacity.

---

## 22.7 Real-Time Prediction API

The trained model could be deployed through:

```text
FastAPI
Flask
```

allowing external systems to request customer churn predictions.

---

## 22.8 Database Integration

Customer data could be retrieved automatically from databases such as:

```text
PostgreSQL
MySQL
MongoDB
Supabase
```

instead of manually entering customer information.

---

## 22.9 Automated Model Retraining

Future versions could automatically retrain the model when new customer data becomes available.

This would help reduce model degradation over time.

---

## 22.10 Customer Retention Recommendation Engine

The system could be extended beyond churn prediction to recommend personalized retention strategies.

For example:

```text
HIGH RISK
+
High Monthly Charges
+
Month-to-Month Contract

→ Recommend discounted annual contract
```

This would transform CHURNGUARD AI from a prediction system into a more complete **Customer Retention Intelligence Platform**.

---

# 23. Optional Streamlit Application

The repository also contains an optional Streamlit interface.

The interface allows users to enter customer information such as:

```text
Gender
Tenure
Contract
Internet Service
Payment Method
Monthly Charges
Total Charges
Services Used
```

The trained model then returns:

```text
Churn Probability
Prediction
Risk Level
```

Run the application using:

```bash
streamlit run app.py
```

---

# 24. Repository Structure

```text
CHURNGUARD-AI/
│
├── CHURNGUARD_AI_Colab.ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── data/
│   └── README.md
│
├── images/
│
└── report/
```

### `CHURNGUARD_AI_Colab.ipynb`

Contains the complete Machine Learning workflow including:

```text
Data Loading
Data Cleaning
EDA
Preprocessing
Model Training
Model Evaluation
Model Improvement
Risk Prediction
Business Interpretation
```

### `app.py`

Optional Streamlit customer churn prediction interface.

### `requirements.txt`

Contains the Python packages required to run the project.

### `data/`

Contains information related to the customer churn dataset.

### `images/`

Can be used to store:

```text
EDA visualizations
Confusion matrices
Feature importance charts
Model comparison charts
```

### `report/`

Can contain the final project documentation or PDF report.

---

# 25. Installation

Clone the repository:

```bash
git clone https://github.com/Monasri29-hub/CHURNGUARD-AI.git
```

Move into the project folder:

```bash
cd CHURNGUARD-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 26. Running the Project in Google Colab

The recommended method is Google Colab.

### Step 1

Open:

```text
CHURNGUARD_AI_Colab.ipynb
```

### Step 2

Upload/open it in Google Colab.

### Step 3

Run:

```text
Runtime → Run all
```

The notebook automatically performs the complete Machine Learning workflow.

---

# 27. Running the Streamlit Application

After training the model and generating:

```text
churnguard_model.joblib
```

place the model inside the project directory.

Then run:

```bash
streamlit run app.py
```

The CHURNGUARD AI customer-risk interface will open in the browser.

---

# 28. Project Workflow Summary

```text
Customer Dataset
       │
       ▼
Data Inspection
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature / Target Separation
       │
       ▼
Train-Test Split
       │
       ▼
Preprocessing Pipeline
       │
       ├───────────────┐
       ▼               ▼
Logistic Regression   Random Forest
       │               │
       └───────┬───────┘
               ▼
        Model Evaluation
               │
               ▼
       Model Improvement
               │
               ▼
        Final Model
               │
       ┌───────┴────────┐
       ▼                ▼
Churn Prediction   Churn Probability
                         │
                         ▼
                  Risk Classification
                         │
                         ▼
               Business Recommendation
```

---

# 29. Final Outcome

CHURNGUARD AI demonstrates a complete Machine Learning workflow for customer churn prediction.

The project moves beyond simply training a classifier by combining:

```text
Machine Learning
+
Probability Estimation
+
Feature Interpretation
+
Customer Risk Classification
+
Business Recommendations
```

The final system is capable of accepting new customer information and producing:

```text
Prediction
+
Churn Probability
+
Risk Level
+
Suggested Retention Action
```

---

# 30. Conclusion

Customer churn prediction is an important application of Machine Learning because retaining an existing customer can often be more valuable than acquiring a replacement.

CHURNGUARD AI demonstrates how historical customer information can be transformed into actionable retention intelligence through a structured Machine Learning pipeline.

The project includes data understanding, preprocessing, exploratory analysis, multiple classification models, model comparison, performance evaluation, model improvement, feature interpretation, probability-based risk assessment, and business recommendations.

The project also highlights an important principle:

> **Machine Learning predictions represent probabilities and patterns, not guarantees.**

CHURNGUARD AI therefore acts as a decision-support system that can help businesses identify customers who may require additional attention before they leave.

---

## Project

**CHURNGUARD AI**

Customer Churn Prediction & Retention Intelligence System

**Artificial Intelligence & Machine Learning**

---

## Author

**Monasri Kundeti**

B.Tech — Artificial Intelligence & Machine Learning

---

## License

This project was developed for educational and Machine Learning project purposes.
