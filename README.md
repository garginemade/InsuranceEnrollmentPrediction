# Insurance Enrollment Prediction & Agentic Outreach Assistant

## Overview

This project was developed as part of an ML Engineer assignment to predict employee insurance enrollment and build an intelligent outreach assistant for HR teams.

The solution follows a complete machine learning pipeline, including data cleaning, feature engineering, model training, honest evaluation, fairness considerations, and an agentic assistant capable of answering HR-related queries using the trained model.

The assistant prioritizes employees for outreach while explicitly refusing to use data leakage features such as `legacy_propensity_score`.

---

## Problem Statement

HR teams have limited outreach capacity during the open enrollment period and cannot personally contact every employee.

The objective of this project is to:

- Predict which employees are most likely to enroll in insurance benefits.
- Prioritize employees for proactive outreach based on predicted probability.
- Build an explainable AI assistant that answers HR queries while avoiding target leakage and unfair explanations.

---

## Dataset

The project uses two datasets:

```
employees.csv
```

Contains employee demographic, employment, communication, and enrollment information.

```
region_benefit_profiles.csv
```

Contains region-level statistics including historical enrollment rates, HR outreach capacity, premium costs, broker ratings, and policy information.

---

## Project Workflow

```
Input Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Data Integration
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Random Forest Classifier
        │
        ▼
Model Evaluation
        │
        ▼
Agentic Outreach Assistant
```

---

## Data Cleaning

The following preprocessing steps were performed:

- Removed duplicate employee records
- Handled missing values
- Parsed mixed date formats
- Standardized text columns
- Corrected inconsistent contact channel values
- Standardized plan tier names
- Handled sentinel values in `prior_year_enrolled`
- Identified and removed invalid contact dates
- Joined employee and regional datasets

Every important cleaning decision has been documented in the notebook.

---

## Feature Engineering

New features created include:

- Days Since Contact
- Application Month
- Contact Month
- Contact Channel Encoding
- Plan Tier Encoding
- Employment Type Encoding
- Broker Channel Encoding
- Region Profile Features

Leaky features such as `legacy_propensity_score` were excluded from model training.

---

## Machine Learning

### Model

Random Forest Classifier

### Train/Test Split

- 80% Training
- 20% Testing
- Stratified Split

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision@K
- Baseline Comparison

---

## Fairness Considerations

This project includes a fairness review.

Sensitive demographic attributes were carefully considered during feature selection.

The generated explanations intentionally avoid referencing:

- Gender
- Age
- Marital Status

to reduce the risk of discriminatory or legally sensitive explanations.

---

## Agentic Outreach Assistant

The project includes a rule-based AI assistant built entirely in Python.

### Available Tools

### 1. Predict Enrollment

```python
predict_enrollment(employee_id)
```

Predicts the enrollment probability for a single employee.

---

### 2. Rank Outreach Candidates

```python
rank_outreach_candidates(region)
```

Ranks employees within a region based on predicted enrollment probability while respecting the HR outreach capacity.

---

### 3. Lookup Region Profile

```python
lookup_region_profile(region)
```

Returns regional statistics from the region dataset.

---

### 4. Explain Prediction

```python
explain_prediction(employee_id)
```

Generates a business-friendly explanation without exposing demographic reasoning.

---

### 5. Assistant Router

```python
assistant(query)
```

Routes natural language commands to the appropriate tool.

---

## Example Queries

```python
assistant("predict 1001")

assistant("explain 1001")

assistant("rank west")

assistant("top 10 west")

assistant("region west")
```

---

## Leakage Protection

The assistant explicitly refuses requests involving:

```
legacy_propensity_score
```

because it reconstructs the target and would introduce target leakage.

Example:

```
assistant("predict using legacy_propensity_score")
```

Output:

```
Request refused.
'legacy_propensity_score' is a leaky feature and cannot be used for prediction or explanation.
```

---

## Repository Structure

```
InsuranceEnrollmentPrediction/
│
├── insurance_enrollment_prediction.ipynb
├── README.md
├── report/
│   ├── report.pdf
├── AI_USAGE.md
├── requirements.txt
│
├── data/
│   ├── employees.csv
│   └── region_benefit_profiles.csv
│
├── docs/
    └── architecture.png

```

---

## Repository

GitHub: https://github.com/garginemade/InsuranceEnrollmentPrediction.git

## Installation

Clone the repository:

```bash
git clone <repository_link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
insurance_enrollment_prediction.ipynb
```

Run all cells in order.

---

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter

Install using:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- SHAP-based model explanations
- Probability calibration
- Interactive Streamlit dashboard
- LLM-powered conversational assistant
- Automated fairness monitoring

---

## Author

**Gargi Nemade**

Machine Learning Engineer Assignment

2026
