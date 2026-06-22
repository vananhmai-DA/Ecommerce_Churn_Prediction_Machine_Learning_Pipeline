# Ecommerce_Churn_Prediction_Machine_Learning_Pipeline

## Table of Contents

* [1. Project Overview](#1-project-overview)

  * [1.1 Abstract](#11-abstract)
  * [1.2 Business Problem](#12-business-problem)
  * [1.3 Dataset](#13-dataset)
* [2. Data Pipeline](#2-data-pipeline)

  * [2.1 Project Workflow](#21-project-workflow)
  * [2.2 Tools and Technologies](#22-tools-and-technologies)
  * [2.3 Database and dbt Pipeline](#23-database-and-dbt-pipeline)
* [3. Machine Learning](#3-machine-learning)

  * [3.1 Exploratory Data Analysis](#31-exploratory-data-analysis)
  * [3.2 Supervised Learning](#32-supervised-learning)
  * [3.3 Model Results](#33-model-results)
  * [3.4 Prediction Output and Risk Segmentation](#34-prediction-output-and-risk-segmentation)
  * [3.5 Unsupervised Learning: KMeans Segmentation](#35-unsupervised-learning-kmeans-segmentation)
* [4. Business Insights](#4-business-insights)

  * [4.1 Key Findings](#41-key-findings)
  * [4.2 Business Recommendations](#42-business-recommendations)
* [5. Project Assets](#5-project-assets)

  * [5.1 Project Structure](#51-project-structure)
  * [5.2 Key Outputs](#52-key-outputs)
  * [5.3 Conclusion](#53-conclusion)

---

# 1. Project Overview

## 1.1 Abstract

This project analyzes customer churn behavior for an ecommerce company and builds machine learning models to support customer retention and promotion strategies.

The project has three main goals:

* Explore the behavior of churned users through EDA
* Build supervised learning models to predict customer churn
* Segment churned users using KMeans clustering to support targeted promotion campaigns

The final supervised model is a Random Forest classifier. It was used to generate churn probability, predicted churn labels, and customer risk levels.

In addition, KMeans clustering was applied only to churned users (`churn == 1`) to identify different churned-user groups. The clustering result suggested two main churned-user segments:

* Established High-Value Churned Customers
* New One-Time Mobile Churned Customers

This project also includes a more realistic data workflow using Python, PostgreSQL, dbt, SQL analysis, model outputs, and business recommendations.

---

## 1.2 Business Problem

An ecommerce company wants to identify customers who are likely to churn so that it can take action earlier with retention campaigns and targeted promotions.

The key business questions are:

1. What are the behaviors of churned users?
2. Can machine learning predict which customers are likely to churn?
3. Can churned users be segmented into different groups for more personalized promotions?

---

## 1.3 Dataset

The dataset contains 5,630 ecommerce customers and includes customer profile, engagement, transaction, and service-related features.

Main variables include:

* Customer ID
* Churn flag
* Tenure
* Preferred login device
* City tier
* Warehouse-to-home distance
* Preferred payment mode
* Gender
* Hours spent on app
* Number of registered devices
* Preferred order category
* Satisfaction score
* Marital status
* Number of addresses
* Complaint flag
* Coupon usage
* Order count
* Days since last order
* Cashback amount

Target variable:

```text
churn
```

Where:

```text
0 = customer did not churn
1 = customer churned
```

---

# 2. Data Pipeline

## 2.1 Project Workflow

```text
Raw CSV
→ Python Ingestion
→ PostgreSQL
→ dbt Silver / Gold Models
→ Exploratory Data Analysis
→ Supervised Learning
→ Model Evaluation
→ Prediction Output
→ Risk Segmentation
→ KMeans Churned User Segmentation
→ Business Recommendations
```

This project was designed not only as a machine learning exercise, but also as a structured data pipeline that simulates a more realistic analytics workflow.

---

## 2.2 Tools and Technologies

### Data Engineering

* Python
* PostgreSQL
* dbt
* SQL
* python-dotenv
* SQLAlchemy

### Data Analysis and Machine Learning

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

### Version Control

* Git
* GitHub

---

## 2.3 Database and dbt Pipeline

The raw CSV data was first loaded into PostgreSQL using a Python ingestion pipeline.

The ingestion pipeline includes:

* Raw data loading
* Standardized column names
* Batch ID creation
* Load timestamp
* Current table
* History table
* Metadata load log
* Audit tracking

dbt was then used to transform the data into analytical layers:

```text
Raw → Staging → Silver → Gold
```

The main Gold table used for EDA and machine learning is:

```text
analytics_gold.gold_churn_model_input
```

The Gold layer includes cleaned customer-level data and engineered feature flags such as:

* `is_new_customer`
* `low_satisfaction_flag`
* `has_complaint`
* `inactive_customer_flag`
* `high_cashback_customer_flag`

dbt tests were also used to validate key assumptions such as customer ID uniqueness, accepted churn values, and valid binary feature flags.

---

# 3. Machine Learning

## 3.1 Exploratory Data Analysis

EDA was performed to understand customer churn behavior and identify important churn patterns.

Notebook:

```text
03_eda/churn_eda.ipynb
```

Main EDA checks:

* Missing values
* Duplicate customer IDs
* Churn distribution
* Numeric variable distributions
* Outlier analysis
* Churn by complaint status
* Churn by tenure
* Churn by satisfaction score
* Churn by days since last order
* Churn by order count
* Churn by preferred order category
* Churn by payment method
* Correlation with churn
* Feature flag analysis

Main findings:

* The overall churn rate is approximately 16.84%.
* Customers with complaints have much higher churn rates.
* Low-tenure customers are more likely to churn.
* New customers are an important churn-risk group.
* Mobile and mobile phone categories show higher churn rates.
* Grocery customers show lower churn rates.
* Satisfaction score does not show a simple expected relationship with churn.
* Several behavioral variables contain outliers, but they were kept because they may represent real customer behavior.

---

## 3.2 Supervised Learning

The supervised learning objective is to predict whether a customer will churn.

The preprocessing pipeline includes:

* Loading data from the Gold table
* Standardizing duplicated categorical values
* Dropping metadata columns
* Splitting features and target
* Encoding categorical features
* Scaling numeric features
* Train/test split with stratification

Models trained:

* Logistic Regression
* Decision Tree
* Random Forest V1
* Random Forest V2

Random Forest V2 was trained after removing selected engineered features as a robustness check.

---

## 3.3 Model Results

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.8028 |    0.4553 | 0.8579 |   0.5949 |  0.8937 |
| Decision Tree       |   0.9725 |    0.8955 | 0.9474 |   0.9207 |  0.9625 |
| Random Forest V1    |   0.9751 |    0.9655 | 0.8842 |   0.9231 |  0.9983 |
| Random Forest V2    |   0.9734 |    0.9762 | 0.8632 |   0.9162 |  0.9984 |

The selected model is:

```text
Random Forest V1
```

Reason for selection:

* Best overall F1-score
* Strong precision and recall balance
* Very high ROC-AUC
* More stable than a single Decision Tree

The final model was saved to:

```text
06_models/random_forest_churn_model.pkl
```

---

## 3.4 Prediction Output and Risk Segmentation

The selected Random Forest model was used to generate churn predictions for all customers.

Prediction output includes:

* Customer ID
* Churn probability
* Predicted churn label
* Risk level

Risk level definition:

| Risk Level  | Definition                       |
| ----------- | -------------------------------- |
| High Risk   | churn_probability >= 0.70        |
| Medium Risk | 0.40 <= churn_probability < 0.70 |
| Low Risk    | churn_probability < 0.40         |

Risk distribution:

| Risk Level  | Number of Customers |  Share |
| ----------- | ------------------: | -----: |
| Low Risk    |               4,680 | 83.13% |
| High Risk   |                 853 | 15.15% |
| Medium Risk |                  97 |  1.72% |

Key High Risk patterns:

* Lower tenure
* Higher complaint concentration
* Strong concentration in mobile-related categories
* Lower average cashback amount

Prediction output:

```text
05_outputs/predictions/churn_predictions.csv
```

High Risk customer output:

```text
05_outputs/segments/high_risk_customers.csv
```

---

## 3.5 Unsupervised Learning: KMeans Segmentation

KMeans clustering was applied only to customers who had already churned:

```text
churn == 1
```

The purpose was to identify different churned-user groups and design more personalized promotion strategies.

The number of clusters was evaluated using:

* Elbow Method
* Silhouette Score

The final number of clusters was:

```text
k = 2
```

### Cluster Results

| Cluster   | Number of Customers | Cluster Name                             |
| --------- | ------------------: | ---------------------------------------- |
| Cluster 0 |                 217 | Established High-Value Churned Customers |
| Cluster 1 |                 731 | New One-Time Mobile Churned Customers    |

### Cluster 0: Established High-Value Churned Customers

This group has:

* Higher tenure
* Higher order count
* Higher coupon usage
* Higher cashback amount
* Stronger historical engagement

Recommended strategy:

```text
Win-back and loyalty recovery campaign
```

### Cluster 1: New One-Time Mobile Churned Customers

This group has:

* Very low tenure
* Low order count
* Lower coupon usage
* Lower cashback amount
* Strong concentration in mobile and mobile phone categories

Recommended strategy:

```text
Second-purchase activation and mobile accessory cross-sell campaign
```

KMeans output:

```text
05_outputs/segments/churned_user_kmeans_segments.csv
```

---

# 4. Business Insights

## 4.1 Key Findings

The analysis suggests that customer churn is mainly related to early customer lifecycle, complaint behavior, product category, and engagement level.

Key findings:

* Churned customers tend to have lower tenure.
* Complaint behavior is a strong churn indicator.
* Mobile-related buyers are more likely to become churn-risk customers.
* Some churned users were previously valuable customers with higher order count and cashback amount.
* Not all churned users should receive the same promotion.
* Risk-based retention and cluster-based promotion should be used together.

---

## 4.2 Business Recommendations

The business recommendations combine both supervised learning and unsupervised learning outputs.

### Risk-Based Retention

For customers who are predicted to be at risk:

1. Prioritize High Risk customers with complaints.
2. Build a structured onboarding journey for low-tenure customers.
3. Create mobile-category cross-sell campaigns.
4. Use targeted cashback instead of broad discounts.
5. Monitor Medium Risk customers before they become High Risk.

### Cluster-Based Promotion

For customers who already churned:

1. Send win-back and loyalty recovery offers to Established High-Value Churned Customers.
2. Send second-purchase and mobile accessory offers to New One-Time Mobile Churned Customers.
3. Avoid sending the same generic promotion to all churned users.

Full recommendation document:

```text
docs/business_recommendations.md
```

---

# 5. Project Assets

## 5.1 Project Structure

```text
Ecommerce_Churn_Prediction_Pipeline/
│
├── 00_setup/
├── 01_ingestion/
├── 02_dbt_churn/
├── 03_eda/
│   └── churn_eda.ipynb
│
├── 04_ml_pipeline/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── prediction/
│   └── segmentation/
│
├── 05_outputs/
│   ├── model_results/
│   ├── predictions/
│   └── segments/
│
├── 06_models/
├── 07_sql_analysis/
├── data/
├── docs/
└── README.md
```

---

## 5.2 Key Outputs

| Output                       | Path                                                            |
| ---------------------------- | --------------------------------------------------------------- |
| Model comparison             | `05_outputs/model_results/model_comparison.csv`                 |
| Feature importance           | `05_outputs/model_results/random_forest_feature_importance.csv` |
| Churn predictions            | `05_outputs/predictions/churn_predictions.csv`                  |
| High Risk customers          | `05_outputs/segments/high_risk_customers.csv`                   |
| KMeans churned user segments | `05_outputs/segments/churned_user_kmeans_segments.csv`          |
| Saved model                  | `06_models/random_forest_churn_model.pkl`                       |
| SQL business analysis        | `07_sql_analysis/churn_business_queries.sql`                    |
| Business recommendations     | `docs/business_recommendations.md`                              |

---

## 5.3 Conclusion

This project successfully answers the three main requirements:

1. **EDA**
   Churn is strongly associated with tenure, complaints, customer lifecycle stage, product category, and engagement behavior.

2. **Supervised Learning**
   A Random Forest model was selected to predict customer churn and generate customer risk levels.

3. **Unsupervised Learning**
   KMeans clustering was applied to churned users only and identified two main churned-user groups for targeted promotion strategies.

The final outputs help the company identify at-risk customers, understand churned-user behavior, and design more personalized retention and win-back campaigns.
