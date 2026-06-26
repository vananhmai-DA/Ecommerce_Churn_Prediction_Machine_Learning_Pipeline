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
  * [3.4 Hyperparameter Tuning and Cross-Validation](#34-hyperparameter-tuning-and-cross-validation)
  * [3.5 Threshold Analysis](#35-threshold-analysis)
  * [3.6 SHAP Model Explainability](#36-shap-model-explainability)
  * [3.7 Prediction Output and Risk Segmentation](#37-prediction-output-and-risk-segmentation)
  * [3.8 Unsupervised Learning: KMeans Segmentation](#38-unsupervised-learning-kmeans-segmentation)

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

* Explore the behavior of churned users through exploratory data analysis.
* Build supervised learning models to predict customer churn.
* Segment churned users using KMeans clustering to support targeted promotion campaigns.

The final supervised model is a Random Forest classifier. It was used to generate churn probability, predicted churn labels, and customer risk levels.

To strengthen the machine learning layer, this project also includes:

* Hyperparameter tuning with cross-validation
* Threshold analysis for business decision-making
* SHAP model explainability

In addition, KMeans clustering was applied only to churned users (`churn == 1`) to identify different churned-user groups. The clustering result suggested two main churned-user segments:

* Established High-Value Churned Customers
* New One-Time Mobile Churned Customers

This project also includes a realistic data workflow using Python, PostgreSQL, dbt, SQL analysis, model outputs, and business recommendations.

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
→ Hyperparameter Tuning
→ Threshold Analysis
→ SHAP Explainability
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
* SHAP
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

## 3.4 Hyperparameter Tuning and Cross-Validation

To deepen the machine learning evaluation, Random Forest hyperparameter tuning was performed using `RandomizedSearchCV`.

Notebook:

```text
04_ml_pipeline/tuning/random_forest_hyperparameter_tuning.ipynb
```

The tuning process used:

* RandomizedSearchCV
* 5-fold cross-validation
* F1-score as the optimization metric

F1-score was selected because the dataset is imbalanced and accuracy alone may not fully reflect the model's ability to identify churned customers.

The tuning search tested parameters such as:

* Number of trees
* Maximum tree depth
* Minimum samples split
* Minimum samples per leaf
* Maximum features
* Class weight strategy

### Tuning Results

| Model                  | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---------------------- | -------: | --------: | -----: | -------: | ------: |
| Baseline Random Forest |   0.9751 |    0.9655 | 0.8842 |   0.9231 |  0.9983 |
| Tuned Random Forest    |   0.9680 |    0.9185 | 0.8895 |   0.9037 |  0.9951 |

The tuned Random Forest slightly improved recall, but it reduced precision, F1-score, accuracy, and ROC-AUC.

Therefore, the baseline Random Forest remained the selected final model.

This shows that hyperparameter tuning was tested, but the simpler baseline Random Forest provided the better overall balance on the test set.

Tuning outputs:

```text
05_outputs/model_results/random_forest_tuning_comparison.csv
05_outputs/model_results/random_forest_randomized_search_cv_results.csv
```

---

## 3.5 Threshold Analysis

The selected Random Forest model outputs churn probabilities. By default, classification models often use a threshold of 0.50 to convert probabilities into class labels.

However, in a churn prediction problem, the best threshold depends on the business objective.

Notebook:

```text
04_ml_pipeline/evaluation/churn_threshold_analysis.ipynb
```

The threshold analysis compared multiple thresholds from 0.30 to 0.80 using:

* Precision
* Recall
* F1-score
* Number of customers flagged as churn

Main interpretation:

* Lower thresholds capture more churned customers but increase campaign cost.
* Higher thresholds improve precision but may miss more churned customers.
* Threshold 0.40 provides a strong balance between precision, recall, F1-score, and business actionability.

Recommended operating threshold for retention campaigns:

```text
0.40
```
### Threshold 0.40 vs 0.50 Comparison

The default classification threshold is usually 0.50. However, in churn prediction, the business objective is often to capture more actual churned customers for retention campaigns.

Compared with the default threshold of 0.50, threshold 0.40 improves recall from 88.42% to 95.26% and improves F1-score from 92.31% to 94.76%. Accuracy also increases from 97.51% to 98.22%.

More importantly, threshold 0.40 reduces false negatives from 22 customers to 9 customers. This means the model misses fewer actual churned customers. True positives also increase from 168 to 181 customers.

The trade-off is that precision decreases from 96.55% to 94.27%, and false positives increase from 6 to 11 customers. The number of customers flagged as churn also increases from 174 to 192.

For a churn retention use case, this trade-off is acceptable because missing actual churned customers may be more costly than targeting a small number of additional customers. Therefore, threshold 0.40 is recommended as the operating threshold for retention campaigns.

Threshold 0.40 is selected because it improves recall and F1-score while significantly reducing false negatives, which is more aligned with the business objective of churn retention.

This does not replace the final model. Instead, it helps the company decide how aggressively to target customers for retention campaigns.

Threshold outputs:

```text
05_outputs/model_results/threshold_analysis.csv
05_outputs/figures/threshold_metric_comparison.png
05_outputs/figures/threshold_customers_flagged.png
```

---

## 3.6 SHAP Model Explainability

SHAP was used to explain the selected Random Forest churn prediction model.

Notebook:

```text
04_ml_pipeline/evaluation/churn_model_explainability_shap.ipynb
```

While traditional feature importance shows which variables are important overall, SHAP provides a more interpretable view of how features contribute to churn predictions.

The most important SHAP features include:

* Tenure
* New customer flag
* Complaint-related features
* Days since last order
* Number of addresses
* Cashback amount
* Warehouse-to-home distance
* City tier
* Satisfaction score
* Product category features

Key interpretation:

* Tenure is the strongest driver of churn prediction.
* New customers are a major churn-risk group.
* Complaint behavior is a strong churn signal.
* Engagement and purchase-related variables also influence churn prediction.
* SHAP supports the business recommendation to prioritize low-tenure customers, complaint customers, and customers with weaker engagement signals.

SHAP outputs:

```text
05_outputs/model_results/shap_feature_importance.csv
05_outputs/figures/shap_summary_plot.png
```

---

## 3.7 Prediction Output and Risk Segmentation

The selected Random Forest model was used to generate churn predictions for all customers.

Prediction output includes:

* Customer ID
* Churn risk score
* Predicted churn label
* Risk level

The model outputs a churn probability score. In this project, this score is used as a **churn risk score** to rank customers and support retention prioritization.

This means the score is used to compare customers by relative churn risk. For example, a customer with a churn risk score of 0.70 is considered higher risk than a customer with a score of 0.30.

However, this score should not be interpreted as an exact real-world probability of churn unless the model probability is fully calibrated.

### Probability Calibration Check

A probability calibration check was added to evaluate whether the predicted probabilities can be interpreted as absolute probabilities.

The model achieved a Brier score of:

```text
0.0264
```

A lower Brier score indicates better probability prediction, so this result suggests that the model has strong overall probability performance.

The calibration analysis shows that the model is effective at separating low-risk and high-risk customers. Customers in low-score buckets have an actual churn rate close to 0%, while customers in the highest probability bucket have an actual churn rate of 100%.

However, the calibration curve shows that the predicted probabilities are not perfectly calibrated across all probability ranges. For example, the highest probability bucket has an average predicted probability of 82.42%, while the actual churn rate is 100%.

Therefore, the churn score is reliable for ranking customers by churn risk, but it should be interpreted as a **risk score** rather than a fully calibrated absolute probability.

### Risk Level Definition

| Risk Level  | Definition                      | Business Meaning                                                       |
| ----------- | ------------------------------- | ---------------------------------------------------------------------- |
| High Risk   | churn risk score >= 0.70        | Customers most likely to churn and should be prioritized for retention |
| Medium Risk | 0.40 <= churn risk score < 0.70 | Customers showing some churn signals and should be monitored           |
| Low Risk    | churn risk score < 0.40         | Customers with lower churn risk and lower immediate retention priority |

### Risk Distribution

| Risk Level  | Number of Customers |  Share |
| ----------- | ------------------: | -----: |
| Low Risk    |               4,680 | 83.13% |
| High Risk   |                 853 | 15.15% |
| Medium Risk |                  97 |  1.72% |

### Risk Segment Insights

The prediction output was merged with the Gold customer table to understand how High, Medium, and Low Risk customers differ.

#### High Risk Segment

High Risk customers show several clear data patterns.

First, High Risk customers have much lower tenure than Low Risk customers. The average tenure of High Risk customers is around 2.63, compared with around 11.07 for Low Risk customers. The median tenure of the High Risk group is only 1.

This suggests that many churn-risk customers are still in the early stage of their customer lifecycle. These customers may not have built strong loyalty or repeat-purchase habits yet.

Second, complaint behavior is strongly concentrated in the High Risk group. More than half of High Risk customers have submitted complaints, suggesting that service experience is an important churn driver.

Third, High Risk customers are strongly concentrated in mobile-related categories. Mobile phone accounts for 38.34% of High Risk customers, and mobile accounts for 23.21%. Together, mobile-related categories represent 61.55% of the High Risk segment.

This is much higher than the Low Risk group, where mobile phone and mobile together represent around 32.20%.

This suggests that mobile-related purchases may be more one-off or replacement-driven. These customers may have fewer natural reasons to return after the first purchase, so they require more targeted retention actions.

High Risk customers also have lower average cashback amounts than Low Risk customers. This may suggest weaker incentive exposure or lower engagement with cashback-based retention mechanisms.

Recommended actions for High Risk customers:

* Create onboarding campaigns for new or low-tenure customers.
* Prioritize complaint recovery before sending generic discounts.
* Follow up with customers after complaint resolution.
* Design mobile-category retention campaigns.
* Use accessory cross-sell campaigns after mobile purchases.
* Offer targeted cashback or second-purchase incentives.
* Avoid applying the same promotion to all High Risk customers without considering complaint history, tenure, and product category.

#### Medium Risk Segment

Medium Risk customers are not as urgent as High Risk customers, but they already show some churn signals.

In terms of preferred order category, Medium Risk customers are spread across mobile phone, laptop & accessory, and mobile. This suggests that Medium Risk is a transition group rather than a single clearly defined customer type.

Recommended actions for Medium Risk customers:

* Send light incentives such as coupons or free shipping.
* Monitor changes in days since last order, complaint status, and order frequency.
* Use personalized product recommendations instead of aggressive discounting.
* Trigger stronger retention actions if the customer moves from Medium Risk to High Risk.

#### Low Risk Segment

Low Risk customers represent the majority of the customer base. This group is more concentrated in laptop & accessory, which accounts for 39.29% of Low Risk customers.

Compared with High Risk customers, Low Risk customers have higher tenure, lower complaint concentration, and lower immediate churn risk. This suggests that they are relatively more stable and should not receive aggressive retention spending.

Recommended actions for Low Risk customers:

* Maintain loyalty communication.
* Send regular product updates or personalized recommendations.
* Avoid unnecessary discounting.
* Continue monitoring for future changes in churn risk.

### Why KMeans Segmentation Is Still Needed

Risk segmentation and KMeans clustering answer different business questions.

Risk segmentation is based on supervised learning and is used to prioritize current customers by churn risk:

```text
Who is most likely to churn?
```

KMeans clustering is based on unsupervised learning and is applied only to customers who already churned:

```text
What different types of churned customers exist?
```

Therefore, the two approaches are complementary.

Risk segmentation supports retention prioritization for current customers, while KMeans segmentation supports personalized win-back and promotion strategies for churned customers.

Prediction output:

```text
05_outputs/predictions/churn_predictions.csv
```

High Risk customer output:

```text
05_outputs/segments/high_risk_customers.csv
```

Risk profiling outputs:

```text
05_outputs/segments/risk_level_numeric_profile.csv
05_outputs/segments/risk_level_category_profile.csv
```

Probability calibration outputs:

```text
05_outputs/model_results/probability_calibration_summary.csv
05_outputs/model_results/probability_calibration_metric.csv
05_outputs/figures/probability_calibration_curve.png
```


---

## 3.8 Unsupervised Learning: KMeans Segmentation

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
* High Risk customers have much lower tenure than Low Risk customers.
* More than half of High Risk customers have complaint history.
* High Risk customers are strongly concentrated in mobile-related categories.
* Mobile phone and mobile together represent 61.55% of the High Risk segment, compared with around 32.20% of the Low Risk segment.
* Low Risk customers are more concentrated in laptop & accessory, suggesting that this group is relatively more stable.
* Some churned users were previously valuable customers with higher order count and cashback amount.
* Not all churned users should receive the same promotion.
* Risk-based retention and cluster-based promotion should be used together.
* Threshold 0.40 is more suitable for retention campaigns because it improves recall and reduces false negatives compared with the default threshold of 0.50.
* The model-generated churn probability is useful as a risk score, but it should not be interpreted as a fully calibrated absolute probability.
* SHAP explainability confirms that tenure, new customer status, complaints, and engagement-related features are important drivers of churn prediction.

---

## 4.2 Business Recommendations

The business recommendations combine supervised learning, risk segmentation, probability calibration, SHAP explainability, and KMeans clustering.

### Risk-Based Retention

Risk-based retention should be used for current customers who are predicted to be at risk.

The main idea is to prioritize customers based on data-driven churn signals rather than applying the same campaign to everyone.

| Risk Level  | Data Insight                                                                                                  | Business Interpretation                                                                                | Recommended Action                                                                           |
| ----------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| High Risk   | Low tenure, high complaint concentration, strong mobile-related category concentration, lower cashback amount | These customers may be new, dissatisfied, or less likely to make repeat purchases without intervention | Prioritize onboarding, complaint recovery, mobile-category cross-sell, and targeted cashback |
| Medium Risk | Mixed product category behavior and moderate churn signals                                                    | These customers are not urgent yet but may become High Risk if engagement declines                     | Monitor behavior, send light incentives, and trigger stronger actions if risk increases      |
| Low Risk    | Higher tenure, lower immediate churn risk, stronger concentration in laptop & accessory                       | These customers are relatively more stable                                                             | Maintain engagement and avoid unnecessary discounting                                        |

Recommended retention actions:

1. Prioritize High Risk customers with complaint history.
2. Build a structured onboarding journey for low-tenure customers.
3. Create mobile-category retention and accessory cross-sell campaigns.
4. Use targeted cashback instead of broad discounts.
5. Monitor Medium Risk customers before they become High Risk.
6. Avoid aggressive discounting for Low Risk customers.
7. Use threshold 0.40 when the business wants to capture more potential churned customers for retention campaigns.

### Cluster-Based Promotion

KMeans clustering should be used for customers who already churned.

While risk segmentation identifies who should be prioritized for retention, KMeans clustering identifies different types of churned customers for personalized win-back campaigns.

For churned customers:

1. Send win-back and loyalty recovery offers to Established High-Value Churned Customers.
2. Send second-purchase and mobile accessory offers to New One-Time Mobile Churned Customers.
3. Avoid sending the same generic promotion to all churned users.
4. Use cluster profiles to design different promotion messages, incentives, and customer journeys.

Full recommendation document:

```text
docs/business_recommendations.md
```
---

# 5. Project Assets

## 5.1 Project Structure

```text
Ecommerce_Churn_Prediction_Machine_Learning_Pipeline/
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
│   ├── tuning/
│   ├── evaluation/
│   ├── prediction/
│   └── segmentation/
│
├── 05_outputs/
│   ├── figures/
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

| Output                             | Path                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------- |
| Model comparison                   | `05_outputs/model_results/model_comparison.csv`                           |
| Random Forest feature importance   | `05_outputs/model_results/random_forest_feature_importance.csv`           |
| Random Forest tuning comparison    | `05_outputs/model_results/random_forest_tuning_comparison.csv`            |
| RandomizedSearchCV results         | `05_outputs/model_results/random_forest_randomized_search_cv_results.csv` |
| Threshold analysis                 | `05_outputs/model_results/threshold_analysis.csv`                         |
| SHAP feature importance            | `05_outputs/model_results/shap_feature_importance.csv`                    |
| Threshold metric comparison figure | `05_outputs/figures/threshold_metric_comparison.png`                      |
| Threshold customer flagging figure | `05_outputs/figures/threshold_customers_flagged.png`                      |
| SHAP summary plot                  | `05_outputs/figures/shap_summary_plot.png`                                |
| Churn predictions                  | `05_outputs/predictions/churn_predictions.csv`                            |
| High Risk customers                | `05_outputs/segments/high_risk_customers.csv`                             |
| KMeans churned user segments       | `05_outputs/segments/churned_user_kmeans_segments.csv`                    |
| Probability calibration summary    | `05_outputs/model_results/probability_calibration_summary.csv`            |
| Probability calibration metric     | `05_outputs/model_results/probability_calibration_metric.csv`             |
| Probability calibration curve      | `05_outputs/figures/probability_calibration_curve.png`                    |
| Risk level numeric profile | `05_outputs/segments/risk_level_numeric_profile.csv`                              |
| Risk level category profile | `05_outputs/segments/risk_level_category_profile.csv`                            |
| Saved final model                  | `06_models/random_forest_churn_model.pkl`                                 |
| SQL business analysis              | `07_sql_analysis/churn_business_queries.sql`                              |
| Business recommendations           | `docs/business_recommendations.md`                                        |

---

## 5.3 Conclusion

This project successfully answers the three main requirements:

1. **EDA**
   Churn is strongly associated with tenure, complaints, customer lifecycle stage, product category, and engagement behavior.

2. **Supervised Learning**
   A Random Forest model was selected to predict customer churn and generate customer risk levels.

3. **Unsupervised Learning**
   KMeans clustering was applied to churned users only and identified two main churned-user groups for targeted promotion strategies.

In addition, the machine learning layer was extended with hyperparameter tuning, cross-validation, threshold analysis, and SHAP explainability. These additions make the model evaluation more robust, more interpretable, and more useful for business decision-making.

The final outputs help the company identify at-risk customers, understand churned-user behavior, and design more personalized retention and win-back campaigns.
