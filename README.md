# Classification Model for Predicting Credit Card Default Payment
## Dataset : Payment Default Prediction
Data source : https://www.kaggle.com/datasets/reverie5/av-janata-hack-payment-default-prediction
## Project Teammates : 
<ol>
 <li><a href="https://www.linkedin.com/in/trisetiawan14ts/">Tri Setiawan</a></li>
 <li><a href="https://www.linkedin.com/in/ulva/">Ulva Dewiyanti</a></li>
 <li><a href="https://www.linkedin.com/in/cristanto99/">Cristanto</a></li>
 <li><a href="https://www.linkedin.com/in/stevenbennyp2/">Steven Benny</a></li>
</ol>

## Project Overview

This project is in the form of making a machine learning model to classify credit card customers will be default or not in the following month by using several machine learning algorithms and followed by tuning hyperparameters to get the optimal model. 

In addition to focusing on model performance, this project also focuses on completing case studies by providing several business recommendations based on the results of data exploration.

## Problem Statement
   
Credit card is a flexible tool by which a customer can use a bank's money for a short period of time and one of the main business of banks. It helps the bank to generate interest revenue but at the same time, it raise the liquidity risk and credit risk to the bank. 

Credit card default happens when **customer have become severely delinquent on credit card payments**. In arrange to increase market share, card-issuing banks in Taiwan **over-issued cash and credit cards to unfit  candidates**. At the same time, most cardholders, irrespective of their repayment ability, the **overused credit card for consumption and accumulated heavy credit and debts**.

Predicting accurately which customers are most probable to default represents a significant business opportunity for all banks. Bank cards are the most common credit card type in Taiwan, which emphasizes the impact of risk prediction on both the consumers and banks.

## Scope of Problem
<ul>
 <li>The data used is taken from Kaggle which consists of 21000 rows, 25 columns, and the data is taken from April 2005 to September 2005,</li>
 <li>The dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card customers.</li>
 <li>The goal is to reduce default rate by building an automated model for identifying the key factors and predicting a credit card default based on the information about the customer and historical transactions, so the bank can give customers some prevention action to prevent default.</li>
</ul>

## Project Environment

The analysis has been completely conducted with Python, by implementing several machine learning and statistical frameworks available such as  `scikit-learn`,  `numpy`,  `pandas`,  `imblearn`, then visualize the EDA by using data visualization libraries such as `matplotlib`  and  `seaborn`.

## Dataset Information

The dataset consists of 21000 rows, 25 variables, and the data is taken from April 2005 to September 2005. The columns devided into some category, such as,

information about the  **client personal information**:

 - variables about **customer demographic**:
	1.  `ID`: unique ID of each customer
	2.  `SEX`: Gender (1=male, 2=female)
	3.  `EDUCATION`: level of education (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
	4.  `MARRIAGE`: Marital status (1=married, 2=single, 3=others)
	5.  `AGE`: Age in years
 - variables about **customer credit data**:
	 1. `LIMIT_BAL`: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
 - variables about **status of the previous payment** for a spesific month:
	1.  `PAY_0`: Repayment status in September 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
	2.  `PAY_2`: Repayment status in August 2005 (same scale as before)
	3.  `PAY_3`: Repayment status in July 2005 (same scale as before)
	4.  `PAY_4`: Repayment status in June 2005 (same scale as before)
	5.  `PAY_5`: Repayment status in May 2005 (same scale as before)
	6.  `PAY_6`: Repayment status in April 2005 (same scale as before)
 - variables about **amount of previous payment** for a spesific month:
	1.  `PAY_AMT1`: Amount of previous payment in September, 2005 (NT dollar)
	2.  `PAY_AMT2`: Amount of previous payment in August, 2005 (NT dollar)
	3.  `PAY_AMT3`: Amount of previous payment in July, 2005 (NT dollar)
	4.  `PAY_AMT4`: Amount of previous payment in June, 2005 (NT dollar)
	5.  `PAY_AMT5`: Amount of previous payment in May, 2005 (NT dollar)
	6.  `PAY_AMT6`: Amount of previous payment in April, 2005 (NT dollar)
 - variables about  **amount of bill statement** for a spesific month:
	1.  `BILL_AMT1`: Amount of bill statement in September, 2005 (NT dollar)
	2.  `BILL_AMT2`: Amount of bill statement in August, 2005 (NT dollar)
	3.  `BILL_AMT3`: Amount of bill statement in July, 2005 (NT dollar)
	4.  `BILL_AMT4`: Amount of bill statement in June, 2005 (NT dollar)
	5.  `BILL_AMT5`: Amount of bill statement in May, 2005 (NT dollar)
	6.  `BILL_AMT6`: Amount of bill statement in April, 2005 (NT dollar)
 -  target variable:
	1.  `default_payment_next_month`: indicate whether the credit card holders are default or not-default (1=yes, 0=no)

## EDA
1. Label default_payment_next_month is not balance
![Imbalance of Label](https://github.com/ulvadewiyanti/classification-model-for-predict-credit-card-default-payment/blob/main/images/Imbalance%20label.png)

## ML Modeling & Evaluation
Before doing the modeling, data preprocessing is carried out as follows,
 - Check for duplicate data and null or empty data in each column.
 - Checking for outliers in each column followed by handling using a Z-score with a value of 3 Standardize numeric data using StandardScaler.
 - Adjusting column names and handling some undocumented categories. 
 - Perform encoding labels on sex and education features, as well as one hot encoding on other categorical features.
 - Do a train test split with a ratio of 7:3.
 - Balancing the labels using a SMOTE of 0.5 on the train dataset only.
 
After the data preprocessing is complete, it is followed by modeling experiments using different datasets, different algorithms, and accompanied by hyperparameter tuning.

The dataset used will be divided into several categories, namely 
 - the original dataset, 
 - the original dataset after the outliers were removed, 
 - the standardized original dataset, and 
 - the dataset after the outliers were removed, standardized, and the presence of one hot encoding.

Classifications algorithm used is,
 - Logistic Regression, 
 - KNN, 
 - Decision Tree, 
 - Random Forest, 
 - AdaBoost, and 
 - XGBoost.

The evaluation metric used is **AUC-ROC** as primary metric, recall & precision as secondary metric for the following reasons,
 1. Technical Reasons
	- Dataset imbalance with proportion almost 8:2 
	- Focusing on reducing the number of false negatives and false positives
	- The accuracy value becomes less representative because a lot of synthetic data is the result of oversampling

2. Business Side Reasons
	- The main purpose of the model is to predict the number of defaults as much as possible so that handling actions can be given, so it is necessary to pay attention to the recall score.
	- However, we must also pay attention to debtors who should not default but are predicted to default (false positive), because if a default is predicted but it is not, then the debtor will be given treatment as a default, this of course can trigger the debtor's discomfort which can lead to complaints or churn, so score precision also needs to be considered.

Here are the top 4 model,
|  Agorithm|AUC-ROC|Precision|Recall|
|--|--|--|--|
| XGBoost | 0.77 | 0.67 | 0.37 |
| XGBoost | 0.77 | 0.67 | 0.34 |
| **XGBoost** | **0.78** | **0.66** | **0.39** |
| AdaBoost| 0.77 | 0.67 | 0.34 |

The best model is **XGBoost** and has been oversampled, where the model has the highest AUC and Recall values.

## Business Insight & Recomendations
