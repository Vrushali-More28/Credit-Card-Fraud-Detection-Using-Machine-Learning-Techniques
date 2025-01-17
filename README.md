# Credit Card Fraud Detection Using Machine Learning Techniques

## Overview
This repository contains the independent study report for the project "Credit Card Fraud Detection Using Machine Learning Techniques" conducted at the State University of New York, Binghamton. The project aims to detect fraudulent credit card transactions using various machine learning algorithms and evaluates their performance based on classification metrics.

## Introduction
### Credit Card Frauds
Credit card frauds have been increasing with the rise of cashless transactions. Fraudulent activities, including lost/stolen card fraud, skimming, false application fraud, data breaches, and mail intercept fraud, result in significant financial losses for banks and customers.

## Importance of Credit Card Fraud Detection
Detecting fraudulent transactions is crucial to minimize losses. Machine learning techniques, particularly data mining, can be used to identify patterns in transaction data and improve the accuracy of fraud detection models.

# Classification Metrics
## Confusion Matrix
The confusion matrix is a method to evaluate the performance of a classification model. It consists of True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN).

1. Accuracy
Accuracy measures the number of correct predictions made by the model. 
2. Precision
Precision is the ratio of correctly predicted positive observations to the total predicted positives. 
3. Recall
Recall is the ratio of correctly predicted positive observations to all observations in the actual class. 
4. F-1 Score
The F-1 score is the harmonic mean of precision and recall.
​
## Steps of Implementation
1. Gathering Data: Collect relevant data for the problem.
2. Preprocessing the Data: Clean and prepare the data for training.
3. Splitting the Dataset: Divide the data into training and testing sets.
4. Choosing a Model: Select appropriate machine learning algorithms.
5. Evaluating the Model: Use metrics to assess the model's performance.

## Methodology
## About Dataset
The dataset used is from Kaggle, containing transaction details of European cardholders in September 2013. The data includes 284,807 instances with 31 columns, anonymized for confidentiality.

## Classification Algorithms
Three classification algorithms were used: Isolation Forest, Random Forest, and Decision Tree Classifier. Each algorithm was evaluated using the same parameters for a fair comparison.

## Validation of Data
Both train-test split and cross-validation techniques were used for validation. A 70-30 split ratio was applied for training and testing the models.

## Results
The results showed that Random Forest performed the best among the three algorithms, with an accuracy of 0.9994, precision, recall, and F-1 score significantly improved after feature selection and parameter tuning.

## Conclusion
Random Forest emerged as the optimal model for credit card fraud detection, demonstrating superior performance in accuracy, precision, recall, and F-1 score compared to Isolation Forest and Decision Tree Classifier.