---
layout: post
author: Lakshmi Balasubramaniam
tags: [cscc]
---

### [Machine Learning II Final Project: Parkinson's Disease Detection]

## Overview
This project demonstrates an end-to-end ML Solution using Neural Networks and Deep Learning, applied to the Parkinson’s Disease Detection dataset originally contributed by **Max A. Little** and colleagues. It is available from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/174/parkinsons) and licensed under **CC BY 4.0**.

Proper credit is given to the original authors:  
> **Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig**  
> “Suitability of dysphonia measurements for telemonitoring of Parkinson's disease,” IEEE Transactions on Biomedical Engineering, 2008.

According to the UCI ML Repository, the dataset "is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD"

## Step 1: Define the Business Problem
- What is the goal? Why does it matter?
- State if it's: Supervised, Unsupervised, Recommender, or Reinforcement and why you chose that method

Goal: Predict whether a patient has Parkinson’s Disease using biomedical voice measurements, enabling early diagnosis and proactive medical care

Why this matters: Parkinson’s is a progressive disease and diagnosing early helps in getting timely medical intervention and monitoring

ML Type: Supervised Learning (Binary Classification) since the model is trained on labeled data and predicts a binary outcome:
- 0 = healthy
- 1 = Parkinson’s Disease

***
## Step 2: Data Acquisition & EDA
- Show key insights and visualizations

![Model Accuracy Comparison](/assets/images/Target.variable.distribution.pie_bar.chart_output.png)

![Model Accuracy Comparison](/assets/images/Correlation.matrix_output.png)

***
## Step 3: Data Cleaning
- Handle missing values, transformations, scaling

***
## Step 4: Modeling
- Build one or more models aligned with your problem type

***
## Step 5: Model Evaluation
- Report metrics
- Show how you improved it
- Write a short paragraph with your model assessment

***
## Step 6: Deployment Plan
- Briefly describe how you could deploy your model (e.g., API, web app, embedded system)

***
