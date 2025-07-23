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

***
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

### Code:
```python
# Import pandas library
import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)

# Display the first few rows
df.head()
```

I got the data from the UCI ML Repository and pulled the data in from the url, which has a Public Domain license. As shown above, the dataset was read in directly through the site.

### Results:
```
    name    MDVP:Fo(Hz) MDVP:Fhi(Hz)    MDVP:Flo(Hz)    MDVP:Jitter(%)  MDVP:Jitter(Abs)    MDVP:RAP    MDVP:PPQ    Jitter:DDP  MDVP:Shimmer    ... Shimmer:DDA NHR HNR status  RPDE    DFA spread1 spread2 D2  PPE
0   phon_R01_S01_1  119.992 157.302 74.997  0.00784 0.00007 0.00370 0.00554 0.01109 0.04374 ... 0.06545 0.02211 21.033  1   0.414783    0.815285    -4.813031   0.266482    2.301442    0.284654
1   phon_R01_S01_2  122.400 148.650 113.819 0.00968 0.00008 0.00465 0.00696 0.01394 0.06134 ... 0.09403 0.01929 19.085  1   0.458359    0.819521    -4.075192   0.335590    2.486855    0.368674
2   phon_R01_S01_3  116.682 131.111 111.555 0.01050 0.00009 0.00544 0.00781 0.01633 0.05233 ... 0.08270 0.01309 20.651  1   0.429895    0.825288    -4.443179   0.311173    2.342259    0.332634
3   phon_R01_S01_4  116.676 137.871 111.366 0.00997 0.00009 0.00502 0.00698 0.01505 0.05492 ... 0.08771 0.01353 20.644  1   0.434969    0.819235    -4.117501   0.334147    2.405554    0.368975
4   phon_R01_S01_5  116.014 141.781 110.655 0.01284 0.00011 0.00655 0.00908 0.01966 0.06425 ... 0.10470 0.01767 19.649  1   0.417356    0.823484    -3.747787   0.234513    2.332180    0.410335
5 rows × 24 columns
```

### Key Insights
- 195 rows & 24 columns
- No missing values
- The name column has 195 unique values, confirming it is an identifier
- Parkinson’s dominates with around 147 samples

### Code:
```python
# Feature Names List
df.columns.tolist()
```

### Results:
``` python
['name',
 'MDVP:Fo(Hz)',
 'MDVP:Fhi(Hz)',
 'MDVP:Flo(Hz)',
 'MDVP:Jitter(%)',
 'MDVP:Jitter(Abs)',
 'MDVP:RAP',
 'MDVP:PPQ',
 'Jitter:DDP',
 'MDVP:Shimmer',
 'MDVP:Shimmer(dB)',
 'Shimmer:APQ3',
 'Shimmer:APQ5',
 'MDVP:APQ',
 'Shimmer:DDA',
 'NHR',
 'HNR',
 'status',
 'RPDE',
 'DFA',
 'spread1',
 'spread2',
 'D2',
 'PPE']
```

### Code:
```python
# Shape of data
df.shape
```

### Results:
```python
(195, 24)
```

Data has 195 rows and 24 columns

### Code:
```python
# Data types in dataset
df.dtypes.value_counts()
```

### Results:
```python
float64    22
object      1
int64       1
Name: count, dtype: int64
```

### Code:
```python
# Inspect the dataset
df.info()
```

### Results:
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 195 entries, 0 to 194
Data columns (total 24 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   name              195 non-null    object 
 1   MDVP:Fo(Hz)       195 non-null    float64
 2   MDVP:Fhi(Hz)      195 non-null    float64
 3   MDVP:Flo(Hz)      195 non-null    float64
 4   MDVP:Jitter(%)    195 non-null    float64
 5   MDVP:Jitter(Abs)  195 non-null    float64
 6   MDVP:RAP          195 non-null    float64
 7   MDVP:PPQ          195 non-null    float64
 8   Jitter:DDP        195 non-null    float64
 9   MDVP:Shimmer      195 non-null    float64
 10  MDVP:Shimmer(dB)  195 non-null    float64
 11  Shimmer:APQ3      195 non-null    float64
 12  Shimmer:APQ5      195 non-null    float64
 13  MDVP:APQ          195 non-null    float64
 14  Shimmer:DDA       195 non-null    float64
 15  NHR               195 non-null    float64
 16  HNR               195 non-null    float64
 17  status            195 non-null    int64  
 18  RPDE              195 non-null    float64
 19  DFA               195 non-null    float64
 20  spread1           195 non-null    float64
 21  spread2           195 non-null    float64
 22  D2                195 non-null    float64
 23  PPE               195 non-null    float64
dtypes: float64(22), int64(1), object(1)
memory usage: 36.7+ KB
```

![Model Accuracy Comparison](/assets/images/df_Target.variable.distribution.pie_bar.chart_output.png)

![Model Accuracy Comparison](/assets/images/df_Correlation.matrix_output.png)

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
