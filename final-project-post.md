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
#### Developed the final project in Visual Studio Code using a Jupyter Notebook (.ipynb) with the Python 3.10.11 kernel

## Import statements
This project uses the following import statements:
```python
# Data Handling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling and Evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score, classification_report
```

To import the libraries, run:
```python
%pip install pandas matplotlib seaborn scikit-learn tensorflow
```

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
### Code:
```python
# Summary Statistics for all columns transposed
df.describe(include='all').T
```

### Results:
```python
          count  unique              top  freq        mean        std        min       25%       50%       75%        max
name     195.0    195.0  phon_R01_S01_1   1.0         NaN        NaN        NaN       NaN       NaN       NaN        NaN
MDVP:Fo(Hz)     195.0      NaN              NaN   NaN  154.228641  41.390065   88.333  117.572   148.79  182.769   260.105
MDVP:Fhi(Hz)    195.0      NaN              NaN   NaN  197.104918  91.491548  102.145  134.8625  175.829  224.2055  592.03
MDVP:Flo(Hz)    195.0      NaN              NaN   NaN  116.324631  43.521413   65.476   84.291  104.315  140.0185  239.17
MDVP:Jitter(%)  195.0      NaN              NaN   NaN    0.00622   0.004848   0.00168   0.00346   0.00494   0.007365   0.03316
MDVP:Jitter(Abs)195.0      NaN              NaN   NaN   0.000044   0.000035   0.000007  0.00002  0.00003  0.00006   0.00026
MDVP:RAP        195.0      NaN              NaN   NaN   0.003306   0.002968   0.00068  0.00166  0.0025   0.003835  0.02144
MDVP:PPQ        195.0      NaN              NaN   NaN   0.003446   0.002759   0.00092  0.00186  0.00269  0.003955  0.01958
Jitter:DDP      195.0      NaN              NaN   NaN   0.00992    0.008903   0.00204  0.004985  0.00749  0.011505  0.06433
MDVP:Shimmer    195.0      NaN              NaN   NaN   0.029709   0.018857   0.00954  0.016505  0.02297  0.037885  0.11908
MDVP:Shimmer(dB)195.0      NaN              NaN   NaN   0.282251   0.194877   0.085    0.1485   0.221    0.35     1.302
Shimmer:APQ3    195.0      NaN              NaN   NaN   0.015664   0.010153   0.00455  0.008245  0.01279  0.020265  0.05647
Shimmer:APQ5    195.0      NaN              NaN   NaN   0.017878   0.012024   0.0057   0.00958   0.01347  0.02238   0.0794
MDVP:APQ        195.0      NaN              NaN   NaN   0.024081   0.016947   0.00719  0.01308   0.01826  0.0294    0.13778
Shimmer:DDA     195.0      NaN              NaN   NaN   0.046993   0.030459   0.01364  0.024735  0.03836  0.060795  0.16942
NHR             195.0      NaN              NaN   NaN   0.024847   0.040418   0.00065  0.005925  0.01166  0.02564   0.31482
HNR             195.0      NaN              NaN   NaN  21.885974   4.425764   8.441    19.198   22.085   25.0755  33.047
status          195.0      NaN              NaN   NaN   0.753846   0.431878   0.0      1.0      1.0      1.0      1.0
RPDE            195.0      NaN              NaN   NaN   0.498536   0.103942   0.25657  0.421306  0.495954 0.587562 0.685151
DFA             195.0      NaN              NaN   NaN   0.718099   0.055336   0.574282 0.674758 0.722254 0.761881 0.825288
spread1         195.0      NaN              NaN   NaN  -5.684397   1.090208  -7.964984 -6.450096 -5.720868 -5.046192 -2.434031
spread2         195.0      NaN              NaN   NaN   0.22651    0.083406   0.006274 0.174351 0.218885 0.279234 0.450493
D2              195.0      NaN              NaN   NaN   2.381826   0.382799   1.423287 2.099125 2.361532 2.636456 3.671155
PPE             195.0      NaN              NaN   NaN   0.206552   0.090119   0.044539 0.137451 0.194052 0.25298   0.527367
```

### Target Variable > Status
- 0 for healthy
- 1 for Parkinson’s

### Code:
```python
# Target variable distribution pie/bar chart
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='status', data=df)
plt.title("Class Distribution (0 = Healthy, 1 = Parkinson's)")
plt.tight_layout()
plt.show()
```

### Results:
![Class Distribution Bar Chart](/assets/images/df_Target.variable.distribution.pie_bar.chart_output.png)

The bar chart displays the count of samples for each status category:  
- **Parkinson’s Disease (1)** → ~147 samples  
- **Healthy (0)** → ~50 samples  
- There is a class imbalance, which should be considered during model evaluation

### Code:
```python
# Separate columns by type
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols, numerical_cols
```

### Results:
```python
(['name'],
 ['MDVP:Fo(Hz)',
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
  'PPE'])
```

### Code:
```python
# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```

### Results:
![Model Accuracy Comparison](/assets/images/df_Correlation.matrix_output.png)

This heatmap visualizes the correlation between all numerical features in the dataset
- Warm colors (red) represent strong positive correlations
- Cool colors (blue) indicate negative or low correlations
- High correlation between features like `MDVP:Jitter(%)`, `MDVP:RAP`, and `Jitter:DDP` suggests potential multicollinearity
- These relationships may influence feature selection or regularization in modeling

***
## Step 3: Data Cleaning
- Handle missing values, transformations, scaling

### Code:
```python
# Drop identifier column of name 
df.drop(['name'], axis=1, inplace=True)
```

- Identifier column **'name'** removed
- Dataset now contains **23 numerical features** suitable for modeling

### Code:
```python
# Check for Missing Values
df.isnull().sum()
```

### Results:
```python
MDVP:Fo(Hz)         0
MDVP:Fhi(Hz)        0
MDVP:Flo(Hz)        0
MDVP:Jitter(%)      0
MDVP:Jitter(Abs)    0
MDVP:RAP            0
MDVP:PPQ            0
Jitter:DDP          0
MDVP:Shimmer        0
MDVP:Shimmer(dB)    0
Shimmer:APQ3        0
Shimmer:APQ5        0
MDVP:APQ            0
Shimmer:DDA         0
NHR                 0
HNR                 0
status              0
RPDE                0
DFA                 0
spread1             0
spread2             0
D2                  0
PPE                 0
dtype: int64
```

- No missing values found in any column
- Dataset is clean and ready for scaling and modeling

### Code:
```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(df.drop(['status'], axis=1))
y = df['status']
```

All numerical features (excluding `status`) scaled using `StandardScaler`  
- `X` now contains the transformed feature matrix  
- `y` retains the binary target labels for classification

***
## Step 4: Modeling
- Build one or more models aligned with your problem type

### Code:
```python
# Preprocess data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

- Dataset is split into the training and testing sets using `train_test_split`  
- `X_train` and `X_test` contain scaled feature inputs  
- `y_train` and `y_test` contain binary classification labels

### Code:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Ensure only features are included (drop 'status' and 'name' if they exist)
features = [col for col in df.columns if col not in ['name', 'status']]
X_scaled = scaler.fit_transform(df[features])

# Set up features and target
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['status'], test_size=0.2, stratify=df['status'], random_state=42)

# Define the model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)
```

### Results:
```
C:\Users\...\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 35ms/step - accuracy: 0.5801 - loss: 0.7088 - val_accuracy: 0.7188 - val_loss: 0.5532
Epoch 2/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - accuracy: 0.7681 - loss: 0.5357 - val_accuracy: 0.8750 - val_loss: 0.4608
Epoch 3/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8526 - loss: 0.4670 - val_accuracy: 0.9062 - val_loss: 0.4013
Epoch 4/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.8298 - loss: 0.4112 - val_accuracy: 0.9062 - val_loss: 0.3615
Epoch 5/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8129 - loss: 0.4010 - val_accuracy: 0.9062 - val_loss: 0.3358
Epoch 6/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8016 - loss: 0.3834 - val_accuracy: 0.9062 - val_loss: 0.3128
Epoch 7/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8678 - loss: 0.3251 - val_accuracy: 0.9062 - val_loss: 0.2915
Epoch 8/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.8561 - loss: 0.3153 - val_accuracy: 0.9062 - val_loss: 0.2773
Epoch 9/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8467 - loss: 0.3150 - val_accuracy: 0.9375 - val_loss: 0.2643
Epoch 10/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8679 - loss: 0.2920 - val_accuracy: 0.9375 - val_loss: 0.2539
Epoch 11/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8966 - loss: 0.2818 - val_accuracy: 0.9375 - val_loss: 0.2433
Epoch 12/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9265 - loss: 0.2605 - val_accuracy: 0.9375 - val_loss: 0.2310
Epoch 13/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8976 - loss: 0.2871 - val_accuracy: 0.9375 - val_loss: 0.2197
Epoch 14/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9155 - loss: 0.2862 - val_accuracy: 0.9375 - val_loss: 0.2150
Epoch 15/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9186 - loss: 0.2395 - val_accuracy: 0.9062 - val_loss: 0.2102
Epoch 16/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9424 - loss: 0.2159 - val_accuracy: 0.9062 - val_loss: 0.2018
Epoch 17/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9488 - loss: 0.1926 - val_accuracy: 0.9062 - val_loss: 0.2013
Epoch 18/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9557 - loss: 0.1970 - val_accuracy: 0.9062 - val_loss: 0.1974
Epoch 19/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9334 - loss: 0.1913 - val_accuracy: 0.9062 - val_loss: 0.1906
Epoch 20/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9459 - loss: 0.1809 - val_accuracy: 0.9062 - val_loss: 0.1858
<keras.src.callbacks.history.History at 0x25acca5fa00>
```

***
## Step 5: Model Evaluation
- Report metrics
- Show how you improved it
- Write a short paragraph with your model assessment

### Code:
```python
# Report metrics / Evaluate the model
from sklearn.metrics import accuracy_score

preds = (model.predict(X_test) > 0.5).astype('int32')
print("TensorFlow Test Accuracy:", accuracy_score(y_test, preds))
```

### Results:
```
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step  
TensorFlow Test Accuracy: 0.8974358974358975
```

- Final test accuracy: **89.74%**  
- This result confirms strong generalization on unseen data

### Code:
```python
# Improved model
from tensorflow.keras.layers import Dropout

# Ensure only features are included (drop 'status' and 'name' if they exist)
features = [col for col in df.columns if col not in ['name', 'status']]
target_column = 'status'

# Scale Features
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(df[features])
y_target = df[target_column]

# Train-Test split
X_train_scaled, X_test_scaled, y_train_target, y_test_target = train_test_split(
    X_features_scaled, y_target, test_size=0.2, stratify=y_target, random_state=42
)

# Define the model with Dropout
model_dropout = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history_dropout = model_dropout.fit(
    X_train_scaled, y_train_target,
    epochs=20,
    batch_size=16,
    validation_split=0.2
)

from sklearn.metrics import accuracy_score, classification_report

# Make Predictions
y_pred_probs = model_dropout.predict(X_test_scaled)
y_pred_class = (y_pred_probs > 0.5).astype('int32')

# Evaluation Metrics
print("Dropout Model Test Accuracy:", accuracy_score(y_test_target, y_pred_class))
```

### Results:
```
Epoch 1/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - accuracy: 0.7638 - loss: 0.5445 - val_accuracy: 0.9062 - val_loss: 0.4461
Epoch 2/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7187 - loss: 0.5532 - val_accuracy: 0.8750 - val_loss: 0.4006
Epoch 3/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.7381 - loss: 0.4761 - val_accuracy: 0.9062 - val_loss: 0.3647
Epoch 4/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.7621 - loss: 0.4490 - val_accuracy: 0.9062 - val_loss: 0.3366
Epoch 5/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.7295 - loss: 0.4354 - val_accuracy: 0.9062 - val_loss: 0.3132
Epoch 6/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.7666 - loss: 0.4364 - val_accuracy: 0.9375 - val_loss: 0.2977
Epoch 7/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.8289 - loss: 0.3316 - val_accuracy: 0.9375 - val_loss: 0.2833
Epoch 8/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8571 - loss: 0.3174 - val_accuracy: 0.9375 - val_loss: 0.2667
Epoch 9/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8729 - loss: 0.2907 - val_accuracy: 0.9375 - val_loss: 0.2551
Epoch 10/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8787 - loss: 0.3078 - val_accuracy: 0.9062 - val_loss: 0.2486
Epoch 11/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8515 - loss: 0.2832 - val_accuracy: 0.9062 - val_loss: 0.2392
Epoch 12/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8247 - loss: 0.3187 - val_accuracy: 0.9375 - val_loss: 0.2306
Epoch 13/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8829 - loss: 0.3096 - val_accuracy: 0.9375 - val_loss: 0.2234
Epoch 14/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8559 - loss: 0.3144 - val_accuracy: 0.9062 - val_loss: 0.2173
Epoch 15/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8441 - loss: 0.2960 - val_accuracy: 0.9375 - val_loss: 0.2072
Epoch 16/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9259 - loss: 0.2393 - val_accuracy: 0.9375 - val_loss: 0.1962
Epoch 17/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.8578 - loss: 0.2696 - val_accuracy: 0.9375 - val_loss: 0.1915
Epoch 18/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8435 - loss: 0.2809 - val_accuracy: 0.9375 - val_loss: 0.1858
Epoch 19/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 0.9121 - loss: 0.2611 - val_accuracy: 0.9375 - val_loss: 0.1813
Epoch 20/20
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.9387 - loss: 0.2165 - val_accuracy: 0.9688 - val_loss: 0.1791
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step  
Dropout Model Test Accuracy: 0.8974358974358975
```

### Model Assessment
To improve generalization and reduce overfitting, a `Dropout` layer was introduced after the first dense layer. While the validation accuracy peaked at **96.88%**, the final test accuracy settled at **89.74%**, matching the baseline model's generalization. The inclusion of dropout enhanced the model’s resistance to overfitting, especially visible through the stability of validation loss. This trade-off reflects a balanced architecture that maintains strong predictive power while prioritizing reliability on unseen data.

***
## Step 6: Deployment Plan
- Briefly describe how you could deploy your model (e.g., API, web app, embedded system)

A way that could deploy the model is via a RESTful web app built using **Streamlit** or **Gradio**.

These frameworks provide lightweight, interactive interfaces where users can:
- Manually input biomedical voice features
- Upload files containing voice analysis metrics
- Receive **real-time predictions** on Parkinson’s disease risk using the trained neural network

Both tools support model hosting locally or on cloud platforms such as **Heroku**, **Render**, or **Hugging Face Spaces**, making them ideal for rapid prototyping and public access. Future enhancements could include:
- User authentication for secure data input  
- Integration with a backend API for automated file parsing and preprocessing  
- Embedding into mobile or desktop diagnostic tools for clinical use

***
