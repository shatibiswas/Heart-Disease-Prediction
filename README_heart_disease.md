# **# Project: Heart Disease Prediction using Logistic Regression**

# **Objective:**

## Develop a machine learning model using Logistic Regression to predict the 10-year risk of Coronary Heart Disease (CHD) in patients based on health metrics.

# Importing Necessary Libraries and Dataset:
Data Insigts


```python
# 1. Importing Necessary Libraries and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Load the dataset
df = pd.read_csv('framingham.csv')

# Drop unnecessary column
df.drop('education', axis=1, inplace=True)

# Rename columns for better readability
df.rename(columns={'male': 'is_male', 'age': 'age_years', 'currentSmoker': 'is_smoker'}, inplace=True)

# Check data structure
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4240 entries, 0 to 4239
    Data columns (total 15 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   is_male          4240 non-null   int64  
     1   age_years        4240 non-null   int64  
     2   is_smoker        4240 non-null   int64  
     3   cigsPerDay       4211 non-null   float64
     4   BPMeds           4187 non-null   float64
     5   prevalentStroke  4240 non-null   int64  
     6   prevalentHyp     4240 non-null   int64  
     7   diabetes         4240 non-null   int64  
     8   totChol          4190 non-null   float64
     9   sysBP            4240 non-null   float64
     10  diaBP            4240 non-null   float64
     11  BMI              4221 non-null   float64
     12  heartRate        4239 non-null   float64
     13  glucose          3852 non-null   float64
     14  TenYearCHD       4240 non-null   int64  
    dtypes: float64(8), int64(7)
    memory usage: 497.0 KB


**#Data Preprocessing**


```python

# Drop rows with missing values
df.dropna(inplace=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['age_years', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split dataset
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

# Exploratory Data Analysis (EDA)


```python

sns.countplot(x='TenYearCHD', data=df)
plt.title('Class Distribution')
plt.show()

df.hist(bins=20, figsize=(15,10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_7_0.png)
    



    
![png](output_7_1.png)
    



    
![png](output_7_2.png)
    


# Correlation Matrix Summary
The correlation heatmap reveals the relationships between different features in the dataset. Key observations include:

Age and Blood Pressure: As expected, there is a strong positive correlation between age (age_years) and both systolic blood pressure (sysBP) and diastolic blood pressure (diaBP).

Smoking and Cigarettes per Day: There is a high positive correlation between being a smoker (is_smoker) and the number of cigarettes smoked per day (cigsPerDay).

Blood Pressure and Cholesterol: Systolic blood pressure (sysBP) shows a moderate positive correlation with total cholesterol (totChol).

Blood Pressure and BMI: Both systolic (sysBP) and diastolic (diaBP) blood pressure have a positive correlation with Body Mass Index (BMI).

Heart Rate and Blood Pressure: Heart rate (heartRate) shows a weak positive correlation with systolic blood pressure (sysBP).

Glucose and Diabetes: There is a positive correlation between glucose levels (glucose) and diabetes (diabetes).

Predicting CHD: The target variable (TenYearCHD) shows some positive correlation with age_years, sysBP, diaBP, and glucose, suggesting these features are important predictors of heart disease.

There is a negative correlation with is_male, indicating that females have a higher risk in this dataset after controlling for other factors (or perhaps due to confounding variables not included).


```python
# Visualize the class distribution of the target variable
sns.countplot(x='TenYearCHD', data=df)
plt.title('Distribution of TenYearCHD')
plt.show()
```


    
![png](output_9_0.png)
    


## Install imbalanced-learn

### Subtask:
Install the library that provides SMOTE.


**Reasoning**:
The subtask requires installing the `imbalanced-learn` library, which can be done using pip in a separate cell.




```python
%pip install imbalanced-learn
```

    Requirement already satisfied: imbalanced-learn in /usr/local/lib/python3.11/dist-packages (0.13.0)
    Requirement already satisfied: numpy<3,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (2.0.2)
    Requirement already satisfied: scipy<2,>=1.10.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (1.16.0)
    Requirement already satisfied: scikit-learn<2,>=1.3.2 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (1.6.1)
    Requirement already satisfied: sklearn-compat<1,>=0.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (0.1.3)
    Requirement already satisfied: joblib<2,>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (1.5.1)
    Requirement already satisfied: threadpoolctl<4,>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (3.6.0)


## Apply smote

### Subtask:
Apply SMOTE to the training data to oversample the minority class.


**Reasoning**:
Apply SMOTE to the training data to address class imbalance.




```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

## Train the model

### Subtask:
Train the logistic regression model using the balanced training data.


**Reasoning**:
Train the logistic regression model using the balanced training data.




```python
model_resampled = LogisticRegression(max_iter=1000, class_weight='balanced')
model_resampled.fit(X_train_resampled, y_train_resampled)
```




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-4 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LogisticRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=1000)</pre></div> </div></div></div></div>



## Evaluate the model

### Subtask:
Evaluate the performance of the trained model.


**Reasoning**:
Evaluate the performance of the trained model by calculating and visualizing various metrics.




```python

```

    Accuracy (Resampled Model): 0.6820603907637656
    Precision (Resampled Model): 0.2926829268292683
    Recall (Resampled Model): 0.6382978723404256
    F1 Score (Resampled Model): 0.4013377926421405



    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    


    ROC AUC Score  Model): 0.7088644921290206



```python
print('Train Accuracy (Resampled Model):', model_resampled.score(X_train_resampled, y_train_resampled))
print('Test Accuracy (Resampled Model):', model_resampled.score(X_test, y_test))
```

    Train Accuracy (Resampled Model): 0.6912092815707274
    Test Accuracy (Resampled Model): 0.6820603907637656


# Example of predicting on new data


```python

# In a real-world scenario, I would load new patient data and preprocess it similarly to the training data


sample_new_patients = X_test.sample(5, random_state=42)

# Predict the probability of CHD
predicted_probabilities = best_model.predict_proba(sample_new_patients)[:, 1]

# Predict the class (0: No CHD, 1: CHD)
predicted_classes = best_model.predict(sample_new_patients)

print("Sample New Patient Data:")
display(sample_new_patients)
print("\nPredicted Probabilities of CHD:")
print(predicted_probabilities)
print("\nPredicted Class (0: No CHD, 1: CHD):")
print(predicted_classes)
```

    Sample New Patient Data:




  <div id="df-5eae8c8d-0cdd-4861-bb64-56d69e2bac3d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>is_male</th>
      <th>age_years</th>
      <th>is_smoker</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2628</th>
      <td>0</td>
      <td>0.049778</td>
      <td>0</td>
      <td>-0.755527</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.683015</td>
      <td>1.480320</td>
      <td>0.591848</td>
      <td>1.146777</td>
      <td>-0.309841</td>
      <td>0.046902</td>
    </tr>
    <tr>
      <th>532</th>
      <td>0</td>
      <td>-1.000512</td>
      <td>1</td>
      <td>-0.420055</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.365926</td>
      <td>-0.878643</td>
      <td>-0.665361</td>
      <td>-0.865496</td>
      <td>0.359347</td>
      <td>-1.125674</td>
    </tr>
    <tr>
      <th>2564</th>
      <td>1</td>
      <td>0.516574</td>
      <td>0</td>
      <td>-0.755527</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.670947</td>
      <td>0.618392</td>
      <td>1.262360</td>
      <td>-0.535857</td>
      <td>-0.895380</td>
      <td>0.381923</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0</td>
      <td>1.450165</td>
      <td>0</td>
      <td>-0.755527</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.046451</td>
      <td>-0.288902</td>
      <td>-0.162478</td>
      <td>0.871258</td>
      <td>1.614075</td>
      <td>-0.497509</td>
    </tr>
    <tr>
      <th>4044</th>
      <td>0</td>
      <td>-1.233910</td>
      <td>0</td>
      <td>-0.755527</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.939971</td>
      <td>-0.130126</td>
      <td>0.885197</td>
      <td>2.221792</td>
      <td>0.777590</td>
      <td>0.130657</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5eae8c8d-0cdd-4861-bb64-56d69e2bac3d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5eae8c8d-0cdd-4861-bb64-56d69e2bac3d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5eae8c8d-0cdd-4861-bb64-56d69e2bac3d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-16ae76a2-d975-4092-ac73-2b503f959b24">
      <button class="colab-df-quickchart" onclick="quickchart('df-16ae76a2-d975-4092-ac73-2b503f959b24')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-16ae76a2-d975-4092-ac73-2b503f959b24 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_d560bb77-280b-4c67-bd41-bafcd51d2c2f">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('sample_new_patients')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d560bb77-280b-4c67-bd41-bafcd51d2c2f button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('sample_new_patients');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    Predicted Probabilities of CHD:
    [0.57466614 0.25246602 0.56303957 0.57162026 0.29530552]
    
    Predicted Class (0: No CHD, 1: CHD):
    [1 0 1 1 0]



```python

```

# Improvement in the existing model

## GridSearchCV in Logistic Refression


```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(max_iter=5000, class_weight='balanced'), param_grid, cv=5, scoring='recall')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and the best score
print("Best Parameters:", grid_search.best_params_)
print("Best Recall Score:", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_
```

    Best Parameters: {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}
    Best Recall Score: 0.7318227012408526



```python

```


```python

```

# Future Improvements

Based on the current evaluation, here are some potential areas for future improvement:

1.  **Explore other resampling techniques**: Investigate other techniques like Undersampling (e.g., RandomUnderSampler) or combination methods (e.g., SMOTE-ENN, SMOTE-Tomek) to see if they yield better results than SMOTE alone.
2.  **Hyperparameter Tuning**: Optimize the hyperparameters of the Logistic Regression model using techniques like GridSearchCV or RandomizedSearchCV to potentially improve performance.
3.  **Try different classification algorithms**: Experiment with other classification algorithms suitable for imbalanced datasets, such as:
    *   Decision Trees
    *   Random Forests
    *   Gradient Boosting Machines (e.g., XGBoost, LightGBM)
    *   Support Vector Machines (SVM)
    *   Ensemble methods
4.  **Feature Engineering**: Create new features or modify existing ones based on domain knowledge or further exploration of the data to potentially improve the model's predictive power.


## Evaluate the Best Model

Evaluate the performance of the best logistic regression model found through hyperparameter tuning.


```python
y_pred_best_model = best_model.predict(X_test)
y_prob_best_model = best_model.predict_proba(X_test)[:, 1]

print('Accuracy (Best Model):', accuracy_score(y_test, y_pred_best_model))
print('Precision (Best Model):', precision_score(y_test, y_pred_best_model))
print('Recall (Best Model):', recall_score(y_test, y_pred_best_model))
print('F1 Score (Best Model):', f1_score(y_test, y_pred_best_model))

cm_best_model = confusion_matrix(y_test, y_pred_best_model)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_best_model, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Best Model)')
plt.show()

fpr_best_model, tpr_best_model, _ = roc_curve(y_test, y_prob_best_model)
plt.figure(figsize=(6, 4))
plt.plot(fpr_best_model, tpr_best_model, label='ROC Curve (Best Model)')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Best Model)')
plt.legend()
plt.show()

print('ROC AUC Score (Best Model):', roc_auc_score(y_test, y_prob_best_model))
```

    Accuracy (Best Model): 0.6447602131438721
    Precision (Best Model): 0.27253218884120173
    Recall (Best Model): 0.675531914893617
    F1 Score (Best Model): 0.38837920489296635



    
![png](output_33_1.png)
    



    
![png](output_33_2.png)
    


    ROC AUC Score (Best Model): 0.7138689153019099


Improvement for better genelization





```python
pip install imbalanced-learn
```

    Requirement already satisfied: imbalanced-learn in /usr/local/lib/python3.11/dist-packages (0.13.0)
    Requirement already satisfied: numpy<3,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (2.0.2)
    Requirement already satisfied: scipy<2,>=1.10.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (1.16.0)
    Requirement already satisfied: scikit-learn<2,>=1.3.2 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (1.6.1)
    Requirement already satisfied: sklearn-compat<1,>=0.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (0.1.3)
    Requirement already satisfied: joblib<2,>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (1.5.1)
    Requirement already satisfied: threadpoolctl<4,>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from imbalanced-learn) (3.6.0)


# Code for SMOTE + Random Forest + Evaluation


```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix (Random Forest + SMOTE)")
plt.show()

# ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.2f})".format(roc_auc_score(y_test, y_prob_rf)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.legend()
plt.show()

```

    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.86      0.90      0.88       938
               1       0.35      0.26      0.29       188
    
        accuracy                           0.80      1126
       macro avg       0.60      0.58      0.59      1126
    weighted avg       0.77      0.80      0.78      1126
    
    ROC AUC Score: 0.6582078210769859



    
![png](output_38_1.png)
    



    
![png](output_38_2.png)
    


# Model Comparison

Let's compare the performance of the models trained so far:

| Model                     | Accuracy | Precision | Recall  | F1 Score | ROC AUC |
|---------------------------|----------|-----------|---------|----------|---------|
| Initial Logistic Regression | 0.682    | 0.293     | 0.638   | 0.401    | 0.709   |
| Best Logistic Regression  | 0.645    | 0.273     | 0.676   | 0.388    | 0.714   |
| Random Forest + SMOTE     | 0.801    | 0.352     | 0.255   | 0.296    | 0.658   |

Based on these metrics, the **Best Logistic Regression Model** achieved the highest recall (0.676) and ROC AUC (0.714), which are important for identifying positive cases of heart disease. Although the Random Forest model has higher accuracy, its recall is significantly lower, making it less suitable for this task where minimizing false negatives is crucial.


```python

```

# Summary and Key Insights

This notebook aimed to build a Logistic Regression model to predict the 10-year risk of Coronary Heart Disease (CHD). Here's a summary of the key insights and results:

## Data Analysis Insights:

*   **Feature Relationships:** The correlation heatmap revealed expected positive correlations between age and blood pressure, smoking habits, and between blood pressure/BMI. Glucose and diabetes also showed a positive correlation.
*   **Predictive Features:** Age, systolic and diastolic blood pressure, and glucose levels appear to be important predictors of CHD.
*   **Gender and Risk:** The negative correlation with 'is_male' suggests that, in this dataset, females might have a higher risk after accounting for other factors.
*   **Class Imbalance:** The dataset exhibits a significant class imbalance, with many more instances of no CHD than CHD.

## Model Performance and Comparison:

To address the class imbalance, SMOTE was applied to the training data. We trained and evaluated three models:

*   **Initial Logistic Regression (with SMOTE and class weight):**
    *   Accuracy: 0.682
    *   Precision: 0.293
    *   Recall: 0.638
    *   F1 Score: 0.401
    *   ROC AUC: 0.709
*   **Best Logistic Regression (with SMOTE and hyperparameter tuning):** This model was optimized for recall using GridSearchCV.
    *   Accuracy: 0.645
    *   Precision: 0.273
    *   Recall: 0.676
    *   F1 Score: 0.388
    *   ROC AUC: 0.714
*   **Random Forest + SMOTE:**
    *   Accuracy: 0.801
    *   Precision: 0.352
    *   Recall: 0.255
    *   F1 Score: 0.296
    *   ROC AUC: 0.658

**Key Finding:** The **Best Logistic Regression Model** achieved the highest recall (0.676) and a good ROC AUC (0.714). While the Random Forest had higher accuracy, its significantly lower recall (0.255) makes it less suitable for this problem where identifying positive cases (high recall) is crucial, even if it means accepting more false positives. The best Logistic Regression model provides a better balance for this specific objective.

## Future Improvements:

*   Further explore the precision-recall trade-off by adjusting the classification threshold or using cost-sensitive learning.
*   Experiment with other classification algorithms (e.g., Gradient Boosting) that might handle the data and imbalance effectively.
*   Continue with feature engineering to potentially improve model performance.
*   Consider collecting more data, particularly for the minority class.
