# Stroke Prediction
### Context
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

### Attribute Information
1. id: unique identifier
2. gender: "Male", "Female" or "Other"
3. age: age of the patient
4. hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5. heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6. ever_married: "No" or "Yes"
7. work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8. Residence_type: "Rural" or "Urban"
9. avg_glucose_level: average glucose level in blood
10. bmi: body mass index
11. smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12. stroke: 1 if the patient had a stroke or 0 if not

### Methodology
We'll be pre processing the given dataset from Kaggle in order to fit the computation. My main aim for this project was to test the difference in the predcition accuracy between three methods
1. Naive Bayes 
2. Decision Tree
3. MLP (multi layer perceptron)

In stroke medical data sets, suppose there are a 1000 data points and 900 do not have stroke and a 100 have stroke, so the model is overall biased towards predicting no stroke while training. Afetr training even if the model predicts that all the 1000 data points as "No Stroke" the total accuracy will be a whooping 90%. 
In such scenarios Confusion matrix comes to a great help.  
To understand the confusion matrix let us consider a two-class classification problem with the two outcomes being “Positive” and “Negative”. Given a data point to predict, the model’s outcome will be any one of these two.
If we plot the predicted values against the ground truth (actual) values, we get a matrix with the following representative elements:

**True Positives (TP)**: These are the data points whose actual outcomes were positive and the algorithm correctly identified it as positive.

**True Negatives (TN)**: These are the data points whose actual outcomes were negative and the algorithm correctly identified it as negative.

**False Positives (FP)**: These are the data points whose actual outcomes were negative but the algorithm incorrectly identified it as positive.

**False Negatives (FN)**: These are the data points whose actual outcomes were positive but the algorithm incorrectly identified it as negative.

![Confusion matrix](https://miro.medium.com/max/546/1*h1MBLDA6bPxNpxwgSD1xNA.png)

We'll calculate which models works the best by calculating the F1 scores of all the models and comparing them.
Before F1 score we have to know **Precision** and **Recall**  
**Precison** -In simple terms, precision means what percentage of the positive predictions made were actually correct.
![precision](https://miro.medium.com/max/444/1*_cYPzG5XV7XaWBRKB-pqWQ.png)   
**Recall** -In simple terms means, what percentage of actual positive predictions were correctly classified by the classifier.
![Recall](https://miro.medium.com/max/431/1*5OA6GNFIl-_VcRbxv6sITg.png)   

Now **F1 score** is calculated as    
![F1](https://miro.medium.com/max/303/1*ZMWbXzr6y1sLxJzbtAkkDQ.png)   

Better the F1 score, better is the model.

### Link to the dataset is given below
[Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset "Stroke Predcition Dataset")   

## Importing all the required libraries 
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
```

## Reading the dataset 
```python
df=pd.read_csv(r'C:\Users\User1\OneDrive\Desktop\strokeproject\data.csv')
```
## Getting a brief look into the dataset 
```python
print(df.head())
print(df.shape)
```

## Checking for Null values in the dataset 
```python
df.isnull().sum()
```

## Filling the Null values with the mean of their respective column 
```python
df['bmi'].fillna(value=df['bmi'].mean(), inplace=True)
df.isnull().sum()
```

## Label Encoding
In order for the model to train on our dataset we need to give it numeric values instead of strings. Label Encoding gives a specific string a specific integer value
```python
df['Residence_type'].unique()
df['gender'].unique()
df['heart_disease'].unique()
df['work_type'].unique()
df['smoking_status'].unique()

label_encoder=LabelEncoder()
str_data=df.select_dtypes(include=['object'])
int_data=df.select_dtypes(include=['integer','float'])

int_data.info()
str_data.info()

features=str_data.apply(label_encoder.fit_transform)
print(features)
features=features.join(int_data)
features.head()
```

## Train Test split 
```python
training_data, testing_data = train_test_split(features, test_size=0.3, random_state=5)
```

## Data Analysis before training
```python
sns.countplot(x=df['stroke'])
plt.title("no of patients affected by stroke")
plt.show()

sns.countplot(x=df['stroke'],hue=df['gender'])

dff=df[df.stroke==1]

sns.histplot(x=dff['stroke'],hue=df['smoking_status'],stat="frequency",multiple="dodge")

corr_matrix=df.corr()
```

## Correlation matrix for all the parameters
```python
sns.heatmap(corr_matrix, annot=True)
plt.show()
```

## Naive Bayes
```python
model= GaussianNB()
y_train=training_data['stroke']
y_test=testing_data['stroke']
x_train=training_data.drop(['stroke'],axis=1)
x_test=testing_data.drop(['stroke'],axis=1)
model.fit(x_train,y_train)
predict=model.predict(x_test)
score=model.score(x_test,y_test)
score

cv_results=cross_validate(model,x_train,y_train,cv=5)
cv_results
confusion=pd.crosstab(y_test,predict)
confusion
nb_report=classification_report(y_test,predict)
print(nb_report)
```

## Decision Tree
```python
dt_mod=DecisionTreeClassifier(criterion='entropy',max_depth=8)
dt_mod.fit(x_train,y_train)
y_predict=dt_mod.predict(x_test)
confusion=pd.crosstab(y_test,y_predict)
confusion
score=dt_mod.score(x_train,y_train)
score
nb_report=classification_report(y_test,y_predict)
print(nb_report)
```

## Multi Layer Perceptron (MLP)
 ```python
 mlp_model=MLPClassifier()
 mlp_model.fit(x_train,y_train)
 mlp_predict=mlp_model.predict(x_test)
 confusion=pd.crosstab(y_test,mlp_predict)
confusion
score=mlp_model.score(x_test,y_test)
score
mlp_report=classification_report(y_test,mlp_predict)
print(mlp_report)
```

## Comparing the F1 scores of different models 
#### Out of the three models that I used, Multi Layer Perceptron which is based on Neural Network was able the get the highest F1 score

