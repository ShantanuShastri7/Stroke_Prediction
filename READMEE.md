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
