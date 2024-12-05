## My Project

I applied machine learning techniques to investigate... Below is my report.

***

## Introduction 

Here is a summary description of the topic. Here is the problem. This is why the problem is important.

There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.

We did this to solve the problem. We concluded that...

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!
Here is a sample of the Titanic Dataset:
Columns
Target column
Object: to find a method to determine who survive
![image](https://github.com/user-attachments/assets/5f5eb7c3-dfd1-410b-b8a4-2fb7ff70dc95)

Feature Engineeing:
Declare your Categorical and Numerical Features
Cabin was transformed to floor, by mapping the floor letter to a number
All Categorical features in 1-hot encoded columns
![image](https://github.com/user-attachments/assets/2518b284-b0c8-4c87-b207-a7a8ec0c3322)


Correlation Matrix:
![image](https://github.com/user-attachments/assets/2782b995-6786-4e3a-9af3-e5f398fbf5e9)
focus the values of the Survived columns


## Modelling

You conducting comparisons along different models
1.Logistic Regression
  Describe the model
  add code
  ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#create confusion matrix and display in graphic with numbers
from sklearn.metrics import confusion_matrix


model = LogisticRegression(max_iter=20000)
model.fit(X_Train,y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("LR Accuracy:", accuracy)
```
Descibe the Accuracy score of the model
Confusion Matrix:
![image](https://github.com/user-attachments/assets/6975d86a-05d2-4b9b-aa42-510ee08bc38a)
correct score for each class
errors for each class

2.Decision Trees
Describe the model
  add code
  ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#create confusion matrix and display in graphic with numbers
from sklearn.metrics import confusion_matrix


model = LogisticRegression(max_iter=20000)
model.fit(X_Train,y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("LR Accuracy:", accuracy)
```
Descibe the Accuracy score of the model
Confusion Matrix:
![image](https://github.com/user-attachments/assets/6975d86a-05d2-4b9b-aa42-510ee08bc38a)
correct score for each class
errors for each class
3.Random Forest
Describe the model
  add code
  ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#create confusion matrix and display in graphic with numbers
from sklearn.metrics import confusion_matrix


model = LogisticRegression(max_iter=20000)
model.fit(X_Train,y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("LR Accuracy:", accuracy)
```
Descibe the Accuracy score of the model
Best fit Tree:
![image](https://github.com/user-attachments/assets/e9ecbbb9-4e34-407d-9d19-54b7b65b06f9)
Descibe the tree
Confusion Matrix:
![image](https://github.com/user-attachments/assets/6975d86a-05d2-4b9b-aa42-510ee08bc38a)
correct score for each class
errors for each class
4.Support Vector Machines
Describe the model
  add code
  ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#create confusion matrix and display in graphic with numbers
from sklearn.metrics import confusion_matrix


model = LogisticRegression(max_iter=20000)
model.fit(X_Train,y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("LR Accuracy:", accuracy)
```
Descibe the Accuracy score of the model
Confusion Matrix:
![image](https://github.com/user-attachments/assets/6975d86a-05d2-4b9b-aa42-510ee08bc38a)
correct score for each class
errors for each class
5.Neural Networks (MLP)
Describe the model
  add code
  ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#create confusion matrix and display in graphic with numbers
from sklearn.metrics import confusion_matrix


model = LogisticRegression(max_iter=20000)
model.fit(X_Train,y_Train)
y_pred = model.predict(X_Test)
accuracy = accuracy_score(y_Test, y_pred)
print("LR Accuracy:", accuracy)
```
Descibe the Accuracy score of the model
Confusion Matrix:
![image](https://github.com/user-attachments/assets/6975d86a-05d2-4b9b-aa42-510ee08bc38a)
correct score for each class
errors for each class

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Compare the results of the models
Which ones did better or worst
compare similiar scores
ROC Curve
![image](https://github.com/user-attachments/assets/c2ef2e85-14ec-4fa0-b126-48c9cbfc8b21)
start from the top left corner, the line closet to the corner is the best model with recall and precision


## Discussion

Discuss the results of the different classifiers
which was the best
why some had similar scores
etc.

## Conclusion

Describe the method that best predicts the survival of passengers
define the most important columns
and/or
pick the best model at predicting

Here is how this work could be developed further in a future project.

## References
[1] Titanic - Machine Learning from Diaster (https://www.kaggle.com/c/titanic/data)

[back](./)
