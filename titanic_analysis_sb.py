import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#Dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv') 

#Data content
print(train_data.head())
print(train_data.info())
print(train_data.describe()) 

#Distributions
sns.countplot(x='Survived', data=train_data)
plt.show()

sns.histplot(train_data['Age'].dropna(), kde=True)
plt.show()

sns.boxplot(x='Pclass', y='Fare', data=train_data)
plt.show() 


#cleaning data
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

#categoriez varieables
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

#Define Feauture & Target
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

#Separation of dateset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#standardazied the feauter to enhance the performance 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

#Training model 
model = LogisticRegression()
model.fit(X_train, y_train)

#Valutation of the model 
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

#Test Set
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

missing_cols = set(X.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[X.columns]



X_test = test_data.drop(columns=['PassengerId'])
X_test = X_test[X.columns]


X_test = scaler.transform(X_test)
test_predictions = model.predict(X_test)    

submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)