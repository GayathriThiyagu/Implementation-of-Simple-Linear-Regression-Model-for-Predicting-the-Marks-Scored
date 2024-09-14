# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: T. Gayathri
RegisterNumber: 212223100007

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
print()
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
*/
```

## Output:

Dataset
![Screenshot 2024-09-14 154819](https://github.com/user-attachments/assets/77cb02a6-0d21-4c16-b836-42713cf9078d)

Head and Tail
![Screenshot 2024-09-14 154846](https://github.com/user-attachments/assets/a9f319a1-c104-4c7f-aead-232905744783)

X and Y
![Screenshot 2024-09-14 154908](https://github.com/user-attachments/assets/53094fb8-61b9-4617-bfd9-2bef5e6b8de6)

Training data
![Screenshot 2024-09-14 154930](https://github.com/user-attachments/assets/608786a8-45c5-4bf4-b022-9a34d9bde8f1)

Plot for training set
![Screenshot 2024-09-14 154950](https://github.com/user-attachments/assets/5c5ae456-cdd0-4536-baec-20b0008bbfa6)

Plot for test set
![Screenshot 2024-09-14 155012](https://github.com/user-attachments/assets/af3f6b1b-7bfd-4f6d-b6d3-f09a9e5528c8)

MSE, MAE, RMSE values
![Screenshot 2024-09-14 155028](https://github.com/user-attachments/assets/807200e9-440c-4453-af3f-285d292451b9)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
