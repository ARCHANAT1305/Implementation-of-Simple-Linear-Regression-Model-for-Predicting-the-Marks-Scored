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
5. Predict the regression for the marks by using the representation of the graph.     
6. Compare the graphs and hence we obtained the linear regression for the given datas    

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARCHANA T
RegisterNumber:  212223240013
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)

```

## Output:

![image](https://github.com/user-attachments/assets/25590f5a-41a7-4cc3-89fc-4a1a0241719a)   

![image](https://github.com/user-attachments/assets/b817498d-6d77-4662-82a5-62f80198073b)   

![image](https://github.com/user-attachments/assets/b103ef21-a58c-4eed-a7d7-44ba7e374732)    

![image](https://github.com/user-attachments/assets/a2c7b60e-c33b-4598-a3af-136cbaddf483)   

![image](https://github.com/user-attachments/assets/e97f75b4-9f22-45f1-94f9-865e7489840d)    

![image](https://github.com/user-attachments/assets/4a26068d-bcc5-4e68-9d40-8194da27d5fb)    




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
