# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
# NAME: V MYTHILI(212223040123)
# DEPT: CSE

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.
   
## Program:
NAME: V MYTHILI CSE
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df=pd.read_csv("student_scores.csv")

df.head()
df.tail()

x=df.iloc[:,:-1].values
x
y=df.iloc[:,-1].values
y


from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.5,random_state=0)

from sklearn.linear_model import LinearRegression
a=LinearRegression()
a.fit(x_train,y_train)

y_pre=a.predict(x_test)
y_pre

y_test

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,a.predict(x_train),color='blue')
plt.title("Hours VS scores(training set)")
plt.xlabel("Houres")
plt.ylabel("Scores")
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pre,color='blue')
plt.title("Hours VS scores(testing set)")
plt.xlabel("Houres")
plt.ylabel("Scores")
plt.show()


mse=mean_squared_error(y_test,y_pre)
print("mean square error =", mse)
mae=mean_absolute_error(y_test,y_pre)
print("mean absolute error =",mae)
rmse=np.sqrt(mse)
print("RMSE=",rmse)
```

## Output:

HEAD:

![image](https://github.com/user-attachments/assets/4d4de72d-deae-4abe-8fbb-a62d6f2f70ec)


TAIL:


![image](https://github.com/user-attachments/assets/6e9493c9-dcb2-4055-a08c-d49dce3fd2e0)

x-values:

![image](https://github.com/user-attachments/assets/ed51f316-59dd-4176-b9ae-0485d31c3031)


y-values:


![image](https://github.com/user-attachments/assets/353cbf2c-8e3b-487e-bb9a-e026bea82c7f)

y_pre:


![image](https://github.com/user-attachments/assets/297a5729-1ed8-44c0-b022-2269487ec9c1)

y_test:


![image](https://github.com/user-attachments/assets/4e791196-9bbd-4946-bd99-20cbbdbfcab5)

plot of training dataset:


![image](https://github.com/user-attachments/assets/20d35085-72e1-449d-93c6-8e917ed52977)

plot of testing dataset:


![image](https://github.com/user-attachments/assets/ed06bb0f-5cd7-4001-a08c-e1c57b9d96d7)

sme,sam & rsme value:


![image](https://github.com/user-attachments/assets/a1c6d14f-e0e9-4f97-b598-df45f0896bad)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
