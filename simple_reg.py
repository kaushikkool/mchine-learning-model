# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

# fitting simple linear regression model to training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)


# predict test set results

y_predict = regressor.predict(x_test)

# plot / visualize model predictions on train set

plt.scatter(x_train,y_train,color ="red")
plt.plot(x_train,regressor.predict(x_train),color = "blue")
plt.title("Salary vs Experience (Train Set)")
plt.xlabel("Year Of Experience")
plt.ylabel("Salary")
plt.show()


# plot / visualize model predictions on test set

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
