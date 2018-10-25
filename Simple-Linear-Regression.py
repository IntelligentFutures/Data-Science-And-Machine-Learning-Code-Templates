# Simple Linear Regression - Python 3.5

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')  #Import Your Own Data Set
X = dataset.iloc[:, :-1].values    # Change According To Dataset
Y = dataset.iloc[:, 1].values      # Change According To Dataset

# Making a Training set and Test set from the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Making a Graph for the Training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title (Training set)')
plt.xlabel('Title - Independant Variable')
plt.ylabel('Title - Dependant Variable')
plt.show()

# Making a Graph for the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title (Test set)')
plt.xlabel('Title - Independant Variable')
plt.ylabel('Title - Dependant Variable')
plt.show()