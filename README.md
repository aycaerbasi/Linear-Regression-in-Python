# Linear-Regression-in-Python
Linear Regression in Python: Predict Revenue of eCommerce

#import the data
from google.colab import files
uploaded = files.upload()

import pandas as pd 
customer = pd.read_csv('Ecommerce Customers.csv') 
customer.head(10)

customer.info()

customer.describe()

#visualize the data
%matplotlib inline
import matplotlib.pyplot as plt
customer.hist(bins=30, figsize=(15,10), color = "#A12BB7" )
plt.title("Attribute Histogram Plots")

#correlation
corr_matrix = customer.corr()
corr_matrix["Yearly Amount Spent"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["Yearly Amount Spent", "Length of Membership", "Time on App",
              "Avg. Session Length" , "Time on Website"]
scatter_matrix(customer[attributes], figsize=(15, 10), color='#840E6B', hist_kwds={'color':['#A029FA']})

plt.title("scatter_matrix_plot")

customer.plot(kind="scatter", x="Time on App", y="Time on Website",
    s=customer["Length of Membership"]*10, label="Length of Membership", figsize=(10,7),
    c="Yearly Amount Spent", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()


#Linear regression
X = customer.drop(['Email','Address','Avatar'], axis=1)

y = customer['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X.shape, y.shape

from sklearn  import linear_model 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

model = linear_model.LinearRegression()

#Train the model
model.fit(X, y)

y_hat = model.predict(X_test)

#model performance
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Coefficient of determination (R^2): %.3f' % r2_score(y_test, y_hat))
print('Mean squared error (MSE): %.3f'% mean_squared_error(y_test, y_hat))
print('Root mean squared error (RMSE) : %.3f'% sqrt(mean_squared_error(y_test, y_hat)) )

#visuliaze prediction and actual
plt.scatter(y_test, y_hat,  alpha=0.5, color="purple")

#Random forest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train, y_train)

print('Random Forest R squared": %.3f' % forest_reg.score(X_test, y_test))

import numpy as np
y_hat = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, y_hat)
forest_rmse = np.sqrt(forest_mse)
print('Random Forest RMSE: %.3f' % forest_rmse)



