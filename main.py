import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("\filepath")

x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values


lr = LinearRegression
poly = PolynomialFeatures
x_poly = poly.fit_transform(x) or poly.fit_transform([[6.5]])
lr.fit(x_poly,y)
#in this program we are destined to find the salary of level 6.5 please change it to your convinience.
lr.predict([[6.5]])

#if you want to plot it

plt.scatter(x,y,color='red')
plt.plot(x,lr.predict(x_poly))
plt.show()

