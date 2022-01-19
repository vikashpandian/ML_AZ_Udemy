import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
#print(x)
#print(y)

regressor = LinearRegression()
regressor.fit(x, y)
y_pred = regressor.predict(x)

plt.scatter(x, y, color = 'magenta')
plt.plot(x, y_pred, color = 'blue')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

y_pred_poly = lin_reg.predict(x_poly)

plt.scatter(x, y, color = 'magenta')
plt.plot(x, y_pred_poly, color = 'blue')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

prevSal = lin_reg.predict(poly_reg.transform([[6.5]]))
print(prevSal)
