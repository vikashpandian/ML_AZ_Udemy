import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor 

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

sal_pred = regressor.predict([[6.5]])
print(sal_pred)

x_grid = np.arange(1, 10, 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'magenta')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
