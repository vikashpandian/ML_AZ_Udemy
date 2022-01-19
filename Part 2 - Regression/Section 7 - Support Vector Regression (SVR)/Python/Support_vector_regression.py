import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

#y = y.reshape(len(y), 1)

# Feature Scaling
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
#print(x)
#print(y)

#Training
regressor = SVR(kernel = 'rbf') # radial based kernel (non-linear)
regressor.fit(x, y[:, -1])

sal_pred = regressor.predict(sc_x.transform([[6.5]]))
print(sc_y.inverse_transform(sal_pred))

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'magenta')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Hi-Res Plot
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'magenta')
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(regressor.predict(x_grid)), color = 'blue')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
