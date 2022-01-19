import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#print(tf.__version__)

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
# print(x)
# print(y)

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
# print(x)

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
# print(x[:,3])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# print(x_train)
# print(x_test)

ann = tf.keras.models.Sequential()
# Adding input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # relu is rectifier activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding Output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=32, epochs=100)

# Predicting
sample_input = [[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]]
df = pd.DataFrame(sample_input)
df = df.iloc[:, :].values
df[:, 2] = le.transform(df[:, 2])
df = np.array(ct.transform(df))
df = sc.transform(df)
#print(df)
samp_pred = ann.predict(df)
print(samp_pred)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
acs = accuracy_score(y_test, y_pred)
print(acs)
