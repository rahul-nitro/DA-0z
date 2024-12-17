import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import  load_iris

np.random.seed(42)


X = np.random.rand(100,1)*10
y = 2*X.squeeze()+np.random.randn(100)*2

plt.figure(figsize=(8,6))
plt.scatter(X,y)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

correlation_coefficient = np.corrcoef(X.squeeze(), y)[0, 1]
print(f"Correlation Coefficient: {correlation_coefficient}")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
mes = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"mean squared error: {mes}")
print(f"R squre:{r2}")

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue')
plt.title('linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
