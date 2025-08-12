import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data (Ad Spend vs Sales)
X = np.array([[50000], [60000], [70000], [80000], [90000], [100000]])
y = np.array([5.1, 5.5, 6.3, 7.0, 7.4, 8.2])  # in lakhs ₹

model = LinearRegression()
model.fit(X, y)

# Predict for custom input
spend = float(input("Enter Ad Spend in ₹: "))
predicted_sales = model.predict([[spend]])
print(f"Predicted Sales: ₹{predicted_sales[0]:.2f} lakhs")

# Plot
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='black', label='Regression Line')
plt.scatter(spend, predicted_sales, color='red', label='Prediction')
plt.xlabel("Ad Spend (₹)")
plt.ylabel("Sales (₹ in lakhs)")
plt.legend()
plt.title("Linear Regression: Ad Spend vs Sales")
plt.grid(True)
plt.show()
