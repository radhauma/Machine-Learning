import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
df_simple = pd.DataFrame({
    'Hours': [2, 4, 6, 8],
    'Score': [81, 93, 91, 97]
})

# Features and target
X_simple = df_simple[['Hours']]
y_simple = df_simple['Score']

# Model
model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

# Coefficients
b0 = model_simple.intercept_
b1 = model_simple.coef_[0]

# Prediction
y_pred_simple = model_simple.predict(X_simple)

b0, b1, y_pred_simple
