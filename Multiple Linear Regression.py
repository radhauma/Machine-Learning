# Features and target
X_multi = df[['Size (X1)', 'Bedrooms (X2)']]
y_multi = df['Price (Y)']

# Model
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# Coefficients
b0_multi = model_multi.intercept_
b1_multi, b2_multi = model_multi.coef_

# Predictions
y_pred_multi = model_multi.predict(X_multi)

b0_multi, b1_multi, b2_multi, y_pred_multi
