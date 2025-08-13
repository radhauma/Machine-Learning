import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Example dataframe (from your housing dataset)
data = {
    "Size": [1.4, 1.6, 1.8, 2.0, 2.2],
    "Bedrooms": [2, 3, 3, 4, 4]
}
df = pd.DataFrame(data)

# Add constant term for intercept
X = add_constant(df)

# Calculate VIF
vif_df = pd.DataFrame()
vif_df["Variable"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_df)
