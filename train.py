import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load data
data = pd.read_csv('data/data.csv')

X = data[['size']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/model.pkl')

print("Model trained and saved!")