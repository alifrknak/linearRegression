import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# this code predicts temperature based on humidity, pressure, and wind speed using a linear regression model.

df = pd.read_csv('weather.csv') 

X = df[['Humidity', 'Pressure', 'WindSpeed']]
y = df['Temperature']

# algecraic equation: y = mx + b
# where y is the dependent variable (Temperature),
# x is the independent variable (Humidity, Pressure, WindSpeed),
# m is the slope (coefficients of the independent variables), and b is bias.

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# mean squared error : lower MSE means predictions are, on average, closer to actual values. 
# Higher R2 score indicates a better fit of the model to the data
# 1.0 = perfect fit 

# r2 = 1- (Σ(actual - predicted)² / Σ(actual - mean(actual))²)
# MSE = 1/n * Σ(actual - predicted)²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")


# Predict with custom data
sample = pd.DataFrame([[60, 1012, 5]], columns=['Humidity', 'Pressure', 'WindSpeed'])
prediction = model.predict(sample)
print(f"Predicted Temperature: {prediction[0]:.2f}")

# output 
# MSE: 0.01
# R2 Score: 0.99
# Predicted Temperature: 25.30
