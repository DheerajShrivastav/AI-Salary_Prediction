# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:52:51 2024

@author: dheer
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
# Load the dataset
file_path = './Males.csv'  # Update with your actual file path
data = pd.read_csv(file_path)

# Handle categorical variables by encoding them
data_encoded = data.copy()
categorical_cols = ['union', 'ethn', 'maried', 'health', 'industry', 'occupation', 'residence']

# Apply Label Encoding to each categorical feature
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    # Fill missing values in 'residence' column for label encoding
    data_encoded[col] = data_encoded[col].fillna('Unknown')
    data_encoded[col] = label_encoders[col].fit_transform(data_encoded[col])

# Define features (X) and target (y)
X = data_encoded[['school', 'exper', 'union', 'ethn', 'maried', 'health', 'industry', 'occupation', 'residence']]
y = data_encoded['wage']
#data1=X
#data1.head()
uni = data['residence'].unique()
print(uni)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print("Random Forest:")
print(f"\tMSE: {rf_mse}")
print(f"\tR2 Score: {rf_r2}")

# Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)
print("Decision Tree:")
print(f"\tMSE: {dt_mse}")
print(f"\tR2 Score: {dt_r2}")

# K-Nearest Neighbors Model
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_r2 = r2_score(y_test, knn_predictions)
print("K-Nearest Neighbors:")
print(f"\tMSE: {knn_mse}")
print(f"\tR2 Score: {knn_r2}")



# Assuming rf_model is your trained model (use the best performing one)
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

