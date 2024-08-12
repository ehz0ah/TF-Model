import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

# 1. Load the trained model
model = tf.keras.models.load_model('hdb_resale_price_predictor.keras')

# 2. Load the original training data to retrieve all unique towns and flat types
training_data = pd.read_csv('Resale Flat Prices.csv')

# Extract unique towns and flat types from the training data
unique_towns = training_data['town'].unique()
unique_flat_types = training_data['flat_type'].unique()

# 3. Fit the label encoders based on the full training data
label_encoder_town = LabelEncoder()
label_encoder_flat_type = LabelEncoder()

label_encoder_town.fit(unique_towns)
label_encoder_flat_type.fit(unique_flat_types)

# 4. Collect input from the user
town = input(f"Enter the town (e.g., {', '.join(unique_towns[:3])}, etc.): ")
flat_type = input(f"Enter the flat type (e.g., {', '.join(unique_flat_types[:3])}, etc.): ")
floor_area_sqm = float(input("Enter the floor area in sqm (e.g., 68): "))
remaining_lease = input("Enter the remaining lease (e.g., '61 years 04 months'): ")

# 5. Preprocess the inputs

# Convert 'remaining_lease' to numerical format
def convert_lease_to_years(lease_str):
    years = re.search(r'(\d+) years', lease_str)
    months = re.search(r'(\d+) months', lease_str)
    
    years = int(years.group(1)) if years else 0
    months = int(months.group(1)) if months else 0
    
    return years + (months / 12)

remaining_lease_years = convert_lease_to_years(remaining_lease)

# Encode the categorical inputs
encoded_town = label_encoder_town.transform([town])[0]
encoded_flat_type = label_encoder_flat_type.transform([flat_type])[0]

# Combine features into a single numpy array
input_features = np.array([[encoded_town, floor_area_sqm, encoded_flat_type, remaining_lease_years]])

# Scale numerical features (use the same scaler as during training)
scaler = StandardScaler()
input_features[:, 1:3] = scaler.fit_transform(input_features[:, 1:3])

# 6. Make the prediction
predicted_price = model.predict(input_features)
print(f"\nPredicted Resale Price: ${predicted_price[0][0]:,.2f}")
