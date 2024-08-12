'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('2024_hdb_resale_price_predictor.keras')

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert remaining_lease to years
    input_data['remaining_lease'] = input_data['remaining_lease'].apply(lambda x: int(x.split()[0]) + (int(x.split()[2]) / 12) if len(x.split()) > 2 else int(x.split()[0]))
    
    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data, columns=['town', 'flat_type'])
    
    # Scale numerical features
    scaler = StandardScaler()
    input_data[['floor_area_sqm', 'remaining_lease']] = scaler.fit_transform(input_data[['floor_area_sqm', 'remaining_lease']])
    
    return input_data

# Function to make predictions
def predict_resale_price(input_data):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    
    # Get the expected number of features from the model's input shape
    expected_features = model.input_shape[1]
    
    # Ensure all columns from training are present
    missing_cols = set(range(expected_features)) - set(processed_data.columns)
    for col in missing_cols:
        processed_data[col] = 0
    
    # Reorder columns to match the training data
    processed_data = processed_data.reindex(columns=range(expected_features), fill_value=0)
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    return predictions.flatten()

# Example usage
if __name__ == "__main__":
    # Sample input data
    sample_data = pd.DataFrame({
        'floor_area_sqm': [95.0],
        'remaining_lease': ['70 years 3 months'],
        'town': ['ANG MO KIO'],
        'flat_type': ['4 ROOM']
    })
    
    # Make prediction
    predicted_price = predict_resale_price(sample_data)
    
    print(f"Predicted Resale Price: ${predicted_price[0]:,.2f}")

    # You can add more samples or create a loop to predict multiple prices
    # For example:
    more_samples = pd.DataFrame({
        'floor_area_sqm': [110.0, 67.0, 93.0],
        'remaining_lease': ['65 years', '80 years 6 months', '72 years 9 months'],
        'town': ['BISHAN', 'CLEMENTI', 'TAMPINES'],
        'flat_type': ['5 ROOM', '3 ROOM', '4 ROOM']
    })
    
    predicted_prices = predict_resale_price(more_samples)
    
    for i, price in enumerate(predicted_prices):
        print(f"Sample {i+1} Predicted Resale Price: ${price:,.2f}")
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the saved model
model = tf.keras.models.load_model('2024_hdb_resale_price_predictor.keras')

print("Model Summary:")
model.summary()

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert remaining_lease to years
    input_data['remaining_lease'] = input_data['remaining_lease'].apply(lambda x: int(x.split()[0]) + (int(x.split()[2]) / 12) if len(x.split()) > 2 else int(x.split()[0]))
    
    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data, columns=['town', 'flat_type'])
    
    # Scale numerical features
    scaler = StandardScaler()
    input_data[['floor_area_sqm', 'remaining_lease']] = scaler.fit_transform(input_data[['floor_area_sqm', 'remaining_lease']])
    
    return input_data

# Function to make predictions
# def predict_resale_price(input_data):
#     # Preprocess the input data
#     processed_data = preprocess_input(input_data)
    
#     # Get the expected number of features from the model's input shape
#     expected_features = model.input_shape[1]
    
#     # Ensure all columns from training are present
#     missing_cols = set(range(expected_features)) - set(processed_data.columns)
#     for col in missing_cols:
#         processed_data[col] = 0
    
#     # Reorder columns to match the training data
#     processed_data = processed_data.reindex(columns=range(expected_features), fill_value=0)
    
#     # Make predictions
#     predictions = model.predict(processed_data)
    
#     return predictions.flatten()

# def predict_resale_price(input_data):
#     # Preprocess the input data
#     processed_data = preprocess_input(input_data)
    
#     print("Processed input data:")
#     print(processed_data)
    
#     # Get the expected number of features from the model's input shape
#     expected_features = model.input_shape[1]
    
#     print(f"Expected number of features: {expected_features}")
#     print(f"Actual number of features: {processed_data.shape[1]}")
    
#     # Ensure all columns from training are present
#     missing_cols = set(range(expected_features)) - set(processed_data.columns)
#     for col in missing_cols:
#         processed_data[col] = 0
    
#     # Reorder columns to match the training data
#     processed_data = processed_data.reindex(columns=range(expected_features), fill_value=0)
    
#     print("Final processed input:")
#     print(processed_data)
    
#     # Make predictions
#     predictions = model.predict(processed_data)
    
#     print("Raw model output:")
#     print(predictions)
    
#     return predictions.flatten()

def predict_resale_price(input_data):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    
    print("Processed input data:")
    print(processed_data)
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    print("Raw model output:")
    print(predictions)
    
    return predictions.flatten()

# Function to get user input
def get_user_input():
    print("Please enter the following information about the HDB flat:")
    floor_area = float(input("Floor area (in square meters): "))
    remaining_lease = input("Remaining lease (e.g., '70 years 3 months' or '65 years'): ")
    town = input("Town (e.g., ANG MO KIO, BISHAN, CLEMENTI): ")
    flat_type = input("Flat type (e.g., 3 ROOM, 4 ROOM, 5 ROOM): ")
    
    return pd.DataFrame({
        'floor_area_sqm': [floor_area],
        'remaining_lease': [remaining_lease],
        'town': [town.upper()],
        'flat_type': [flat_type.upper()]
    })

# Main function
if __name__ == "__main__":
    while True:
        user_input = get_user_input()
        predicted_price = predict_resale_price(user_input)
        print(f"\nPredicted Resale Price: ${predicted_price[0]:,.2f}")
        
        another = input("\nWould you like to predict another price? (yes/no): ")
        if another.lower() != 'yes':
            break

    print("\nThank you for using the HDB Resale Price Predictor!")

# Function to evaluate model accuracy
def evaluate_model_accuracy(X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Accuracy Metrics:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R-squared Score: {r2:.4f}")

# To evaluate accuracy, you need your test data
# Uncomment and run this section if you have access to your test data
"""
# Load your test data
test_data = pd.read_csv('your_test_data.csv')  # Replace with your test data file
X_test = test_data.drop(columns=['resale_price'])  # Adjust column names as needed
y_test = test_data['resale_price']

# Preprocess test data
X_test_processed = preprocess_input(X_test)

# Evaluate model accuracy
evaluate_model_accuracy(X_test_processed, y_test)
"""
