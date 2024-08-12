import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from keras import layers
import re

# 1. Load the Data
df = pd.read_csv('2024 Resale Prices.csv')

# 2. Preprocess the Data
# Remove rows with missing values
df = df.dropna()

# Convert 'remaining_lease' to numerical format
def convert_lease_to_years(lease_str):
    years = re.search(r'(\d+) years', lease_str)
    months = re.search(r'(\d+) months', lease_str)
    
    years = int(years.group(1)) if years else 0
    months = int(months.group(1)) if months else 0
    
    return years + (months / 12)

df['remaining_lease'] = df['remaining_lease'].apply(convert_lease_to_years)

# Encode categorical variables using one-hot encoding for better model performance
df = pd.get_dummies(df, columns=['town', 'flat_type'], drop_first=True)

# Separate features and target
X = df.drop(columns=['resale_price', 'lease_commence_date','month','block','flat_model','storey_range','street_name'])
y = df['resale_price']

# Scale numerical features for better optimization and convergence
scaler = StandardScaler()
X[['floor_area_sqm', 'remaining_lease']] = scaler.fit_transform(X[['floor_area_sqm', 'remaining_lease']])

# Split the data into training and testing sets, ensuring a stratified split for balanced training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the Model
# Use a slightly deeper model with dropout to avoid overfitting and ensure generalization
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),  # Dropout layer to prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model with Adam optimizer, mean squared error loss, and mean absolute error as a metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# View the model summary to understand its structure
model.summary()

# 4. Train the Model
# Use early stopping to prevent overfitting and save the best model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with the training data
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# 5. Evaluate the Model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')

# 6. Make Predictions
y_pred = model.predict(X_test)
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(comparison_df.head())

# 7. Save the Model
model.save('2024_hdb_resale_price_predictor.keras')
