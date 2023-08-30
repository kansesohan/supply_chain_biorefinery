import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load and Prepare Data
data = pd.read_csv('/content/drive/MyDrive/supplychain/Biomass_History.csv')

features = data[['2010', '2011', '2012', '2013', '2014', '2015', '2016']].values
target = data['2017'].values

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.reshape(-1, 1))

train_features, test_features, train_target, test_target = train_test_split(features_scaled, target_scaled, test_size=0.3, random_state=42)

# Build and Train the Feedforward Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(train_features.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(train_features, train_target, epochs=100, batch_size=16, validation_data=(test_features, test_target))

# Make predictions
predictions = model.predict(test_features)

# Inverse transform predictions to get original scale
predicted_values = scaler.inverse_transform(predictions)

# Calculate R2 score
r2 = r2_score(scaler.inverse_transform(test_target), predicted_values)

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.plot(predicted_values, label='Predicted')
plt.plot(scaler.inverse_transform(test_target), label='Actual', color='black')
plt.legend()
plt.title(f"Predictions using Feedforward Neural Network\nR2 Score: {r2:.4f}")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.show()

# After training the model
model.save('/content/drive/MyDrive/supplychain/Feedforward_Neural_Network')  # Save the model

# Load the pre-trained model
model = load_model('/content/drive/MyDrive/supplychain/Feedforward_Neural_Network')  # Load the trained model here

# Load and Prepare the new Data
data = pd.read_csv('/content/drive/MyDrive/supplychain/Biomass_History_with_Predictions.csv')

# Extract original features for prediction
original_features_to_predict = data[['2010', '2011', '2012', '2013', '2014', '2015', '2016']].values

scaler = MinMaxScaler()
original_features_scaled_to_predict = scaler.fit_transform(original_features_to_predict)

# Make predictions for 2019
predictions_2019 = model.predict(original_features_scaled_to_predict)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predictions_2019, columns=['2019'])

# Concatenate the original data with the predictions
data_with_predictions = pd.concat([data, predictions_df], axis=1)

# Save the updated dataset
data_with_predictions.to_csv('/content/drive/MyDrive/supplychain/Biomass_History_with_Predictions_Updated.csv', index=False)