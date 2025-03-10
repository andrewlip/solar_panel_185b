import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('cooling_data_complete.csv')

# Define features and target
X = data[['Tamb', 'I', 'Tcoolant_in']]
y = data['mass_flowrate_optimal']

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Option 1: Scale the target variable (if values vary widely)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Define the neural network model with an explicit Input layer.
inputs = keras.Input(shape=(X_train_scaled.shape[1],))
# Hidden layer with 9 neurons and L2 regularization
x = keras.layers.Dense(9, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
x = keras.layers.Dropout(0.1)(x)
# Use a linear activation for the output layer.
outputs = keras.layers.Dense(1, activation='linear')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=200,
    batch_size=16,
    verbose=1,
    callbacks=[early_stopping]
)

# Make predictions on the scaled target
y_pred_scaled = model.predict(X_test_scaled).flatten()
# Invert the scaling to get predictions in the original target space
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
# Optionally, clip negatives if your target should be non-negative:
y_pred = np.maximum(y_pred, 0)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"Test R² Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual mass_flowrate_optimal')
plt.ylabel('Predicted mass_flowrate_optimal')
plt.title('Neural Network: Actual vs Predicted Flowrate')
plt.grid(True)
plt.show()

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
