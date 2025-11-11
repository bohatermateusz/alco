import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Data configuration
sequence_length = 4
predict_column_name = "close"

# Training hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
LSTM_UNITS = 50
DENSE_UNITS = 25
EPOCHS = 100
BATCH_SIZE = 1
EARLY_STOPPING_PATIENCE = 10

scaler = MinMaxScaler(feature_range=(0, 1))

def load_and_preprocess_data(csv_file):
    # Load the data
    data = pd.read_csv(csv_file)

    # Determine num_features, excluding the datetime column
    num_features = data.shape[1]  # Assuming 'datetime' is not included as a feature

    # Preprocess data (e.g., normalization)
    scaled_data = scaler.fit_transform(data.values)
    close_idx = data.columns.get_loc(predict_column_name)

    print(f"\nData preprocessing:")
    print(f"  Sequence length: {sequence_length} days")
    print(f"  Number of features: {num_features}")
    print(f"  Scaled data shape: {scaled_data.shape}")
    print(f"  Column to predict (close): index {close_idx}")

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, num_features, close_idx)
    print(f"\nSequence creation:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    return X, y, sequence_length, num_features, scaled_data, close_idx

def create_sequences(data, sequence_length, num_features, close_idx):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :num_features])
        y.append(data[i + sequence_length, close_idx])  # close_idx is value to be predicted
    return np.array(X), np.array(y)

X, y, sequence_length, num_features, scaled_data, close_idx = load_and_preprocess_data(r'.\Data\coin.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Create model

model = Sequential([
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), input_shape=(sequence_length, num_features)),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dense(DENSE_UNITS, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predicting and reverse scaling
print(f"\nMaking prediction:")
last_sequence = X[-1].reshape(1, sequence_length, num_features)
predicted_price = model.predict(last_sequence).flatten()
print(f"  Predicted price (scaled): {predicted_price[0]:.6f}")

# Inverse transform to get actual price
predicted_price_reshaped = predicted_price.reshape(-1, 1)
dummy_features = np.zeros((1, num_features))
dummy_features[0, close_idx] = predicted_price_reshaped

predicted_price_unscaled = scaler.inverse_transform(dummy_features)
predicted_close_price = predicted_price_unscaled[0, close_idx]

print(f"\nFinal prediction:")
print(f"  Predicted Close Price: ${predicted_close_price:.2f}")
