import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Bidirectional, Dense, Dropout, Attention, Concatenate, LayerNormalization
from tensorflow.keras.layers import LSTM, Dense, Bidirectional  # Import Bidirectional
from sklearn.model_selection import train_test_split

sequence_length = 5
predict_column_name = "close"

scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler(feature_range=(0, 1))

def load_and_preprocess_data(csv_file):
    # Load the data
    data = pd.read_csv(csv_file)

    # Determine num_features, excluding the datetime column
    num_features = data.shape[1]  # Assuming 'datetime' is not included as a feature

    # Preprocess data (e.g., normalization)

    scaled_data = scaler.fit_transform(data.values)
    close_idx = data.columns.get_loc(predict_column_name) 

    print("Sequence_length in day length:", sequence_length)
    print("Number of features based on Excel:",  num_features)
    print("Number of features based on scaler:",  scaled_data.shape[1])
    print("Column to predict:", close_idx)
    print("Total data points:", scaled_data.shape)
    print("Last sequence initial shape:", scaled_data[-sequence_length:].shape)

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, num_features, close_idx)

    return X, y, sequence_length, num_features, scaled_data 

def create_sequences(data, sequence_length, num_features, close_idx):


    X, y = [], []
    for i in range(len(data) - sequence_length):
        #print(data.iloc[i:(i + sequence_length), :num_features]) # <- to debug X, before normalizaion
        X.append(data[i:(i + sequence_length), :num_features])
        #print(data.iloc[i + sequence_length, close_idx]) # <- to debug Y, before normalizaion
        y.append(data[i + sequence_length, close_idx])  #close_idx is value to be prediceted
    return np.array(X), np.array(y)

def predict_new_data(new_data, model, scaler, sequence_length, num_features):
    scaled_data = scaler.transform(new_data.reshape(1, -1))
    X_new = create_sequences(scaled_data, sequence_length, num_features, -1)[0]  # Assuming last column is the target
    prediction = model.predict(X_new)
    return prediction


X, y, sequence_length, num_features, scaled_data = load_and_preprocess_data(r'.\Data\coin.csv')




# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
#model = create_predictive_model(sequence_length, num_features)

model = Sequential([
Bidirectional(LSTM(50, input_shape=(sequence_length, num_features), return_sequences=True)),
Bidirectional(LSTM(50)),
Dense(25),
Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test))

#adding new demesion as we have batch 1
last_value_X = X[-1]
last_value_X = last_value_X[np.newaxis, :] 

predicted_price_tomorrow = model.predict(last_value_X)

print("Predicted unormalized price:", predicted_price_tomorrow)

#original_value = scaler.inverse_transform(predicted_price_tomorrow)


# #Predict tomorrow's price using the last sequence from the dataset
# if scaled_data.shape[0] >= sequence_length:
#     last_sequence = scaled_data[-sequence_length:]  # Take the last 'SEQUENCE_LENGTH' rows
#     print(last_sequence)
#     print(last_value_X)
#     last_sequence = last_sequence.reshape((1, sequence_length, num_features))  # Reshape for the model
#     print(last_sequence)
#     predicted_price_tomorrow = model.predict(last_value_X)
#     print("Predicted price for tomorrow:", predicted_price_tomorrow[0][0])
# else:
#     print("Not enough data to create a full sequence for prediction.")


# Assuming your model predicted the scaled value of the target feature (let's say it's the last column)
predicted_scaled_price = np.array(predicted_price_tomorrow)  # Your model's output

# Create a dummy array with the same number of features
dummy_input = np.zeros((1, num_features))
# Place the predicted price in the correct column (assuming the 'close' column is the last one)
dummy_input[0, -1] = predicted_scaled_price

# Use the scaler to inverse transform the data
predicted_price = scaler.inverse_transform(dummy_input)[0, -1]

print("Predicted price for tomorrow (denormalized):", predicted_price)
