import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Bidirectional, Dense, Dropout, Attention, Concatenate, LayerNormalization, Flatten
from tensorflow.keras.layers import LSTM, Dense  # Import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MultiHeadAttention
from sklearn.model_selection import train_test_split


sequence_length = 4
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
    #scaled_data = data.values
    close_idx = data.columns.get_loc(predict_column_name)

    print("Sequence_length in day length:", sequence_length)
    print("Number of features based on Excel:",  num_features)
    print("Number of features based on scaler:",  scaled_data.shape[1])
    print("Column to predict:", close_idx)
    print("Scaled data shape:", scaled_data.shape) 
    print("Last sequence shape:", scaled_data[-1].shape)
    print("Last sequence value:", scaled_data[-1])

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, num_features, close_idx)
    print("X Shape:")
    print(X.shape)
    print("Y Shape:")
    print(y.shape)
    print("X[-1] Value:")
    print(X[-1])
    print("y[-1] Value:")
    print(y[-1])

    return X, y, sequence_length, num_features, scaled_data, close_idx

def create_sequences(data, sequence_length, num_features, close_idx):


    X, y = [], []
    for i in range(len(data) - sequence_length):
        #print(data.iloc[i:(i + sequence_length), :num_features]) # <- to debug X, before normalizaion
        X.append(data[i:(i + sequence_length), :num_features])
        #print(data.iloc[i + sequence_length, close_idx]) # <- to debug Y, before normalizaion
        y.append(data[i + sequence_length, close_idx])  #close_idx is value to be prediceted
    return np.array(X), np.array(y)

def create_predictive_model(sequence_length, num_features):
    # Input Layer
    input_layer = Input(shape=(sequence_length, num_features))
    
    # CNN Layers
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    cnn_out = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_out)


    
    # Bi-LSTM Layer
    bi_lstm_out = Bidirectional(LSTM(50, return_sequences=True))(cnn_out)
    ###
    attn_out = MultiHeadAttention(num_heads=2, key_dim=50)(bi_lstm_out, bi_lstm_out)
    ###
    # Attention Layer
    attn_layer = Attention(use_scale=True)
    attn_out = attn_layer([bi_lstm_out, bi_lstm_out])
    
    # Concatenating CNN and Bi-LSTM outputs
    concatenated = Concatenate()([cnn_out, attn_out])
    
    # Layer Normalization
    concatenated = LayerNormalization()(concatenated)
    
    # Dense Layer
    dense_out = Dense(64, activation='relu')(concatenated)
    dropout = Dropout(0.5)(dense_out)

    # Flattening the output to prepare for the final Dense layer
    flattened = Flatten()(dropout)
    
    # Final Output Layer
    output_layer = Dense(1, activation='linear')(flattened)  # Predicting the next value
    
    # Creating and compiling the model
    model = Model(inputs=input_layer, outputs=output_layer)

    
    
    return model

X, y, sequence_length, num_features, scaled_data, close_idx = load_and_preprocess_data(r'.\Data\coin.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





model = create_predictive_model(sequence_length, num_features)

# model = Sequential([
# Bidirectional(LSTM(50, input_shape=(sequence_length, num_features), return_sequences=True)),
# Bidirectional(LSTM(50)),
# Dense(25),
# Dense(1)
# ])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Predicting and reverse scaling
last_sequence = X[-1].reshape(1, sequence_length, num_features)
print("NEW LAST SEQUENCE:")
print(last_sequence)
print("NEW LAST SEQUENCE SHAPE:")
print(last_sequence.shape)
predicted_price = model.predict(last_sequence).flatten()
print("NEW LAST predicted_price:")
print(predicted_price)
print("NEW LAST predicted_price SHAPE:")
print(predicted_price.shape) 

predicted_price_reshaped = predicted_price.reshape(-1, 1)
# Prepare a dummy array with the shape of the original feature set
dummy_features = np.zeros((1, num_features))
# Place the predicted value in the correct column
dummy_features[0, close_idx] = predicted_price_reshaped

print("NEW LAST dummy_array:")
print(dummy_features)
print("NEW LAST dummy_array SHAPE:")
print(dummy_features.shape)

predicted_price_unscaled = scaler.inverse_transform(dummy_features)

#predicted_price_unscaled = scaler.inverse_transform(dummy_features)[0, -1]
print("NEW LAST predicted_price_unscaled:")
print(predicted_price_unscaled)

# Extracting the specific predicted close price using the close_idx
predicted_close_price = predicted_price_unscaled[0, close_idx]
print("Predicted Close Price (Unscaled), final prediction:", predicted_close_price)

# Print the predicted close price
print("Metadata (Unscaled):", predicted_price_unscaled[0, 0])
print("Metadata (Unscaled):", predicted_price_unscaled[0, 1])
print("Metadata (Unscaled):", predicted_price_unscaled[0, 2])
print("Metadata (Unscaled):", predicted_price_unscaled[0, 3])
print("Metadata (Unscaled):", predicted_price_unscaled[0, 4])
