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
    data = pd.read_csv(csv_file, index_col='datetime')

    # Determine num_features, excluding the datetime column
    num_features = data.shape[1] - 1  # Assuming 'datetime' is not included as a feature

    # Preprocess data (e.g., normalization)

    scaled_data = scaler.fit_transform(data.values)
    close_idx = data.columns.get_loc(predict_column_name)

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, num_features, close_idx)

    return X, y, sequence_length, num_features, scaled_data 
def create_sequences(data, sequence_length, num_features, close_idx):


    X, y = [], []
    for i in range(len(data) - sequence_length):
        #print(data.iloc[i:(i + sequence_length), :num_features]) <- to debug X, before normalizaion
        X.append(data[i:(i + sequence_length), :num_features])
        #print(data.iloc[i + sequence_length, close_idx]) <- to debug Y, before normalizaion
        y.append(data[i + sequence_length, close_idx])  #close_idx is value to be prediceted
    return np.array(X), np.array(y)

def create_predictive_model(sequence_length, num_features):
    # Input Layer
    input_layer = Input(shape=(sequence_length, num_features))

    # CNN Layers (Without MaxPooling)
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    cnn_out = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_out)

    # Bi-LSTM Layer (Matching sequence length with CNN)
    bi_lstm_out = Bidirectional(LSTM(50, return_sequences=True))(input_layer)

    # Attention Layer
    attn_layer = Attention(use_scale=True)
    attn_out = attn_layer([bi_lstm_out, bi_lstm_out])

    # Concatenating CNN and Bi-LSTM outputs (Should have matching shapes now)
    concatenated = Concatenate()([cnn_out, attn_out])

    # Proceeding as before
    concatenated = LayerNormalization()(concatenated)
    dense_out = Dense(64, activation='relu')(concatenated)
    dense_out = Dropout(0.5)(dense_out)
    output_layer = Dense(1, activation='linear')(dense_out)  # Predicting the next value

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

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






