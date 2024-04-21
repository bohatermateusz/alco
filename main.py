import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
    
    return X, y, sequence_length, num_features
def create_sequences(data, sequence_length, num_features, close_idx):
    
     
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :num_features])
        y.append(data[i + sequence_length, close_idx])  # Assuming 'close' is the last column
    return np.array(X), np.array(y)


X, y, sequence_length, num_features = load_and_preprocess_data(r'C:\Users\bohat\source\repos\AIcrypto2\Data\coin.csv')
