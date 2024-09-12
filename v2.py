import requests
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

def fetch_crypto_data(coin_id, days=365):
    """
    Fetch historical data for a cryptocurrency from CoinGecko.
    :param coin_id: Cryptocurrency ID (e.g., 'bitcoin', 'ethereum').
    :param days: Number of days of historical data.
    :return: Pandas DataFrame with timestamp and price columns.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}'
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data['prices'], columns=['Timestamp', 'Price'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    
    return df

# Example usage
coins = ['bitcoin', 'ethereum', 'litecoin']
crypto_data = {coin: fetch_crypto_data(coin) for coin in coins}
print(crypto_data['bitcoin'].head())


def add_technical_indicators(df):
    """
    Add technical indicators like moving averages to the DataFrame.
    :param df: Pandas DataFrame with 'Price' column.
    :return: DataFrame with additional features.
    """
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_30'] = df['Price'].rolling(window=30).mean()
    df['Volatility'] = df['Price'].rolling(window=30).std()
    
    # Drop rows with NaN values (resulting from rolling windows)
    df.dropna(inplace=True)
    return df

# Apply to the fetched data
crypto_data['bitcoin'] = add_technical_indicators(crypto_data['bitcoin'])
print(crypto_data['bitcoin'].head())



def preprocess_data(df, feature_columns, time_step=60):
    """
    Preprocess the data for LSTM input.
    :param df: DataFrame with features and target columns.
    :param feature_columns: List of columns to use as features.
    :param time_step: Number of time steps for LSTM.
    :return: Scaled and reshaped data for training/testing.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_columns].values)
    
    # Split into train/test
    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    # Create dataset for LSTM
    def create_dataset(data):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :-1])
            Y.append(data[i + time_step, -1])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    return X_train, y_train, X_test, y_test, scaler


def build_lstm_model(input_shape):
    """
    Build and compile LSTM model.
    :param input_shape: Shape of the input data.
    :return: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_gru_model(input_shape):
    """
    Build and compile GRU model.
    """
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_cnn_lstm_model(input_shape):
    """
    Build and compile CNN-LSTM hybrid model.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
