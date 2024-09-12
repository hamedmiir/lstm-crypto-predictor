import requests
import pandas as pd
from datetime import datetime

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
