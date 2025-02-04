import json
#from matplotlib import pyplot as plt
import numpy as np
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.preprocessing import MinMaxScaler
from utils.trade_utils import compute_rsi, moving_average
#load env vars
load_dotenv()

API_KEY=os.getenv("API_KEY")
API_SECRET=os.getenv("API_SECRET")

def fetch_data(symbol="btcusd",limit=500):
    """
Fetch historical trade data 

Args:
    symbol (str): The trading pair .
    limit (int): Number of recent trades to fetch.
"""
    url = f"https://api.gemini.com/v1/trades/{symbol}?limit_trades={limit}"
    response =requests.get(url, headers={"X-GEMINI-APIKEY":API_KEY})
    if response.status_code==200:
        data=response.json()
        df=pd.DataFrame(data)
        df['timestamp'] =pd.to_datetime(df['timestamp'], unit='ms')
        df=df[['timestamp','price','amount']].sort_values('timestamp')
        return df
    else:
        raise Exception(f"failed to fetch data {response.status_code}")
    
def preprocess_data(df,seq_len=60):
    """
    Preprocess the data for LSTM training.

    Args:
        df (pd.DataFrame): DataFrame containing price data.
        seq_len (int): Length of input sequences.
    """
    rsi_vals=[]
    short_vals=[]
    long_vals=[]
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to numeric
    df['price'].ffill()  # Fill NaNs with forward fill
    prices =df['price'].values
    rsi_period =14
    short_period=20
    long_period=60
    
    for i in range(len(prices)):
        #get rsi vals
        rsi_vals.append(compute_rsi(prices[:i+1],period=rsi_period))
        
        #short MA 
        if i <short_period:
            short_vals.append(np.mean(prices[:i+1]))
        else:
            short_vals.append(moving_average(prices[:i+1],short_period))
            
        # long ma
        if i<long_period:
            long_vals.append(np.mean(prices[:i+1]))
        else:
            long_vals.append(moving_average(prices[:i+1],long_period))
    #columns
    scaler= MinMaxScaler()
    df['price'] = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    
    df['rsi'] =rsi_vals
    df['ma_short']=short_vals
    df['ma_long']= long_vals
    indicator_scaler= MinMaxScaler()
    df[['rsi', 'ma_short', 'ma_long']] = indicator_scaler.fit_transform(
        df[['rsi', 'ma_short', 'ma_long']]
    )
    # put it together 
    features = ['price', 'rsi', 'ma_short', 'ma_long']
    

    sequences,labels=[],[]
    for i in range(seq_len,len(df)):
        seq_data=df[features].iloc[i-seq_len:i].values #shape= (seq,4)
        sequences.append(seq_data)
        labels.append(df['price'].iloc[i])
    return np.array(sequences), np.array(labels), scaler, indicator_scaler
    
    
    
    
def save_to_json(X, y, file_path):
    #save X and y
    data = {
        "X": X.tolist(),
        "y": y.tolist()
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print("saved data")
    
def load_data(file_path, seq_len=60):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    X = np.array(data["X"])
    y = np.array(data["y"])
    
   
    price_scaler = MinMaxScaler()
    y_reshaped = y.reshape(-1, 1)
    price_scaler.fit(y_reshaped)
    
    # Create indicator scaler
    indicator_scaler = MinMaxScaler()
    # Fit it on example data (it will be refit in preprocess_data anyway)
    indicator_scaler.fit([[0], [100]])  
    
    print("loaded data")
    return X, y, price_scaler, indicator_scaler

"""def plot_predictions(y_true,y_pred, title="Model Predictions vs. Actual Price"):
    plt.figure(figsize=(10,6))
    plt.plot(y_true,label="actual price",color="blue",linewidth=2)
    plt.plot(y_pred,label="pred price",color="green",linewidth=2, linestyle="--")
    plt.title(title,fontsize=16)
    plt.xlabel("time steps", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()"""