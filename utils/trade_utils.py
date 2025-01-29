import numpy as np
import requests
import time
import json
import hmac
import hashlib
import base64

BASE_URL = "https://api.gemini.com"

def fetch_price(symbol="btcusd"):
    response=requests.get(f"{BASE_URL}/v1/pubticker/{symbol}")
    if response.status_code==200:
        return float(response.json()["last"])
    raise Exception(f"failed to fetch price{response.status_code}")

def place_trade(api_key, api_secret, symbol, amount, price, side="buy"):
    # Format amount and price
    amount = amount / price  # Amount in BTC
    amount = f"{amount:.8f}"
    price = f"{price:.2f}"    
    payload={
        "request": "/v1/order/new",
        "nonce": str(int(time.time() * 1000)),
        "symbol":symbol,
        "amount":str(amount),
        "price":str(price),
        "side":side,
        "type": "exchange limit",
        "account": "primary"
    }
    encoded_payload = base64.b64encode(json.dumps(payload).encode())
    signature = hmac.new(api_secret.encode(), encoded_payload, hashlib.sha384).hexdigest()
    headers = {
        "Content-Type": "text/plain",
        "X-GEMINI-APIKEY": api_key,
        "X-GEMINI-PAYLOAD": encoded_payload,
        "X-GEMINI-SIGNATURE": signature
    }
    response = requests.post(f"{BASE_URL}/v1/order/new", headers=headers)
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to place trade: {response.status_code} {response.text}")

def log_trade(trade_data, file_path="trade_log.json"):
    with open(file_path,"a") as f:
        f.write(json.dumps(trade_data)+"\n")
    print(f"trade logged {trade_data}")
    
    
def compute_rsi(prices, period=14):
    """computes rsi for the last period
    from 0-100 
    rsi <30 oversold (buy area)
    rsi> overbought (sell area)
    """
    if len(prices)<period:
        #not enough info
        return 50.0
    gains=0.0
    losses=0/0
    for i in range(1, period):
        diff= prices[-i]-prices[-i-1]
        if diff>=0:
            gains+=diff
        else:
            losses-=diff
            
    if losses==0:
        return 100 #prevent div by 0 error -> bullish
    
    
    avg_gain=gains/period
    avg_loss=losses/period
    rs=avg_gain/avg_loss
    rsi =100-(100/(1+rs))
    
    return rsi

def moving_average(values,window):
    if len(values)<window:
        return np.mean(values)
    return np.mean(values[-window:])