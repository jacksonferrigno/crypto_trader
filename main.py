import os
import time
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from dotenv import load_dotenv
from utils.trade_utils import fetch_price, place_trade, log_trade

# Load environment variables
load_dotenv()

# Configurations
SEQ_LEN = int(os.getenv("SEQ_LEN"))
MODEL_PATH = os.getenv("MODEL_SAVE_PATH")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", 0.0125))  # Min price increase to buy
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", -0.025))  # Min price decrease to sell
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", 1.25))  # Default trade amount
MAX_CONSECUTIVE_TRADES = 4
def main(symbol="btcusd"):
    print("loading model")
    try:    
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"error {e}")    
        return 
    price_window=[]
    trade_tracker =0 # + =more buys, -=more sells
    
    print("starting to trade muhaha")
    while True:
        try:
            current_price =fetch_price(symbol)
            price_window.append(current_price)
            
            #do we have enough for a predictions
            if len(price_window)>SEQ_LEN:
                #set up window
                price_window=price_window[-SEQ_LEN:] #keep curr sequence
                scaled_window= np.array(price_window)/max(price_window)
                input_data= scaled_window[np.newaxis,:,np.newaxis] # Shape: (1, SEQ_LEN, 1)
                
                #predict next price
                predicted_price= model.predict(input_data)[0][0]*max(price_window)
            
                
                #calc change
                price_change=(predicted_price-current_price)/current_price
                print(f"Current: {current_price}, Predicted: {predicted_price}, Change: {price_change:.4f}")
                
                #make trading decision 
                if price_change>BUY_THRESHOLD:
                    if trade_tracker>=MAX_CONSECUTIVE_TRADES:
                        print("holding, too many buys")
                    else:
                        print(f"placingt buy at {current_price}")
                        trade = place_trade(API_KEY, API_SECRET, symbol, TRADE_AMOUNT, current_price, side="buy")
                        log_trade(trade)
                        trade_tracker+=1
                elif price_change<=SELL_THRESHOLD:
                    if trade_tracker<=-MAX_CONSECUTIVE_TRADES:
                        print(f"too many sells, waiting until buy to sell again")
                    else:
                        print(f"placing sell order at {current_price}")
                        trade=place_trade(API_KEY, API_SECRET, symbol, TRADE_AMOUNT, current_price, side="sell")
                        log_trade(trade)
                        trade_tracker-=1
                else:
                    print("no indication to buy or sell")
                    
            time.sleep(60) #every 60 seconds 
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
            
            
if __name__=="__main__":
    main()
                
            