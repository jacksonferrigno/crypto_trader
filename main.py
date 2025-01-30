import os
import time
import threading
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from dotenv import load_dotenv
from utils.trade_utils import fetch_price, place_trade, log_trade, compute_rsi, moving_average
from utils.data_utils import fetch_data, preprocess_data, save_to_json, load_data
from utils.model_utils import build_lstm_model, train_model
from sklearn.model_selection import train_test_split

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
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
DATA_SAVE_PATH = os.getenv("DATA_SAVE_PATH")
TRAINING_DATA_JSON = os.getenv("TRAINING_DATA_JSON")
MAX_CONSECUTIVE_TRADES = 4

price_window=[] #keep track of price history for trader 
rsi_window=[] #for rsi and moving avg calcs
trade_tracker=0
lock =threading.Lock()


def trade(symbol="btcusd"):
    """training function to run every 1 minutes
    """
    global price_window, trade_tracker, rsi_window
    print("loading model...")
    try:    
        model = load_model(MODEL_PATH)
        _,_, price_scaler, indicator_scaler= preprocess_data(
            fetch_data(symbol,limit=500),seq_len=SEQ_LEN
        )
    except Exception as e:
        print(f"error {e}")    
        return 
    
    print("starting to trade muhaha")
    while True:
        try:
            current_price =fetch_price(symbol)
            with lock:
                price_window.append(current_price)
                rsi_window.append(current_price)
                #do we have enough for a predictions
                if len(price_window)>SEQ_LEN:
                    #set up window
                    price_window = price_window[-SEQ_LEN:] #keep curr sequence
                    
                if len(rsi_window)>2000:
                    rsi_window=rsi_window[-2000:] #longer window for rsi 
                    
            if len(price_window)==SEQ_LEN:
                
                rsi_val = compute_rsi(rsi_window,14)
                short_ma= moving_average(rsi_window,20)
                long_ma= moving_average(rsi_window,60)
                
                scaled_prices= price_scaler.transform(
                    np.array(price_window).reshape(-1,1)
                ).flatten()
                
                indicators = indicator_scaler.transform(
                    np.array([[rsi_val,short_ma,long_ma]])
                )[0]
                #prep input data with feats
                input_sequence = np.column_stack([
                    scaled_prices,
                    np.full(SEQ_LEN,indicators[0]),#rsi
                    np.full(SEQ_LEN,indicators[1]), # short ma
                    np.full(SEQ_LEN,indicators[2]), # long ma
                ])
                #reshape for model i/p 
                input_data= input_sequence[np.newaxis,:,:]
                #predict next price
                predicted_scaled = model.predict(input_data)[0][0]
                predicted_price = price_scaler.inverse_transform(
                    [[predicted_scaled]]
                )[0][0]
                #calc change
                price_change=(predicted_price-current_price)/current_price
                
            
                print(
                    f"\nCurrent: {current_price:.2f}, "
                    f"Predicted: {predicted_price:.2f}, "
                    f"Change: {price_change:.4f}, "
                    f"RSI: {rsi_val:.1f}, "
                    f"SMA(20): {short_ma:.2f}, "
                    f"SMA(60): {long_ma:.2f}"
                )
                
                should_buy=(price_change>BUY_THRESHOLD)
                should_sell=(price_change<SELL_THRESHOLD)
                
                oversold=(rsi_val<30)
                overbought=(rsi_val>70)
                
                uptrend=(short_ma>long_ma)
                downtrend=(long_ma>short_ma)
                
                bullish_sig=sum([should_buy,oversold,uptrend])
                bearish_sig=sum([should_sell,overbought,downtrend])
                
                #buying logic
                if bullish_sig>=2 and bullish_sig>bearish_sig:
                    if trade_tracker>=MAX_CONSECUTIVE_TRADES:
                        print("too many buys")
                    else:
                        print("BUY")
                        trade_result= place_trade(
                            API_KEY,API_SECRET,symbol,
                            TRADE_AMOUNT,current_price,
                            side="buy"
                        )
                        
                        log_trade(trade_result)
                        trade_tracker+=1
                elif bearish_sig>=2 and bearish_sig>= bullish_sig:
                    if trade_tracker<= - MAX_CONSECUTIVE_TRADES:
                        print("too many sells")
                    else:
                        print("SELL")
                        trade_result= place_trade(
                            API_KEY,API_SECRET,symbol,
                            TRADE_AMOUNT,current_price,
                            side="sell"
                        )
                        
                        log_trade(trade_result)
                        trade_tracker-=1
                else:
                    print("doing nothing no good indicator of either")
                    
            time.sleep(60) #every 60 seconds 
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
            
            
def train():
    """training function that runs every 10 minutes
    """
    #get the data
    while True:
        try:
            #get data
            if not os.path.exists(TRAINING_DATA_JSON):
                print("fetching data")
                df= fetch_data('btcusd',limit=500)
                df.to_csv(DATA_SAVE_PATH,index=False)
                print("raw data saved")
                
                print("preprocessing data....")
                X,y, scaler, indicator_scaler = preprocess_data(df,seq_len=SEQ_LEN)
                save_to_json(X,y,TRAINING_DATA_JSON)
            else:
                print("loading data from json")
                X,y,scaler,indicator_scaler= load_data(TRAINING_DATA_JSON,seq_len=SEQ_LEN)
            #split data for training     
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            #build it 
            print("building lstm model")
            if os.path.exists(MODEL_PATH):
                model=load_model(MODEL_PATH)
                print(f"    -retraining loaded model")
            else:
                model= build_lstm_model((SEQ_LEN,4))
            history= train_model(model, X_train, y_train, X_val, y_val, save_path=MODEL_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE)
            
            
            y_pred=model.predict(X_val)
            y_val_norm=scaler.inverse_transform(y_val.reshape(-1,1))
            y_pred_norm= scaler.inverse_transform(y_pred)
            
            """    print("visualizing data")
            plot_predictions(y_val_norm,y_pred_norm)"""
            print("model training complete")
            time.sleep(3600)
        except Exception as e:
            print(f"error in training {e}")
            time.sleep(60)
            
if __name__ == "__main__":
    print("Starting trading and training threads...", flush=True)

    trader_thread = threading.Thread(target=trade, daemon=False)
    training_thread = threading.Thread(target=train, daemon=False)

    trader_thread.start()
    training_thread.start()

    print("Threads started successfully!", flush=True)

    trader_thread.join()
    training_thread.join()
                
            