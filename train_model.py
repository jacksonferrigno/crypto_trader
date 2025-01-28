import os
from sklearn.model_selection import train_test_split
from utils.data_utils import fetch_data, preprocess_data, save_to_json, load_data #plot_predictions
from utils.model_utils import build_lstm_model, train_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load configurations from .env
SEQ_LEN = int(os.getenv("SEQ_LEN"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
DATA_SAVE_PATH = os.getenv("DATA_SAVE_PATH")
TRAINING_DATA_JSON = os.getenv("TRAINING_DATA_JSON")


def main():
    #get the data
    if not os.path.exists(TRAINING_DATA_JSON):
        print("fetching data")
        df= fetch_data('btcusd',limit=500)
        df.to_csv(DATA_SAVE_PATH,index=False)
        print("raw data saved")
        
        print("preprocessing data....")
        X,y, scaler= preprocess_data(df,seq_len=SEQ_LEN)
        save_to_json(X,y,TRAINING_DATA_JSON)
    else:
        print("loading data from json")
        X,y,scaler= load_data(TRAINING_DATA_JSON,seq_len=SEQ_LEN)
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("building lstm model")
    model= build_lstm_model((SEQ_LEN,1))
    history= train_model(model, X_train, y_train, X_val, y_val, save_path=MODEL_SAVE_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    
    y_pred=model.predict(X_val)
    y_val_norm=scaler.inverse_transform(y_val.reshape(-1,1))
    y_pred_norm= scaler.inverse_transform(y_pred)
    
    """    print("visualizing data")
    plot_predictions(y_val_norm,y_pred_norm)"""
    print("model training complete")
    
if __name__ =="__main__":
    main()
    
     