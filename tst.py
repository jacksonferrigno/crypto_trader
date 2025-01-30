import os
import pandas as pd
from utils.data_utils import fetch_data, preprocess_data, save_to_json

# Create a separate folder for temporary data
SAVE_DIR = "temp_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# File paths (ensuring no interference with main)
RAW_DATA_PATH = os.path.join(SAVE_DIR, "raw_data.csv")
TRAINING_DATA_JSON = os.path.join(SAVE_DIR, "training_data.json")

# Configuration
SEQ_LEN = 60  # Default sequence length for preprocessing

def main(symbol="btcusd"):
    try:
        print("Fetching market data...")
        df = fetch_data(symbol, limit=500)  # Fetch latest data
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Raw data saved at {RAW_DATA_PATH}")

        print("Preprocessing data...")
        X, y, scaler, indicator_scaler = preprocess_data(df, seq_len=SEQ_LEN)
        save_to_json(X, y, TRAINING_DATA_JSON)
        print(f"Preprocessed data saved at {TRAINING_DATA_JSON}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
