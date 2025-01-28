from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50,return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1) #pred next price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, save_path='lstm_model.keras', epochs=10, batch_size=32):
    checkpoint= ModelCheckpoint(save_path,monitor='val_loss',save_best_only=True,verbose=1)
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size,callbacks=[checkpoint])
    return history