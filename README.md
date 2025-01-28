# Cryptocurrency Trading Bot

This project is a basic cryptocurrency trading bot that leverages machine learning to predict price movements and execute trades on the Gemini exchange.

## Overview

- **Data Handling**: The bot fetches historical trade data from the Gemini API and preprocesses it for model training.
- **Model Training**: Utilizes a Long Short-Term Memory (LSTM) neural network to predict future price changes.
- **Trading Execution**: Makes buy/sell decisions based on model predictions and executes trades on Gemini using API keys.

## Configuration


## Gemini Integration

- **API Usage**: The bot interacts with the Gemini exchange through its public and private APIs to fetch price data and place trades.
- **Trade Execution**: Trades are executed based on predicted price changes, with all transactions logged for review.

## Docker Hosting

- **Containerization**: The bot is containerized using Docker, ensuring a consistent and isolated environment for both training and trading operations.
- **Semaphore Flag**: A semaphore-like flag is used to prevent multiple instances of the trading bot from running simultaneously, avoiding conflicts and ensuring safe access to shared resources.

This setup provides a robust foundation for automated trading on the Gemini platform, leveraging Docker for easy deployment and management.
