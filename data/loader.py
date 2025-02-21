"""
Loads TwelveData data into pandas daaframe and updates to TimescaleDB
"""
import pandas as pd
import requests
import os
import dotenv
from sqlalchemy import create_engine, text
import math
import time

class Loader:
    def __init__(self, CONNECTION: str, TWELVEDATA_API_KEY: str):
        self.conn = CONNECTION
        self.api = TWELVEDATA_API_KEY
        self.engine = create_engine(self.conn)

    def load_data(self, symbols: list[str], interval: str = "1day", ticks: int = 5000) -> pd.DataFrame:
        """
        Loads TwelveData data into pandas daaframe and updates to TimescaleDB
        """
        for symbol in symbols:
            try:
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={ticks}&apikey={self.api}"
                response = requests.get(url).json()
                df = pd.DataFrame(response["values"])
                
                # organize columns
                df["timestamp"] = pd.to_datetime(df["datetime"])
                df.set_index("timestamp", inplace=True)
                df.drop(columns=["datetime"], inplace=True)
                df = df.astype(float)
                df["return"] = df["close"].pct_change(-1)
                df["log_return"] = df["return"].apply(lambda x: math.log(1+x))
                df.fillna({"return": 0, "log_return": 0}, inplace=True)
                # update timescale db
                df.to_sql(symbol, self.engine, if_exists="replace", index=True)
                with open("data_loader.log", "a") as f:
                    f.write(f"SUCCESS: {symbol}\n")
            except Exception as e:
                with open("data_loader.log", "a") as f:
                    f.write(f"ERROR: {symbol}: {e}\n")

if __name__ == "__main__":
    dotenv.load_dotenv()
    conn = os.getenv("SQLALCHEMY_STOCKS1DAY_URI")
    api_key = os.getenv("TWELVEDATA_API_KEY")
    loader = Loader(conn, api_key)
    
    # stocks list
    STOCK_LIST_URL = "https://api.twelvedata.com/stocks?exchange={}&apikey={}"
    NYSE_STOCK_LIST_RESPONSE: dict = requests.get(STOCK_LIST_URL.format("NYSE", api_key)).json()
    NASDAQ_STOCK_LIST_RESPONSE: dict = requests.get(STOCK_LIST_URL.format("NASDAQ", api_key)).json()
    stock_symbols = set(stock["symbol"] for stock in NYSE_STOCK_LIST_RESPONSE["data"] if stock["symbol"].isalnum()) | set(stock["symbol"] for stock in NASDAQ_STOCK_LIST_RESPONSE["data"] if stock["symbol"].isalnum())
    # etf list
    ETF_LIST_URL = "https://api.twelvedata.com/etfs?exchange={}&apikey={}"
    NYSE_ETF_LIST_RESPONSE: dict = requests.get(ETF_LIST_URL.format("NYSE", api_key)).json()
    NASDAQ_ETF_LIST_RESPONSE: dict = requests.get(ETF_LIST_URL.format("NASDAQ", api_key)).json()
    etf_symbols = set(stock["symbol"] for stock in NYSE_ETF_LIST_RESPONSE["data"] if stock["symbol"].isalnum()) | set(stock["symbol"] for stock in NASDAQ_ETF_LIST_RESPONSE["data"] if stock["symbol"].isalnum())
    
    # existing tables
    engine = create_engine(conn)
    existing_tables = set()
    with engine.connect() as conn:
        existing_tables = set([table[0] for table in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))])
    symbols = list((stock_symbols | etf_symbols) - existing_tables)
    print("Num symbols to fill: ", symbols.__len__())
    
    # batch in groups of 55
    time.sleep(61)
    for i in range(0, len(symbols), 55):
        loader.load_data(symbols[i:i+55])
        time.sleep(61)