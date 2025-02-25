import extra_streamlit_components as stx
import streamlit as st
from utils import get_ticker_data, list_tickers, get_ticker_close, get_raw_options_data, get_options_chain
import datetime

def main():
    # configs
    st.set_page_config(layout="wide")
    manager = stx.CookieManager()

    # -----------

    # sidebar
    with st.sidebar:
        # ticker selection
        selected_ticker = manager.get("ticker")
        tickers = list_tickers()
        selected_ticker = st.selectbox("Ticker", tickers, index=tickers.index(selected_ticker or "SPY"))
        manager.set("ticker", selected_ticker, key="ticker")
        
        # set trading day
        date = st.date_input("Trading Day", value=manager.get("date") or "today", max_value="today", min_value="2008-01-01")
        manager.set("date", date.strftime("%Y-%m-%d"), key="date")

        try:
            close = get_ticker_close(selected_ticker, date)
        except Exception as e:
            st.error(str(e))

        raw_options_data = get_raw_options_data(selected_ticker, date)

        expirations = raw_options_data["expiration"].unique()
        selected_expiration = st.selectbox("Expiration", expirations, index=manager.get("expiration") or expirations[0])
        manager.set("expiration", selected_expiration, key="expiration")

if __name__ == "__main__":
    main()