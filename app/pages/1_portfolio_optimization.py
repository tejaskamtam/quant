import time
import streamlit as st
from pypfopt import EfficientFrontier
import extra_streamlit_components as stx
from utils import get_ticker_data, list_tickers, calculate_expected_return
import pandas as pd
import plotly.graph_objects as go


def main():
    # configs
    st.set_page_config(layout="wide")
    manager = stx.CookieManager()
    st.title("Portfolio Optimization")
    st.write("Expected return is calculated with the following method:")
    st.code("""
    from scipy.stats import gaussian_kde
    # get rolling annual returns at every day
    annual_returns = df["return"][::-1].rolling(window=251).apply(lambda x: np.prod(x+1)-1).dropna()
    # estimate the PDF using gaussian kernel density estimation
    kde = gaussian_kde(annual_returns)
    # get a range of 1000 discrete possible returns, bounding daily returns between 100% and -100%
    possible_returns = np.linspace(-1, 1, 1000)
    # get density estimates for each possible return
    densities = kde.pdf(possible_returns)
    # get the expected return by multiplying each possible return by its probability and summing over all possible returns
    np.sum(possible_returns * (densities/np.sum(densities)))
    """,
            language="python")

    # -------------------------

    # Sidebar
    st.sidebar.header("Select Tickers")
    available_tickers = list_tickers()

    # get risk free rate
    risk_free_rate = manager.get("risk_free_rate")

    # select tickers
    selected_tickers = manager.get("tickers")
    time.sleep(0.2)  # wait for the cookie to be set
    selected_tickers = st.sidebar.multiselect(
        "Tickers", available_tickers, default=selected_tickers or ["SPY"])
    manager.set("tickers", selected_tickers, key="tickers")
    # get ticker data
    tickers: dict[str, pd.DataFrame] = get_ticker_data(selected_tickers)

    # -------------------------

    # build expected returns
    expected_returns = {ticker: calculate_expected_return(
        ticker, df) for ticker, df in tickers.items()}
    for ticker, expected_return in expected_returns.items():
        if expected_return == -1:
            st.error(f"Not enough data for {ticker}")

    # build returns dataframe
    returns_df = pd.DataFrame()
    for ticker, df in tickers.items():
        returns_df[ticker] = df["return"]
    cov_df = returns_df.cov()
    # efficient frontier
    
    
    # -------------------------
    risk_free_rate = st.number_input("Risk Free Rate (%)", min_value=0.0, max_value=100.0, value=float(manager.get("risk_free_rate")) or 4.625, step=0.01)
    manager.set("risk_free_rate", risk_free_rate, key="risk_free_rate")
    col1, col2 = st.columns(2)
    with col1:
        # max sharpe portfolio
        st.subheader("Max Sharpe Portfolio")
        max_sharpe_ef = EfficientFrontier(list(expected_returns.values()), cov_df)
        max_sharpe_weights = max_sharpe_ef.max_sharpe(risk_free_rate=risk_free_rate/100)
        cleaned_weights = pd.DataFrame(
            {"tickers": list(expected_returns.keys()), "weights": list(max_sharpe_weights.values())})
        
        # plot pie charts using plotly
        fig = go.Figure(data=[go.Pie(labels=cleaned_weights["tickers"],
                        values=cleaned_weights["weights"], textinfo="label+percent", insidetextorientation="auto")])
        st.plotly_chart(fig)
        max_sharpe_performance = max_sharpe_ef.portfolio_performance(verbose=True)
        
    with col2:
        # min volatility portfolio
        st.subheader("Min Volatility Portfolio")
        min_vol_ef = EfficientFrontier(list(expected_returns.values()), cov_df)
        min_vol_weights = min_vol_ef.min_volatility()
        cleaned_weights = pd.DataFrame(
            {"tickers": list(expected_returns.keys()), "weights": list(min_vol_weights.values())})
        
        # plot pie charts using plotly
        fig = go.Figure(data=[go.Pie(labels=cleaned_weights["tickers"],
                        values=cleaned_weights["weights"], textinfo="label+percent", insidetextorientation="auto")])
        st.plotly_chart(fig)
        min_vol_performance = min_vol_ef.portfolio_performance(verbose=True)
    
    # write performance metrics
    st.write(pd.DataFrame({"Max Sharpe Portfolio": max_sharpe_performance, "Min Volatility Portfolio": min_vol_performance}, index=["Expected Return", "Volatility", "Sharpe Ratio"]))

if __name__ == "__main__":
    main()
