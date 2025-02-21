import time
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from utils import list_tickers, get_ticker_data, calculate_betas, calculate_annualized_volatility, calculate_KLDivergence
import extra_streamlit_components as stx
from plotly.subplots import make_subplots
import pandas as pd
from numpy import sqrt


def main():
    # configs
    st.set_page_config(layout="wide")
    manager = stx.CookieManager()

    # -------------------------

    
    selected_ticker = manager.get("ticker")
    selected_chart_type = manager.get("chart_type")
    time.sleep(0.2) # wait for cookie to be set

    # Sidebar for ticker selection
    st.sidebar.header("Select Ticker")
    tickers = list_tickers()
    selected_ticker = st.sidebar.selectbox(
        "Ticker", tickers, index=tickers.index(selected_ticker or "SPY"))
    manager.set("ticker", selected_ticker, key="ticker")
    # get ticker data
    df: pd.DataFrame = get_ticker_data(selected_ticker)
    
    
    # Sidebar for chart type selection
    selected_chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Candlestick"], index=[
                                               "Line", "Candlestick"].index(selected_chart_type or "Line"))
    manager.set("chart_type", selected_chart_type, key="chart_type")

    # -------------------------

    st.header(f"{selected_ticker}")
    st.write(
        f"Daily data from {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')} to {df['timestamp'].iloc[0].strftime('%Y-%m-%d')}")

    # Price
    price_fig = make_subplots(specs=[[{"secondary_y": True}]])
    if selected_chart_type == "Candlestick":
        price_fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
        ),
            secondary_y=False,
        )
    else:
        price_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df['close'],
            name='Price',
        ),
            secondary_y=False,
        )
    # Plot Volume
    price_fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df['volume'],
        marker_color='rgba(246, 78, 139, 0.6)',
        name='Volume',
    ),
        secondary_y=True,
    )
    price_fig.update_yaxes(title_text='Price (USD)', secondary_y=False)
    price_fig.update_yaxes(title_text='Volume', secondary_y=True)
    price_fig.update_layout(height=600, xaxis_rangeslider_visible=False, legend=dict(
        orientation="h", yanchor="bottom", y=1, xanchor="left", x=0))
    st.plotly_chart(price_fig)

    # -------------------------

    # Returns
    returns_col, returns_dist_col = st.columns(2)
    with returns_col:
        fig_returns = go.Figure()
        # log return bar
        fig_returns.add_trace(go.Bar(
            x=df["timestamp"], y=df['log_return'], name='Log Returns', marker_color='#b75e11'))
        # return bar
        fig_returns.add_trace(go.Bar(
            x=df["timestamp"], y=df['return'], name='Returns', marker_color='#1a5885'))
        # volatility line
        df["volatility"] = df["return"][::-
                                        1].rolling(window=251).std().apply(lambda x: x * sqrt(251))
        fig_returns.add_trace(go.Scatter(x=df["timestamp"], y=df['volatility'],
                              name='1Y-Historical Annualized Volatility', marker_color='#562aaf', opacity=0.5))
        fig_returns.update_layout(yaxis_title='Return/Volatility', xaxis_title='Date', height=500,
                                  legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0))
        st.plotly_chart(fig_returns)
    with returns_dist_col:
        fig_distribution = ff.create_distplot([df["return"], df["log_return"]], [
                                              "Returns", "Log Returns"], histnorm="probability", bin_size=0.001, show_rug=False)
        fig_distribution.update_layout(barmode='overlay', xaxis_title='Return', yaxis_title='Probability', height=500, legend=dict(
            orientation="h", yanchor="bottom", y=1, xanchor="left", x=0))
        st.plotly_chart(fig_distribution)

    # calculate metrics
    ann_vols: pd.DataFrame = calculate_annualized_volatility(selected_ticker)
    betas: pd.DataFrame = calculate_betas(selected_ticker, 'SPY')
    kl_div: pd.DataFrame = calculate_KLDivergence(selected_ticker)

    metrics = pd.concat([ann_vols, betas, kl_div],
                        axis='index').set_index('Stat')
    st.write(metrics)


if __name__ == "__main__":
    main()
