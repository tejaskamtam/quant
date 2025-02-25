import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from pypfopt import EfficientFrontier, plotting
import extra_streamlit_components as stx
from utils import get_ticker_data, list_tickers, calculate_expected_return
import pandas as pd
import plotly.graph_objects as go
import plotly.tools as tls


def main():
    # configs
    st.set_page_config(layout="wide")
    manager = stx.CookieManager()
    st.title("Portfolio Optimization")
    st.write("Expected return is calculated with the following method:")
    st.code("""
    from scipy.stats import gaussian_kde
    # get rolling cumulative annual compounded returns (CAGR) for each day of data
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
    expected_returns = {ticker: calculate_expected_return(df) for ticker, df in tickers.items()}
    for ticker, expected_return in expected_returns.items():
        if expected_return == -1:
            st.error(f"Not enough data for {ticker}")

    # build returns dataframe
    returns_df = pd.DataFrame()
    for ticker, df in tickers.items():
        returns_df[ticker] = df["return"]
    cov_df = returns_df.cov()

    risk_free_rate = st.number_input("Risk Free Rate (%)", min_value=0.0, max_value=100.0, value=float(manager.get("risk_free_rate") or 4.625) , step=0.01)
    manager.set("risk_free_rate", risk_free_rate, key="risk_free_rate")
    # data info
    col00, col01, col02 = st.columns(3, gap="large")
    with col00:
        st.subheader("Asset Correlation")
        # plot correlation matrix
        corr_fig = go.Figure(data=go.Heatmap(z=cov_df.corr().values, x=cov_df.corr().columns, y=cov_df.corr().index, colorscale="Viridis"))
        st.plotly_chart(corr_fig, key="corr_fig")
    with col01:
        st.subheader("Asset Covariance")
        # plot covariance matrix
        cov_fig = go.Figure(data=go.Heatmap(z=cov_df.values, x=cov_df.columns, y=cov_df.index, colorscale="Viridis"))
        st.plotly_chart(cov_fig, key="cov_fig")
    with col02:
        st.subheader("Asset Information")
        st.dataframe(pd.DataFrame({"Expected Return": list(expected_returns.values()), "Volatility": list(np.sqrt(np.diag(cov_df))), "Sharpe Ratio": (np.array(list(expected_returns.values()))-risk_free_rate/100)/np.sqrt(np.diag(cov_df))}, index=list(expected_returns.keys())), use_container_width=True)
    
    # -------------------------
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        # max sharpe portfolio
        st.subheader("Max Sharpe Portfolio")
        st.write(f"Maximizes portfolio Sharpe Ratio given a Risk Free Rate of {round(risk_free_rate, 3)}%")
        max_sharpe_ef = EfficientFrontier(list(expected_returns.values()), cov_df)
        max_sharpe_weights = max_sharpe_ef.max_sharpe(risk_free_rate=risk_free_rate/100)
        cleaned_weights = pd.DataFrame(
            {"tickers": list(expected_returns.keys()), "weights": list(max_sharpe_weights.values())})
        
        # plot pie charts using plotly
        max_sharpe_fig = go.Figure(data=[go.Pie(labels=cleaned_weights["tickers"],
                        values=cleaned_weights["weights"], textinfo="label+percent", insidetextorientation="auto")])
        st.plotly_chart(max_sharpe_fig, key="max_sharpe_fig")
        max_sharpe_performance = max_sharpe_ef.portfolio_performance(risk_free_rate=risk_free_rate/100)
        
        
    with col2:
        # min volatility portfolio
        st.subheader("Min Volatility Portfolio")
        st.write(f"Minimizes portfolio volatility independent of other variables.")
        min_vol_ef = EfficientFrontier(list(expected_returns.values()), cov_df)
        min_vol_weights = min_vol_ef.min_volatility()
        cleaned_weights = pd.DataFrame(
            {"tickers": list(expected_returns.keys()), "weights": list(min_vol_weights.values())})
        
        # plot pie charts using plotly
        min_vol_fig = go.Figure(data=[go.Pie(labels=cleaned_weights["tickers"],
                        values=cleaned_weights["weights"], textinfo="label+percent", insidetextorientation="auto")])
        st.plotly_chart(min_vol_fig, key="min_vol_fig")
        min_vol_performance = min_vol_ef.portfolio_performance(risk_free_rate=risk_free_rate/100)


    st.warning('Unreasonable targets for the following portfolio optimizations may lead to erroneous portfolio allocations.', icon="⚠️")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Max Return Portfolio")
        st.write(f"Maximizes portfolio return for a given target volatility:")
        target_volatility = st.number_input("Target Volatility (%)", min_value=0.0, max_value=100.0, value=max(np.sqrt(np.diag(cov_df)))*100-np.finfo(float).eps, step=0.01)
        max_return_ef = EfficientFrontier(list(expected_returns.values()), cov_df)
        try:
            max_return_weights = max_return_ef.efficient_risk(target_volatility=target_volatility/100)
            cleaned_weights = pd.DataFrame(
            {"tickers": list(expected_returns.keys()), "weights": list(max_return_weights.values())})

            # plot pie charts using plotly
            max_return_fig = go.Figure(data=[go.Pie(labels=cleaned_weights["tickers"],
                            values=cleaned_weights["weights"], textinfo="label+percent", insidetextorientation="auto")])
            st.plotly_chart(max_return_fig, key="max_return_fig")
            max_return_performance = max_return_ef.portfolio_performance(risk_free_rate=risk_free_rate/100)
        except Exception as e:
            st.error(str(e))
            max_return_performance = (None, None, None)

    with col4:
        # markowitz (efficient return) portfolio
        st.subheader("Markowitz Portfolio")
        st.write(f"Minimizes portfolio volatility for a given target return:")
        target_return = st.number_input("Target Return (%)", min_value=0.0, max_value=100.0, value=np.mean(list(expected_returns.values()))*100, step=0.01)
        efficient_return_ef = EfficientFrontier(list(expected_returns.values()), cov_df)
        try:
            efficient_return_weights = efficient_return_ef.efficient_return(target_return=target_return/100)
            cleaned_weights = pd.DataFrame(
            {"tickers": list(expected_returns.keys()), "weights": list(efficient_return_weights.values())})

            # plot pie charts using plotly
            efficient_return_fig = go.Figure(data=[go.Pie(labels=cleaned_weights["tickers"],
                            values=cleaned_weights["weights"], textinfo="label+percent", insidetextorientation="auto")])
            st.plotly_chart(efficient_return_fig, key="efficient_return_fig")

            efficient_return_performance = efficient_return_ef.portfolio_performance(risk_free_rate=risk_free_rate/100)
        except Exception as e:
            st.error(str(e))
            efficient_return_performance = (None, None, None)
            
    st.write("---")
    st.subheader("Efficient Frontier")
    col5, col6 = st.columns(2)
    with col5:
        # write performance metrics
        # get SPY benchmark
        spy_return_df = pd.DataFrame({"SPY":get_ticker_data("SPY")["return"]})
        spy_portfolio = EfficientFrontier([calculate_expected_return("SPY")], spy_return_df.cov())
        spy_portfolio.min_volatility()
        spy_performance = spy_portfolio.portfolio_performance(risk_free_rate=risk_free_rate/100)

        st.write(f"Risk Free Rate: {risk_free_rate}%")
        st.dataframe(pd.DataFrame({"SPY": spy_performance, "Max Sharpe Portfolio": max_sharpe_performance, "Min Volatility Portfolio": min_vol_performance, "Max Return Portfolio": max_return_performance, "Markowitz Portfolio": efficient_return_performance}, index=["Expected Return", "Volatility", "Sharpe Ratio"]), use_container_width=True)

    with col6:
        # plot efficient frontier
        min_vol_ret, min_vol_sigma, _ = min_vol_performance
        max_return_ret, max_return_sigma, _ = max_return_performance
        max_sharpe_ret, max_sharpe_sigma, _ = max_sharpe_performance
        efficient_return_ret, efficient_return_sigma, _ = efficient_return_performance

        capm = risk_free_rate/100 + (max_sharpe_ret-risk_free_rate/100)*np.linspace(0, 2, 100)
        capm_stdev = np.linspace(0, 2, 100)*max_sharpe_sigma

        ef = EfficientFrontier(list(expected_returns.values()), cov_df)
        fig, ax = plt.subplots()
        # plot efficient frontier
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        ax.legend().remove()
        plot_fig = tls.mpl_to_plotly(fig)
        plot_fig.update_layout(
            xaxis_title="Volatility",
            yaxis_title="Expected Return",
        )
        
        # plot CAPM
        plot_fig.add_trace(go.Scatter(x=capm_stdev, y=capm, mode="lines", name="CAPM", marker_symbol="star", marker_size=10, marker_color="orange"))
        # plot efficient portfolios
        for ret, sigma, name in [(min_vol_ret, min_vol_sigma, "Min Volatility Portfolio"), (max_return_ret, max_return_sigma, "Max Return Portfolio"), (max_sharpe_ret, max_sharpe_sigma, "Max Sharpe Portfolio"), (efficient_return_ret, efficient_return_sigma, "Markowitz Portfolio")]:
            plot_fig.add_trace(go.Scatter(x=[sigma], y=[ret], mode="markers", name=name, marker_symbol="star" if name=="Max Sharpe Portfolio" else "circle", marker_size=15))

        st.plotly_chart(plot_fig, key="efficient_frontier_fig")


if __name__ == "__main__":
    main()
