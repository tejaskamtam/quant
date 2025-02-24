import pandas as pd
from sqlalchemy import create_engine, text
import os
import dotenv
from scipy.stats import entropy, gaussian_kde, norm
import numpy as np
dotenv.load_dotenv()
CONNECTION = os.getenv("SQLALCHEMY_STOCKS1DAY_URI")
engine = create_engine(CONNECTION)


def list_tickers() -> list[str]:
    with engine.connect() as conn:
        cursor = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table';"))
        return [name[0] for name in cursor]


def get_ticker_data(tickers: str | list[str], index_col: str = None) -> list[float]:
    assert isinstance(tickers, str) or isinstance(
        tickers, list), "Tickers must be a string or a list"
    if isinstance(tickers, str):
        assert tickers in list_tickers(), f"Ticker {tickers} not found"
        return pd.read_sql_table(tickers, engine, index_col=index_col)
    else:
        assert all(ticker in list_tickers()
                   for ticker in tickers), "All tickers must be in the list of tickers"
        return {ticker: pd.read_sql_table(ticker, engine, index_col=index_col) for ticker in tickers}


def calculate_expected_return(df: pd.DataFrame | str) -> float:
    if isinstance(df, str):
        df = get_ticker_data(df, index_col="timestamp")
    # get rolling annual returns at every day
    annual_returns = df["return"][::-
                                  1].rolling(window=251).apply(lambda x: np.prod(x+1)-1).dropna()
    # estimate the PDF using gaussian kernel density estimation
    try:
        kde = gaussian_kde(annual_returns)
    except:
        return -1
    # get a range of 1000 discrete possible returns,
    # bounding daily returns between 100% and -100%
    possible_returns = np.linspace(-1, 1, 1000)
    # get density estimates for each possible return
    densities = kde.pdf(possible_returns)
    # get the expected return by multiplying each possible return by its probability
    # then summing over all possible returns
    return np.sum(possible_returns * (densities/np.sum(densities)))


def calculate_annualized_volatility(ticker: str) -> pd.DataFrame:
    assert ticker in list_tickers(), f"Ticker {ticker} not found"
    df = get_ticker_data(ticker, index_col="timestamp")

    def vol(df, period): return df.resample(
        period).last().dropna()["return"].std()

    return pd.DataFrame({
        "Stat": "Annualized Volatility",
        "D": vol(df, "D") * np.sqrt(251),
        "W": vol(df, "W") * np.sqrt(52),
        "M": vol(df, "M") * np.sqrt(12),
        "Q": vol(df, "Q") * np.sqrt(4),
        "Y": vol(df, "Y")
    }, index=["Stat"])


def calculate_betas(ticker: str, market: str) -> pd.DataFrame:
    assert ticker in list_tickers(), f"Ticker {ticker} not found"
    assert market in list_tickers(), f"Market {market} not found"
    df = get_ticker_data(ticker, index_col="timestamp")
    df_market = get_ticker_data(market, index_col="timestamp")

    def beta(df, df_market, period): return df.resample(period).last().dropna()["return"].cov(df_market.resample(
        period).last().dropna()["return"]) / df_market.resample(period).last().dropna()["return"].var()

    daily_beta = beta(df, df_market, "D")
    weekly_beta = beta(df, df_market, "W")
    Monthly_beta = beta(df, df_market, "M")
    quarterly_beta = beta(df, df_market, "Q")
    yearly_beta = beta(df, df_market, "Y")
    return pd.DataFrame({
        "Stat": "Beta ({})".format(market),
        "D": daily_beta,
        "W": weekly_beta,
        "M": Monthly_beta,
        "Q": quarterly_beta,
        "Y": yearly_beta
    }, index=["Stat"])


def calculate_KLDivergence(ticker: str) -> pd.DataFrame:
    assert ticker in list_tickers(), f"Ticker {ticker} not found"
    df = get_ticker_data(ticker, index_col="timestamp")

    def kl_divergence(df, period): return entropy(gaussian_kde(df.resample(period).last().dropna()["return"]).pdf(
        np.linspace(-1, 1, 1000)), norm.pdf(np.linspace(-1, 1, 1000), loc=df["return"].mean(), scale=df["return"].std()))

    daily_kl_divergence = kl_divergence(df, "D")
    weekly_kl_divergence = kl_divergence(df, "W")
    Monthly_kl_divergence = kl_divergence(df, "M")
    quarterly_kl_divergence = kl_divergence(df, "Q")
    yearly_kl_divergence = kl_divergence(df, "Y")
    return pd.DataFrame({
        "Stat": "KL Div (Normality)",
        "D": daily_kl_divergence,
        "W": weekly_kl_divergence,
        "M": Monthly_kl_divergence,
        "Q": quarterly_kl_divergence,
        "Y": yearly_kl_divergence
    }, index=["Stat"])


def calculate_future_value(
    principal: float,
    contribution: float,
    contribution_frequency: str,
    interest_rate: float,
    compound_frequency: str,
    time_period: float
) -> pd.DataFrame:
    """
    Calculate the future value of an investment with regular contributions at each timestep.

    Parameters:
    - principal (float): The initial amount of money.
    - contribution (float): The regular contribution amount.
    - contribution_frequency (str): 'Monthly' or 'Yearly'.
    - interest_rate (float): Annual interest rate (in percentage).
    - compound_frequency (str): 'Monthly' or 'Yearly'.
    - time_period (float): Time period in years.

    Returns:
    - pd.DataFrame: A DataFrame with 'Period' and 'Balance' columns showing the balance at each timestep.
    """
    r = interest_rate / 100  # Convert percentage to decimal

    # Determine compound frequency
    if compound_frequency == 'Monthly':
        n = 12
    elif compound_frequency == 'Yearly':
        n = 1
    else:
        raise ValueError(
            "Invalid compound frequency. Choose from 'Monthly' or 'Yearly'.")

    # Determine contribution frequency
    if contribution_frequency == 'Monthly':
        k = 12
    elif contribution_frequency == 'Yearly':
        k = 1
    else:
        raise ValueError(
            "Invalid contribution frequency. Choose from 'Monthly' or 'Yearly'.")

    t = time_period
    total_periods = int(n * t)
    periods_per_contribution = int(n / k)

    balance = principal
    data = []

    for period in range(1, total_periods + 1):
        # Apply interest
        balance += balance * (r / n)

        # Apply contribution if it's the contribution period
        if period % periods_per_contribution == 0:
            balance += contribution

        # Append the current balance
        data.append({
            "Period": period,
            "Balance": balance
        })

    df = pd.DataFrame(data)
    return df
