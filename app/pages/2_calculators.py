import streamlit as st
from utils import calculate_future_value
import plotly.graph_objects as go
from tenforty import evaluate_return
import pandas as pd

def main():
    st.set_page_config(layout="wide")
    st.title("Calculators")
    st.write("---")

    st.subheader("Interest Calculator")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        principal = st.number_input(
            "Principal Amount ($)", min_value=0.0, value=1000.0, step=100.0)
        contribution = st.number_input(
            "Regular Contribution ($)", min_value=0.0, value=100.0, step=50.0)
        contribution_frequency = st.selectbox(
            "Contribution Frequency", ["Monthly", "Yearly"])
        interest_rate = st.number_input(
            "Annual Interest Rate (%)", min_value=0.0, value=5.0, step=0.5)
        compound_frequency = st.selectbox(
            "Compound Frequency", ["Monthly", "Yearly"])
        time_period = st.number_input(
            "Time Period (years)", min_value=0.1, value=10.0, step=0.5)
    with col2:
        # Calculate future values over time
        try:
            df = calculate_future_value(
                principal=principal,
                contribution=contribution,
                contribution_frequency=contribution_frequency,
                interest_rate=interest_rate,
                compound_frequency=compound_frequency,
                time_period=time_period
            )
        except Exception as e:
            st.error(str(e))

        final_value = df["Balance"].iloc[-1]

        st.write(
            f"**Future Value After {time_period} Years: ${final_value:,.2f}**")
        
        # Plotting the growth using Plotly
        interest_fig = go.Figure()
        interest_fig.add_trace(go.Scatter(
            x=df["Period"],
            y=df["Balance"],
            mode="lines",
            name="Balance",
        ))
        interest_fig.update_layout(
            xaxis_title=f"Period ({compound_frequency})",
            yaxis_title="Balance ($)",
        )
        st.plotly_chart(interest_fig)
    
    with col3:
        st.dataframe(df.style.format({"Balance": "${:,.2f}"}))

    st.write("---")

    # -------------------------

    st.subheader("Tax Calculator")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        filing_year = st.selectbox("Filing Year", options=range(2024,2018, -1), index=0)
        filing_status = st.selectbox("Filing Status", options=["Single", "Married/Joint", "Married/Sep", "Head_of_House", "Widow(er)"], index=0)
        state = st.selectbox("State", options=["TX", "NY", "MA", "AK", "FL", "NV", "SD", "CA", "WA", "WY", "None"], index=0)
        dependents = st.number_input("Dependents", min_value=0, value=0, step=1)
        deduction_type = st.selectbox("Deduction Type", options=["Standard", "Itemized"], index=0)
        income = st.number_input("W2 Income", min_value=0, value=100000, step=1000)
        
        advanced = st.checkbox("Advanced", value=False)
        
        #interest, qualified_dividends, ordinary_dividends, short_term_cap_gains, long_term_cap_gains, schedule_1_income, itemized_deductions, state_adjustment, incentive_stock_option_gain = None, None, None, None, None, None, None, None, None
        
        if advanced:
            interest                = st.number_input("Taxable Interest", min_value=0, value=0, step=1)
            qualified_dividends     = st.number_input("Qualified Dividends", min_value=0, value=0, step=1)
            ordinary_dividends      = st.number_input("Ordinary Dividends", min_value=0, value=0, step=1)
            short_term_cap_gains    = st.number_input("Short-Term Capital Gains", min_value=0, value=0, step=1)
            long_term_cap_gains     = st.number_input("Long-Term Capital Gains", min_value=0, value=0, step=1)
            schedule_1_income       = st.number_input("Schedule 1 Income", min_value=0, value=0, step=1)
            itemized_deductions         = st.number_input("Itemized Deductions", min_value=0, value=0, step=1)
            state_adjustment        = st.number_input("State Adjustment", min_value=0, value=0, step=1)
            incentive_stock_option_gain = st.number_input("Incentive Stock Option Gain", min_value=0, value=0, step=1)

    with col2:
        statement = evaluate_return(
                year=filing_year,
                filing_status=filing_status,
                state=state,
                num_dependents=dependents,
                standard_or_itemized=deduction_type,
                w2_income=income,
        ).model_dump()
        
        if advanced:
            statement = evaluate_return(
                year=filing_year,
                filing_status=filing_status,
                state=state,
                num_dependents=dependents,
                standard_or_itemized=deduction_type,
                w2_income=income,
                taxable_interest=interest,
                qualified_dividends=qualified_dividends,
                ordinary_dividends=ordinary_dividends,
                short_term_capital_gains=short_term_cap_gains,
                long_term_capital_gains=long_term_cap_gains,
                schedule_1_income=schedule_1_income,
                itemized_deductions=itemized_deductions,
                state_adjustment=state_adjustment,
                incentive_stock_option_gains=incentive_stock_option_gain,
        ).model_dump()
        
        form = pd.DataFrame({"values": statement.values()}, index=statement.keys())
        st.dataframe(form.style.format({"values": "${:,.2f}"}), use_container_width=True)
            



if __name__ == "__main__":
    main()
