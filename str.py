import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("top1kstocks.csv")
df.replace(["", 0], np.nan, inplace=True)
df['p/e'] = df['p/e'].fillna(df['p/e'].mean())
df_other_than_pe = df.columns.difference(['p/e'])
df[df_other_than_pe] = df[df_other_than_pe].ffill()

df['future_cmp'] = df['cmp'].shift(-1)
df['target'] = np.where(df['future_cmp'] > df['cmp'] * 1.1, '0', '1')

x = df[['cmp', 'p/e', 'mar_cap', 'div_yld_%', 'np_qtr', 'qtr_profit_%',
        'sales_qtr', 'qtr_saeles_%', 'roce_%', 'pat_ann']]
y = df['target']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(train_x, train_y)

st.title("Stock Buy Recommendation System")
st.markdown("### Enter Stock Details and Fundamentals:")

stock_name = st.text_input("Enter the Name of the Stock")
current_market_price = st.number_input("Current Market Price", min_value=0.0, step=1.0)
price_to_earnings = st.number_input("Price to Earnings (P/E)", min_value=0.0, step=0.1)
market_cap = st.number_input("Market Capitalization", min_value=0.0, step=100.0)
dividend_yield_percent = st.number_input("Dividend Yield (%)", min_value=0.0, step=0.1)
net_profit_quarter = st.number_input("Net Profit (Quarter)", min_value=0.0, step=100.0)
quarter_profit_percent = st.number_input("Quarter Profit Change (%)", step=0.1)
sales_quarter = st.number_input("Sales (Quarter)", min_value=0.0, step=100.0)
quarter_sales_percent = st.number_input("Quarter Sales Change (%)", step=0.1)
return_on_capital_employed = st.number_input("ROCE (%)", step=0.1)
annual_profit = st.number_input("Annual Profit", min_value=0.0, step=100.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[
        current_market_price, price_to_earnings, market_cap, dividend_yield_percent,
        net_profit_quarter, quarter_profit_percent, sales_quarter, quarter_sales_percent,
        return_on_capital_employed, annual_profit
    ]], columns=[
        'cmp', 'p/e', 'mar_cap', 'div_yld_%', 'np_qtr', 'qtr_profit_%',
        'sales_qtr', 'qtr_saeles_%', 'roce_%', 'pat_ann'
    ])

    prediction = model.predict(input_data)[0]
    result = "Buy ✅" if prediction == '0' else "Not Buy ❌"
    st.subheader(f"Recommendation for {stock_name if stock_name else 'this stock'}: {result}")
