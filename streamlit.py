import streamlit as st
import pandas as pd
import pickle
import joblib
# Load best model
with open("model.joblib", "rb") as f:
    best_model = joblib.load(f)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
        body {
            background-color: #f4f6f9;
        }
        .main {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-size: 40px !important;
            font-family: 'Arial Black', sans-serif;
        }
        .stButton button {
            background-color: #2ecc71;
            color: white;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-size: 18px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #27ae60;
            color: white;
        }
        .prediction-box {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            font-size: 22px;
            font-weight: bold;
        }
        .buy {
            background-color: #e8f8f5;
            color: #16a085;
        }
        .not-buy {
            background-color: #fdecea;
            color: #c0392b;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>💹 Stock Buy Recommendation System</h1>", unsafe_allow_html=True)

st.write("Fill in the stock fundamentals below to check whether you should **Buy ✅** or **Not Buy ❌** the stock.")

# Two-column layout for input
col1, col2 = st.columns(2)

with col1:
    stock_name = st.text_input("📌 Stock Name")
    cmp = st.number_input("💰 Current Market Price", min_value=0.0, step=1.0)
    pe = st.number_input("📊 Price to Earnings (P/E)", min_value=0.0, step=0.1)
    mar_cap = st.number_input("🏦 Market Capitalization", min_value=0.0, step=100.0)
    div_yld = st.number_input("💵 Dividend Yield (%)", min_value=0.0, step=0.1)

with col2:
    np_qtr = st.number_input("📈 Net Profit (Quarter)", min_value=0.0, step=100.0)
    qtr_profit_pct = st.number_input("📊 Quarter Profit Change (%)", step=0.1)
    sales_qtr = st.number_input("📉 Sales (Quarter)", min_value=0.0, step=100.0)
    qtr_sales_pct = st.number_input("📊 Quarter Sales Change (%)", step=0.1)
    roce = st.number_input("📊 ROCE (%)", step=0.1)
    pat_ann = st.number_input("📈 Annual Profit", min_value=0.0, step=100.0)

# Prediction Button
if st.button("🔍 Predict Recommendation"):
    input_data = pd.DataFrame([[cmp, pe, mar_cap, div_yld, np_qtr,
                                qtr_profit_pct, sales_qtr, qtr_sales_pct, roce, pat_ann]],
                                columns=['cmp', 'p/e', 'mar_cap', 'div_yld_%', 'np_qtr',
                                         'qtr_profit_%', 'sales_qtr', 'qtr_saeles_%',
                                         'roce_%', 'pat_ann'])
    
    prediction = best_model.predict(input_data)[0]
    result = "Buy ✅" if prediction == 0 else "Not Buy ❌"

    # Show styled result box
    if prediction == 0:
        st.markdown(f"<div class='prediction-box buy'>Recommendation for {stock_name if stock_name else 'this stock'}: <br> {result}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-box not-buy'>Recommendation for {stock_name if stock_name else 'this stock'}: <br> {result}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>⚡ Built with Streamlit | Smart Stock Advisor ⚡</p>", unsafe_allow_html=True)
