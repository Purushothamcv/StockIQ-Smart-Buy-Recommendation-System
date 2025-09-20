import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle 
df=pd.read_csv("top1kstocks.csv")
print(df.shape)
df.replace(["",0],np.nan,inplace=True)
df['p/e']=df['p/e'].fillna(df['p/e'].mean())
df_other_than_pe=df.columns.difference(['p/e'])
df[df_other_than_pe]=df[df_other_than_pe].ffill()
df.isnull().sum()
# print(df)
x=df[['cmp', 'p/e', 'mar_cap', 'div_yld_%', 'np_qtr', 'qtr_profit_%', 'sales_qtr', 'qtr_saeles_%', 'roce_%', 'pat_ann']]
df['future_cmp']=df['cmp'].shift(-1)
df['target']=np.where(df['future_cmp']>df['cmp']*1.1,'0','1')
# print(df['target'])
y=df['target']
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
models = {
    'lr':LogisticRegression(),
    'rf':RandomForestClassifier(),
    'knn':KNeighborsClassifier(n_neighbors=3),
    'gbc':GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    }
s=0
for name,model in models.items():
    model.fit(train_x,train_y)
    y_pred=model.predict(test_x)
    acc=accuracy_score(test_y,y_pred)
    print(f"{name}:Accuracy:{acc}")
    scores = cross_val_score(model,x, y, cv=5)  
    print(scores)
    print("Average Score:", scores.mean())
    if scores.mean()>s:
        s=scores.mean()
        best_model=model
# rf=RandomForestClassifier()
# rf.fit(train_x,train_y)
def predict_stock(best_model, input_data):
    prediction = best_model.predict(input_data)
    return prediction[0]
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
import joblib
joblib.dump(best_model, "model.joblib")

# import streamlit as st
# st.title("Stock Buy Recommendation System")     
# st.markdown("### Enter Stock Details and Fundamentals:")
# stock_name = st.text_input("Enter the Name of the Stock")
# current_market_price = st.number_input("Current Market Price", min_value=0.0, step=1.0)
# price_to_earnings = st.number_input("Price to Earnings (P/E)", min_value=0.0, step=0.1)
# market_cap = st.number_input("Market Capitalization", min_value=0.0, step=100.0)
# dividend_yield_percent = st.number_input("Dividend Yield (%)", min_value=0.0, step=0.1)
# net_profit_quarter = st.number_input("Net Profit (Quarter)", min_value=0.0, step=100.0)
# quarter_profit_percent = st.number_input("Quarter Profit Change (%)", step=0.1)     
# sales_quarter = st.number_input("Sales (Quarter)", min_value=0.0, step=100.0)
# quarter_sales_percent = st.number_input("Quarter Sales Change (%)", step=0.1)
# return_on_capital_employed = st.number_input("ROCE (%)", step=0.1)
# annual_profit = st.number_input("Annual Profit", min_value=0.0, step=100.0)
# if st.button("Predict"):
#     input_data = pd.DataFrame([[
#         current_market_price, price_to_earnings, market_cap, dividend_yield_percent,
#         net_profit_quarter, quarter_profit_percent, sales_quarter, quarter_sales_percent,
#         return_on_capital_employed, annual_profit
#     ]], columns=[
#         'cmp', 'p/e', 'mar_cap', 'div_yld_%', 'np_qtr', 'qtr_profit_%',
#         'sales_qtr', 'qtr_saeles_%', 'roce_%', 'pat_ann'
#     ])
#     prediction = predict_stock(best_model, input_data)
#     result = "Buy ✅" if prediction == '0' else "Not Buy ❌"
#     st.subheader(f"Recommendation for {stock_name if stock_name else 'this stock'}: {result}")
# # print(predict_stock(rf,[[100,15,20000,2,500,10,1000,5,15,200]]))


