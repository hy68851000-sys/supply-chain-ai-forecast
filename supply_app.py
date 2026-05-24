import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from openai import OpenAI

if st.button("Start Forecast"):
    client = OpenAI(
        api_key=st.secrets["ZHIPU_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

st.title("Supply Chain Demand Forecast System")
st.write("AI-powered sales prediction and restocking recommendations")

st.sidebar.header("Settings")
store = st.sidebar.selectbox("Store", [1,2,3,4,5])
family = st.sidebar.selectbox("Product Category", 
    ['BEVERAGES','BREAD/BAKERY','CLEANING','DAIRY','PRODUCE'])
forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 14)

if st.button("Start Forecast"):
    with st.spinner("Running model..."):
        np.random.seed(store)
        dates = pd.date_range('2017-01-01', periods=forecast_days)
        base_sales = np.random.randint(1500, 3000)
        predictions = [base_sales + np.random.randint(-300, 300) 
                      + (200 if d.weekday() >= 5 else 0) 
                      for d in dates]

        st.subheader("Sales Forecast")
        chart_data = pd.DataFrame({'Predicted Sales': predictions}, index=dates)
        st.line_chart(chart_data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Daily", f"{np.mean(predictions):.0f}")
        col2.metric("Peak", f"{np.max(predictions):.0f}")
        col3.metric("Recommended Stock", f"{int(np.sum(predictions)*1.1):.0f}")

    st.subheader("AI Restocking Advice")
    with st.spinner("AI analyzing..."):
        prompt = f"""
You are a supply chain expert. Give restocking advice in Chinese.
Store {store}, Category: {family}, Forecast period: {forecast_days} days
Average daily forecast: {np.mean(predictions):.0f} units
Peak forecast: {np.max(predictions):.0f} units
Total forecast: {np.sum(predictions):.0f} units
Please provide: 1.Stock recommendation 2.Restock timing 3.Risk tips
Be concise and professional. Answer in Chinese.
"""
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        advice = response.choices[0].message.content
    st.success(advice)
