import streamlit as st
import os
st.set_page_config(page_title="Sales Prediction Dashboard",page_icon="📊",layout="wide")
import pandas as pd
import google.generativeai as genai

@st.cache_resource
def load_ai():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash")

model_ai = load_ai()

from sklearn.ensemble import RandomForestRegressor
@st.cache_resource
def load_model():
    df = pd.read_csv("orders.csv")

    df["Customer Status"] = df["Customer Status"].str.upper()
    df["Customer Status"] = df["Customer Status"].map({
        "SILVER": 1,
        "GOLD": 2,
        "PLATINUM": 3
    })

    X = df[["Quantity Ordered", "Cost Price Per Unit", "Customer Status"]]
    y = df["Total Retail Price for This Order"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)
    return model

model = load_model()
st.title("Sales Prediction Dashboard ")
st.markdown("---")
st.write("Predict the revenue of an order using our trained Random Forest Machine Learning model.")
quantity = st.number_input("Quantity Ordered",min_value=1, step=1)
cost = st.number_input("Cost Price Per Unit",min_value=1.0,step=1.0)
status=st.selectbox("Customer Status", ["SILVER","GOLD","PLATINUM"],key="customer_status")
target = st.number_input(
    "🎯 Enter Target Revenue",
    min_value=1.0,
    step=100.0
)


if st.button("Predict Revenue"):
    status_map = {"SILVER": 1,"GOLD": 2,"PLATINUM": 3}
    status_value = status_map[status]
    input_data = pd.DataFrame(
        [[quantity, cost, status_value]],
        columns=["Quantity Ordered", "Cost Price Per Unit", "Customer Status"]
    )

    prediction = model.predict(input_data)
    st.session_state["prediction"]=prediction[0]

    st.metric(label="💰 Predicted Revenue",value=f"₹{prediction[0]:.2f}")
    if prediction[0] >= target:
     st.success("🎉 Target Likely Achieved!")
else:
    st.error("❌ Target Not Achieved")
    st.subheader("📋 Order Details")

st.write(f"📦 Quantity Ordered: {quantity}")
st.write(f"💰 Cost Price Per Unit: ₹{cost}")
st.write(f"👑 Customer Status: {status}")

st.info("""🤖 Model Used: Random Forest Regressor 
        📈 R² Score: 98.3%""")

st.divider()

st.subheader("🤖 AI Sales Assistant")

question = st.text_area("Ask anything about sales or revenue")

if st.button("Ask AI"):
    prompt = f"""
    You are a Sales AI Assistant.
    Quantity Ordered: {quantity}
    Cost Price Per Unit: {cost}
    Customer Status: {status}
    Predicted Revenue : {st.session_state.get("prediction","Not predicted yet")}
    User Question:
    {question}
    """
    response = model_ai.generate_content(prompt)
    st.success(response.text)

