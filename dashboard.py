
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="E-Commerce Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "data/ecommerce_sales.csv"

st.title("📊 E-Commerce Customer & Revenue Intelligence")
st.caption("Python • NumPy • Pandas • Matplotlib • Seaborn • RFM Analytics")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df = df.drop_duplicates()
    df["City"] = df["City"].fillna("Unknown")
    df["Discount"] = df["Discount"].fillna(df["Discount"].median())
    df["Rating"] = df["Rating"].fillna(df["Rating"].median())
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

categories = st.sidebar.multiselect(
    "Category",
    sorted(df["Category"].dropna().unique()),
    default=sorted(df["Category"].dropna().unique())
)

cities = st.sidebar.multiselect(
    "City",
    sorted(df["City"].dropna().unique()),
    default=sorted(df["City"].dropna().unique())
)

filtered = df[
    df["Category"].isin(categories) &
    df["City"].isin(cities)
].copy()

# KPI cards
total_revenue = filtered["Revenue"].sum()
total_profit = filtered["Profit"].sum()
orders = filtered["Order_ID"].nunique()
customers = filtered["Customer_ID"].nunique()
margin = total_profit / total_revenue * 100 if total_revenue else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Profit", f"₹{total_profit:,.0f}")
c3.metric("Orders", f"{orders:,}")
c4.metric("Customers", f"{customers:,}")
c5.metric("Profit Margin", f"{margin:.2f}%")

st.divider()

# Monthly performance
filtered["Year_Month"] = filtered["Order_Date"].dt.to_period("M").astype(str)

monthly = filtered.groupby("Year_Month").agg(
    Revenue=("Revenue", "sum"),
    Profit=("Profit", "sum")
)

st.subheader("📈 Monthly Performance")
st.line_chart(monthly)

# Category analysis
category = filtered.groupby("Category").agg(
    Revenue=("Revenue", "sum"),
    Profit=("Profit", "sum")
).sort_values("Revenue", ascending=False)

st.subheader("🏷️ Category Performance")
st.bar_chart(category)

# City analysis
city = filtered.groupby("City").agg(
    Revenue=("Revenue", "sum"),
    Profit=("Profit", "sum")
).sort_values("Revenue", ascending=False)

st.subheader("🏙️ City Performance")
st.bar_chart(city)

# RFM
analysis_date = filtered["Order_Date"].max() + pd.Timedelta(days=1)

rfm = filtered.groupby("Customer_ID").agg(
    Recency=("Order_Date", lambda x: (analysis_date - x.max()).days),
    Frequency=("Order_ID", "nunique"),
    Monetary=("Revenue", "sum")
)

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5, labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5, labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5, labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["Segment"] = np.select(
    [
        (rfm["R_Score"] >= 4) & (rfm["F_Score"] >= 4) & (rfm["M_Score"] >= 4),
        (rfm["R_Score"] >= 4) & (rfm["F_Score"] >= 3),
        (rfm["R_Score"] >= 3) & (rfm["M_Score"] >= 3),
        (rfm["R_Score"] <= 2) & (rfm["F_Score"] >= 3),
        (rfm["R_Score"] <= 2) & (rfm["M_Score"] <= 2)
    ],
    [
        "Champions",
        "Loyal Customers",
        "Potential Customers",
        "At Risk",
        "Lost Customers"
    ],
    default="Occasional Customers"
)

segment_counts = rfm["Segment"].value_counts()

st.subheader("👥 Customer Segmentation")
st.bar_chart(segment_counts)

st.subheader("⚠️ At-Risk Customers")

at_risk = rfm[
    rfm["Segment"].isin(["At Risk", "Lost Customers"])
].sort_values("Monetary", ascending=False)

st.dataframe(at_risk.head(20), use_container_width=True)

# Discount analysis
st.subheader("💰 Discount vs Profit")

discount = filtered.groupby(
    pd.cut(
        filtered["Discount"],
        bins=[-0.001, 0.10, 0.20, 0.30, 1.0],
        labels=["0-10%", "10-20%", "20-30%", "30%+"]
    ),
    observed=False
).agg(
    Revenue=("Revenue", "sum"),
    Profit=("Profit", "sum")
)

discount["Profit Margin %"] = discount["Profit"] / discount["Revenue"] * 100

st.dataframe(discount.round(2), use_container_width=True)

st.success("Dashboard loaded successfully.")
