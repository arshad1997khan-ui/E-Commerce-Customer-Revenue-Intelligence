
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# E-COMMERCE CUSTOMER & REVENUE INTELLIGENCE SYSTEM

# Tools: Python, NumPy, Pandas, Matplotlib, Seaborn
# Focus: Data Quality, NumPy Analytics, Profitability, RFM,
#        Customer Segmentation, Outliers, Business Insights

DATA_PATH = "data/ecommerce_sales.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. LOAD DATA

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("E-COMMERCE CUSTOMER & REVENUE INTELLIGENCE SYSTEM")
print("=" * 70)

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# 2. DATA QUALITY AUDIT

print("\n--- DATA QUALITY AUDIT ---")
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# 3. DATA CLEANING

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

df = df.drop_duplicates()

df["City"] = df["City"].fillna("Unknown")
df["Discount"] = df["Discount"].fillna(df["Discount"].median())
df["Rating"] = df["Rating"].fillna(df["Rating"].median())

# Remove invalid numerical rows if any
df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0) &
    (df["Revenue"] >= 0)
].copy()


# 4. FEATURE ENGINEERING

df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.strftime("%b")
df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)

df["Discount_Pct"] = df["Discount"] * 100

# NumPy: vectorized business classification
df["Profit_Status"] = np.where(
    df["Profit"] > 0,
    "Profitable",
    "Loss"
)

df["Order_Value_Band"] = np.select(
    [
        df["Revenue"] < 1000,
        df["Revenue"].between(1000, 3000),
        df["Revenue"].between(3000, 6000),
        df["Revenue"] > 6000
    ],
    [
        "Low Value",
        "Medium Value",
        "High Value",
        "Premium Value"
    ],
    default="Unknown"
)


# 5. NUMPY DESCRIPTIVE ANALYTICS

revenue = df["Revenue"].to_numpy()
profit = df["Profit"].to_numpy()
discount = df["Discount"].to_numpy()

print("\n--- NUMPY BUSINESS STATISTICS ---")
print("Average Revenue:", np.mean(revenue))
print("Median Revenue:", np.median(revenue))
print("Revenue Std Dev:", np.std(revenue))
print("Minimum Revenue:", np.min(revenue))
print("Maximum Revenue:", np.max(revenue))

# Percentiles
Q1 = np.percentile(revenue, 25)
Q3 = np.percentile(revenue, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["Revenue_Outlier"] = np.where(
    (df["Revenue"] < lower_bound) |
    (df["Revenue"] > upper_bound),
    "Outlier",
    "Normal"
)

print("\n--- NUMPY REVENUE OUTLIER ANALYSIS ---")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Revenue Outliers:", (df["Revenue_Outlier"] == "Outlier").sum())

# NumPy argsort: highest revenue orders
revenue_array = df["Revenue"].to_numpy()
top_indices = np.argsort(revenue_array)[-10:][::-1]

print("\nTop 10 Revenue Orders:")
print(
    df.iloc[top_indices][
        ["Order_ID", "Category", "Product", "Revenue", "Profit"]
    ]
)


# 6. CORE BUSINESS KPIs

total_revenue = df["Revenue"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()
avg_order_value = df["Revenue"].mean()
overall_margin = (total_profit / total_revenue) * 100 if total_revenue else 0

print("\n--- BUSINESS KPIs ---")
print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Total Profit: ₹{total_profit:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Unique Customers: {total_customers:,}")
print(f"Average Order Value: ₹{avg_order_value:,.2f}")
print(f"Overall Profit Margin: {overall_margin:.2f}%")


# 7. CATEGORY PERFORMANCE

category_analysis = (
    df.groupby("Category")
    .agg(
        Orders=("Order_ID", "nunique"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Avg_Order_Value=("Revenue", "mean"),
        Avg_Discount=("Discount", "mean")
    )
    .sort_values("Revenue", ascending=False)
)

category_analysis["Profit_Margin_%"] = (
    category_analysis["Profit"] / category_analysis["Revenue"] * 100
)

print("\n--- CATEGORY PERFORMANCE ---")
print(category_analysis.round(2))


# 8. CITY PERFORMANCE

city_analysis = (
    df.groupby("City")
    .agg(
        Orders=("Order_ID", "nunique"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .sort_values("Revenue", ascending=False)
)

city_analysis["Profit_Margin_%"] = (
    city_analysis["Profit"] / city_analysis["Revenue"] * 100
)

print("\n--- TOP CITIES BY REVENUE ---")
print(city_analysis.head(10).round(2))


# 9. PRODUCT PROFITABILITY

product_analysis = (
    df.groupby(["Category", "Product"])
    .agg(
        Orders=("Order_ID", "nunique"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Avg_Discount=("Discount", "mean")
    )
    .sort_values("Profit", ascending=False)
)

print("\n--- TOP 10 PRODUCTS BY PROFIT ---")
print(product_analysis.head(10).round(2))

# Products with high revenue but relatively low profit
product_analysis["Profit_Margin_%"] = (
    product_analysis["Profit"] / product_analysis["Revenue"] * 100
)

high_revenue_low_margin = product_analysis[
    (product_analysis["Revenue"] >= product_analysis["Revenue"].quantile(0.75)) &
    (product_analysis["Profit_Margin_%"] <= product_analysis["Profit_Margin_%"].median())
].sort_values("Revenue", ascending=False)

print("\n--- HIGH REVENUE / LOW MARGIN PRODUCTS ---")
print(high_revenue_low_margin.head(10).round(2))


# 10. DISCOUNT IMPACT ANALYSIS

df["Discount_Band"] = pd.cut(
    df["Discount"],
    bins=[-0.001, 0.10, 0.20, 0.30, 1.0],
    labels=["0-10%", "10-20%", "20-30%", "30%+"]
)

discount_analysis = (
    df.groupby("Discount_Band", observed=False)
    .agg(
        Orders=("Order_ID", "nunique"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Avg_Profit=("Profit", "mean")
    )
)

discount_analysis["Profit_Margin_%"] = (
    discount_analysis["Profit"] / discount_analysis["Revenue"] * 100
)

print("\n--- DISCOUNT IMPACT ---")
print(discount_analysis.round(2))


# 11. RFM CUSTOMER ANALYSIS

analysis_date = df["Order_Date"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("Customer_ID")
    .agg(
        Recency=("Order_Date", lambda x: (analysis_date - x.max()).days),
        Frequency=("Order_ID", "nunique"),
        Monetary=("Revenue", "sum")
    )
)

# NumPy percentile scoring
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str) +
    rfm["F_Score"].astype(str) +
    rfm["M_Score"].astype(str)
)


# 12. CUSTOMER SEGMENTATION

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

segment_summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("Segment", "size"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Revenue=("Monetary", "sum")
    )
    .sort_values("Revenue", ascending=False)
)

print("\n--- RFM CUSTOMER SEGMENTS ---")
print(segment_summary.round(2))

print("\n--- TOP 10 CUSTOMER VALUES ---")
print(
    rfm.sort_values("Monetary", ascending=False)
    .head(10)
    .round(2)
)

# 13. AT-RISK CUSTOMER LIST
at_risk = (
    rfm[rfm["Segment"].isin(["At Risk", "Lost Customers"])]
    .sort_values("Monetary", ascending=False)
)

print("\n--- TOP AT-RISK CUSTOMERS ---")
print(at_risk.head(20).round(2))


# 14. MONTHLY TREND

monthly = (
    df.groupby("Year_Month")
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "nunique")
    )
)

monthly["Profit_Margin_%"] = (
    monthly["Profit"] / monthly["Revenue"] * 100
)

print("\n--- MONTHLY PERFORMANCE ---")
print(monthly.round(2))


# 15. NUMPY CORRELATION
discount_profit_corr = np.corrcoef(
    df["Discount"].to_numpy(),
    df["Profit"].to_numpy()
)[0, 1]

quantity_revenue_corr = np.corrcoef(
    df["Quantity"].to_numpy(),
    df["Revenue"].to_numpy()
)[0, 1]

print("\n--- NUMPY CORRELATION ANALYSIS ---")
print("Discount vs Profit Correlation:", round(discount_profit_corr, 4))
print("Quantity vs Revenue Correlation:", round(quantity_revenue_corr, 4))


# 16. BUSINESS INSIGHTS

best_category = category_analysis["Revenue"].idxmax()
most_profitable_category = category_analysis["Profit"].idxmax()
best_city = city_analysis["Revenue"].idxmax()
best_product = product_analysis["Profit"].idxmax()
best_segment = segment_summary["Revenue"].idxmax()

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

print(f"1. Highest revenue category: {best_category}")
print(f"2. Most profitable category: {most_profitable_category}")
print(f"3. Highest revenue city: {best_city}")
print(f"4. Most profitable product: {best_product}")
print(f"5. Highest-value customer segment by revenue: {best_segment}")
print(f"6. Revenue outliers detected: {(df['Revenue_Outlier'] == 'Outlier').sum():,}")
print(f"7. Discount vs Profit correlation: {discount_profit_corr:.3f}")


# 17. EXPORT ANALYTICS FILES

df.to_csv(f"{OUTPUT_DIR}/cleaned_sales_data.csv", index=False)
category_analysis.to_csv(f"{OUTPUT_DIR}/category_performance.csv")
city_analysis.to_csv(f"{OUTPUT_DIR}/city_performance.csv")
product_analysis.to_csv(f"{OUTPUT_DIR}/product_profitability.csv")
discount_analysis.to_csv(f"{OUTPUT_DIR}/discount_impact.csv")
rfm.to_csv(f"{OUTPUT_DIR}/customer_rfm_segments.csv")
segment_summary.to_csv(f"{OUTPUT_DIR}/customer_segment_summary.csv")
monthly.to_csv(f"{OUTPUT_DIR}/monthly_performance.csv")


# 18. VISUALIZATIONS

sns.set_theme(style="whitegrid")

# Monthly Revenue
plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly["Revenue"], marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_monthly_revenue.png", dpi=150)
plt.close()

# Monthly Profit
plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly["Profit"], marker="o")
plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_monthly_profit.png", dpi=150)
plt.close()

# Category Revenue
plt.figure(figsize=(10, 6))
category_analysis["Revenue"].sort_values().plot(kind="barh")
plt.title("Revenue by Category")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_category_revenue.png", dpi=150)
plt.close()

# Category Profit
plt.figure(figsize=(10, 6))
category_analysis["Profit"].sort_values().plot(kind="barh")
plt.title("Profit by Category")
plt.xlabel("Profit")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_category_profit.png", dpi=150)
plt.close()

# Customer Segments
plt.figure(figsize=(10, 6))
segment_summary["Customers"].sort_values().plot(kind="barh")
plt.title("Customers by RFM Segment")
plt.xlabel("Customers")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_customer_segments.png", dpi=150)
plt.close()

# Discount vs Profit
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x="Discount",
    y="Profit",
    hue="Category",
    alpha=0.65
)
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_discount_vs_profit.png", dpi=150)
plt.close()

# Revenue Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Revenue"], bins=40, kde=True)
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Order Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_revenue_distribution.png", dpi=150)
plt.close()

# Profit Margin by Category
plt.figure(figsize=(10, 6))
category_analysis["Profit_Margin_%"].sort_values().plot(kind="barh")
plt.title("Profit Margin by Category")
plt.xlabel("Profit Margin (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_category_profit_margin.png", dpi=150)
plt.close()

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"Analytics files and charts saved inside: {OUTPUT_DIR}/")
