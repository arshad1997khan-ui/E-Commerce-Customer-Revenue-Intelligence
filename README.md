
# E-Commerce Customer & Revenue Intelligence System

## Project Overview

An end-to-end E-Commerce analytics project built using Python, NumPy, Pandas, Matplotlib and Seaborn.

The project analyzes:
- Revenue and profitability
- Customer behavior
- Product performance
- City performance
- Discount impact
- Revenue outliers
- RFM customer segmentation
- At-risk customers
- Monthly business trends

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit

## Advanced Features

### 1. NumPy Analytics
- Vectorized calculations
- Boolean indexing
- `np.where`
- `np.select`
- Percentiles
- IQR outlier detection
- Standard deviation
- Correlation
- `np.argsort`

### 2. RFM Customer Segmentation

Customers are evaluated using:

- Recency: How recently the customer purchased
- Frequency: How often the customer purchased
- Monetary: How much revenue the customer generated

Segments:
- Champions
- Loyal Customers
- Potential Customers
- At Risk
- Lost Customers
- Occasional Customers

### 3. Business Analysis

The project identifies:
- Highest revenue categories
- Most profitable categories
- Top cities
- Most profitable products
- High-revenue / low-margin products
- Discount impact on profitability
- Revenue outliers
- At-risk customers

## Folder Structure

```text
Ecommerce_Analytics/
│
├── data/
│   └── ecommerce_sales.csv
│
├── outputs/
│   └── generated automatically
│
├── analysis.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## How to Run

### Step 1: Install libraries

```bash
python -m pip install -r requirements.txt
```

### Step 2: Run analytics

```bash
python analysis.py
```

Charts and CSV reports will be created inside `outputs/`.

### Step 3: Run dashboard

```bash
streamlit run dashboard.py
```

## Interview Explanation

> I developed an end-to-end E-Commerce Customer and Revenue Intelligence System using Python, NumPy, Pandas, Matplotlib and Seaborn. I performed data cleaning, feature engineering, statistical analysis and IQR-based outlier detection using NumPy. I also implemented RFM-based customer segmentation to identify high-value, loyal and at-risk customers. Finally, I analyzed category, product, city and discount-level profitability and built a Streamlit dashboard for business reporting.

## Business Value

This project helps a business understand:
- Where revenue is coming from
- Which categories and products are profitable
- Which customers are valuable
- Which customers may churn
- Whether discounts are affecting profit
- Which transactions require investigation

## Resume Project Title

**E-Commerce Customer & Revenue Intelligence System**

## Resume Bullet

Developed an end-to-end E-Commerce analytics system using Python, NumPy, Pandas, Matplotlib and Seaborn to analyze revenue, profitability and customer behavior; implemented IQR outlier detection, RFM segmentation, discount-impact analysis and at-risk customer identification with a Streamlit dashboard.
