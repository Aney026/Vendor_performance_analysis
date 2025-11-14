#!/usr/bin/env python
# coding: utf-8

# In[32]:


# EDA

import pandas as pd
import sqlite3


# In[33]:


# creating db connection
conn = sqlite3.connect('inventory.db')


# In[34]:


# checking tables present in the DB
tables = pd.read_sql_query("select name from sqlite_master where type = 'table'",conn)
tables


# In[35]:


for table in tables['name']:
    print('-'*50, f'{table}', '-'*50)
    query = f"SELECT COUNT(*) AS count FROM {table}"
    print('Count of records:', pd.read_sql(query, conn)['count'].values[0])
    display(pd.read_sql(f"select * from {table} limit 5",conn))


# In[36]:


purchases = pd.read_sql_query("select * from purchases where VendorNumber = 4466",conn)
purchases


# In[37]:


purchases_price = pd.read_sql_query("select * from purchase_prices where VendorNumber = 4466",conn)
purchases_price


# In[38]:


invoices = pd.read_sql_query("select * from vendor_invoice where VendorNumber = 4466",conn)
invoices


# In[39]:


sales = pd.read_sql_query("select * from sales where VendorNo = 4466",conn)
sales


# In[40]:


purchases.groupby(['Brand', 'PurchasePrice'])[['Quantity', 'Dollars']].sum()


# In[41]:


purchases_price


# In[42]:


invoices['PONumber'].nunique()


# In[43]:


invoices.shape


# In[44]:


sales.groupby('Brand')[['SalesDollars','SalesPrice','SalesQuantity']].sum()


# In[45]:


invoices.columns


# In[46]:


freight_summary = pd.read_sql_query("""select VendorNumber, sum(Freight) as FreightCost
from vendor_invoice
group by VendorNumber""",conn)
freight_summary


# In[47]:


pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
purchases_price.columns


# In[48]:


pd.read_sql_query("""
    SELECT
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.PurchasePrice,
        pp.Volume,
        pp.Price AS ActualPrice,
        SUM(p.Quantity) AS TotalPurchaseQuantity,
        SUM(p.Dollars) AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp
        ON p.Brand = pp.Brand
        where p.PurchasePrice > 0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand
    ORDER BY TotalPurchaseDollars
""", conn)



# In[49]:


sales.columns


# In[50]:


pd.read_sql_query("""SELECT
VendorNo,
Brand,
sum(SalesDollars) as TotalSalesDollars,
sum(SalesPrice) as TotalSalesPrice,
sum(SalesQuantity) as TotalSalesQuantity,
sum(ExciseTax) as TotalExciseTax
from sales
group by VendorNo,Brand""",conn)


# In[51]:


vendor_sales_summary = pd.read_sql_query("""
    WITH FreightSummary AS (
        SELECT
            VendorNumber,
            SUM(Freight) AS FreightCost
        FROM vendor_invoice
        GROUP BY VendorNumber
    ),

    PurchaseSummary AS (
        SELECT
            p.VendorNumber,
            p.VendorName,
            p.Brand,
            p.Description,        
            p.PurchasePrice,
            pp.Volume,
            pp.Price AS ActualPrice,
            SUM(p.Quantity) AS TotalPurchaseQuantity,
            SUM(p.Dollars) AS TotalPurchaseDollars
        FROM purchases p
        JOIN purchase_prices pp
            ON p.Brand = pp.Brand
        WHERE p.PurchasePrice > 0
        GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Volume, pp.Price
    ),

    SalesSummary AS (
        SELECT
            VendorNo,
            Brand,
            SUM(SalesDollars) AS TotalSalesDollars,
            SUM(SalesPrice) AS TotalSalesPrice,
            SUM(SalesQuantity) AS TotalSalesQuantity,
            SUM(ExciseTax) AS TotalExciseTax
        FROM sales
        GROUP BY VendorNo, Brand
    )

    SELECT
        ps.VendorNumber,
        ps.VendorName,
        ps.Brand,
        ps.Description,      
        ps.PurchasePrice,
        ps.ActualPrice,
        ps.Volume,
        ps.TotalPurchaseQuantity,
        ps.TotalPurchaseDollars,
        ss.TotalSalesQuantity,
        ss.TotalSalesDollars,
        ss.TotalSalesPrice,
        ss.TotalExciseTax,
        fs.FreightCost
    FROM PurchaseSummary ps
    LEFT JOIN SalesSummary ss
        ON ps.VendorNumber = ss.VendorNo
        AND ps.Brand = ss.Brand
    LEFT JOIN FreightSummary fs
        ON ps.VendorNumber = fs.VendorNumber
    ORDER BY ps.TotalPurchaseDollars DESC
""", conn)


# In[52]:


vendor_sales_summary


# In[53]:


vendor_sales_summary.dtypes


# In[54]:


vendor_sales_summary.isnull().sum()


# In[55]:


vendor_sales_summary['VendorName'].unique


# In[56]:


vendor_sales_summary['Description'].unique


# In[26]:


vendor_sales_summary['Volume'] = vendor_sales_summary['Volume'].astype('float64')


# In[57]:


vendor_sales_summary.fillna(0,inplace = True)


# In[58]:


vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars']-vendor_sales_summary['TotalPurchaseDollars']


# In[59]:


vendor_sales_summary['GrossProfit'].min()


# In[60]:


vendor_sales_summary['ProfitMargin'] = (vendor_sales_summary['GrossProfit']/-vendor_sales_summary['TotalSalesDollars'])*100


# In[61]:


vendor_sales_summary['StockTurnover'] = vendor_sales_summary['TotalSalesQuantity']/vendor_sales_summary['TotalPurchaseQuantity']


# In[62]:


vendor_sales_summary['SalestoPurchaseRatio'] = vendor_sales_summary['TotalSalesDollars']/vendor_sales_summary['TotalPurchaseDollars']


# In[71]:


vendor_sales_summary.columns


# In[72]:


cursor = conn.cursor()


# In[73]:


cursor.execute("""CREATE TABLE vendor_sales_summary (
    VendorNumber INT,
    VendorName VARCHAR(100),
    Brand VARCHAR(100),
    Description VARCHAR(255),
    PurchasePrice DECIMAL(10,2),
    ActualPrice DECIMAL(10,2),
    Volume INT,
    TotalPurchaseQuantity INT,
    TotalPurchaseDollars DECIMAL(15,2),
    TotalSalesQuantity INT,
    TotalSalesDollars DECIMAL(15,2),
    TotalSalesPrice DECIMAL(15,2),
    TotalExciseTax DECIMAL(15,2),
    FreightCost DECIMAL(15,2),
    GrossProfit DECIMAL(15,2),
    ProfitMargin DECIMAL(15,2),
    StockTurnover DECIMAL(15,2),
    SalestoPurchaseRatio DECIMAL(8,2),
    PRIMARY KEY (VendorNumber, Brand)
);
""")


# In[74]:


pd.read_sql_query("select * from vendor_sales_summary",conn # final table from different table


# In[75]:


vendor_sales_summary.to_sql('vendor_sales_summary',conn, if_exists = 'replace',index = False)


# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sqlite3
from scipy.stats import ttest_ind
import scipy.stats as stats
warnings.filterwarnings('ignore')


# In[77]:


# creating database connection 
conn = sqlite3.connect('inventory.db')

# fetching data

df = pd.read_sql_query("select * from vendor_sales_summary",conn)
df.head()


# In[78]:


#EDA part 2
#summary stats

df.describe().T


# In[79]:


# Distribution plots for numerical columns
num_col = df.select_dtypes(include = np.number).columns

plt.figure(figsize =(15,10))
for i, col in enumerate(num_col):
    plt.subplot(4,4,i+1) # adjust grid layout as needed
    sns.histplot(df[col],kde = True,bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()
    


# In[80]:


# Outlier Detection

plt.figure(figsize =(15,10))
for i, col in enumerate(num_col):
    plt.subplot(4,4,i+1) # adjust grid layout as needed
    sns.boxplot(y= df[col])
    plt.title(col)
plt.tight_layout()
plt.show()
    


# In[81]:


#Removing inconsistencies


# Replace zero values temporarily to avoid division by zero
vendor_sales_summary['TotalSalesDollars'] = vendor_sales_summary['TotalSalesDollars'].replace(0, np.nan)
vendor_sales_summary['TotalPurchaseDollars'] = vendor_sales_summary['TotalPurchaseDollars'].replace(0, np.nan)
vendor_sales_summary['TotalPurchaseQuantity'] = vendor_sales_summary['TotalPurchaseQuantity'].replace(0, np.nan)

# Correct GrossProfit (your original was correct)
vendor_sales_summary['GrossProfit'] = (
    vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']
)

# ✅ FIXED Profit Margin (removed the wrong minus sign)
vendor_sales_summary['ProfitMargin'] = (
    vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars']
) * 100

# Stock Turnover
vendor_sales_summary['StockTurnover'] = (
    vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']
)

# Sales to Purchase Ratio
vendor_sales_summary['SalestoPurchaseRatio'] = (
    vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']
)

# Replace NaN back to 0 after calculations
vendor_sales_summary = vendor_sales_summary.fillna(0)


# In[82]:


vendor_sales_summary.to_sql(
    "vendor_sales_summary",
    conn,
    if_exists="replace",
    index=False
)


# In[83]:


df = pd.read_sql_query("""
    SELECT * 
    FROM vendor_sales_summary
    WHERE GrossProfit > 0
      AND ProfitMargin > 0
      AND TotalSalesQuantity > 0
""", conn)

print("Rows returned:", len(df))
df.head()


# In[ ]:


df


# In[84]:


# Distribution plots for numerical columns
num_col = df.select_dtypes(include = np.number).columns

plt.figure(figsize =(15,10))
for i, col in enumerate(num_col):
    plt.subplot(4,4,i+1) # adjust grid layout as needed
    sns.histplot(df[col],kde = True,bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[85]:


# counts plots for categorical columns
cat_cols = ["VendorName","Description"]

plt.figure(figsize =(12,5))
for i, col in enumerate(cat_cols):
    plt.subplot(1,2,i+1)
    sns.countplot(y=df[col], order =df [col].value_counts().index [:10]) # top 10 categories
    plt.title(f'Count plot of {col}')
plt.tight_layout()
plt.show()


# In[86]:


# correlation heatmap

plt.figure(figsize =(12,8))
corr_mat = df [num_col].corr()
sns.heatmap(corr_mat,annot = True, fmt = ".2f",cmap = "coolwarm",linewidth= 0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:


# identify brands that needs promotional or pricing adjustments which exhibit lower sales performance but higher profit margins.


# In[90]:


brand_performance = df.groupby ('Description').agg({
    'TotalSalesDollars' : 'sum',
    'ProfitMargin': 'mean'}).reset_index()
    


# In[91]:


low_sales_threshold = brand_performance['TotalSalesDollars'].quantile(0.15)
high_margin_threshold = brand_performance['ProfitMargin'].quantile(0.85)


# In[92]:


low_sales_threshold


# In[93]:


high_margin_threshold


# In[96]:


# filter brands with low sales but high sales margins

target_brands = brand_performance[
(brand_performance['TotalSalesDollars']<= low_sales_threshold) &
( brand_performance['ProfitMargin']>= high_margin_threshold)
]

print("Brands with low sales but high sales margins")
display(target_brands.sort_values('TotalSalesDollars'))


# In[104]:


def format_dollars(value):
    if value >= 1000000:
        return f'{value/1000000:.2f}M'
    elif value >= 1000:
        return f'{value/1000:.2f}K'
    else:
        return str(value)


# In[98]:


# which vendor and brands by sales petrformance

top_vendors = df.groupby('VendorName')['TotalSalesDollars'].sum().nlargest(10)
top_brands = df.groupby('Description')['TotalSalesDollars'].sum().nlargest(10)



# In[106]:


top_vendors.apply(lambda x : format_dollars (x))


# In[105]:


top_brands.apply(lambda x : format_dollars (x))


# In[108]:


# which vendors contribute the most to total purchase dollars?
vendor_performance = df.groupby('VendorName').agg ({
    'TotalPurchaseDollars':'sum',
    'GrossProfit': 'sum',
    'TotalSalesDollars': 'sum'
}).reset_index()    


# In[109]:


vendor_performance['PurchaseContribution%']=vendor_performance['TotalPurchaseDollars']/vendor_performance['TotalPurchaseDollars'].sum()


# In[113]:


vendor_performance=round(vendor_performance.sort_values('PurchaseContribution%',ascending = False),2)


# In[114]:


#Display top 10 vendors
top_vendors = vendor_performance.head(10)
top_vendors ['TotalSalesDollars'] = top_vendors ['TotalSalesDollars'].apply(format_dollars)
top_vendors ['TotalPurchaseDollars'] = top_vendors ['TotalPurchaseDollars'].apply(format_dollars)
top_vendors ['GrossProfit'] = top_vendors ['GrossProfit'].apply(format_dollars)
top_vendors


# In[117]:


# Create numeric backup columns BEFORE formatting
vendor_performance['TotalSalesDollars_num'] = vendor_performance['TotalSalesDollars']
vendor_performance['TotalPurchaseDollars_num'] = vendor_performance['TotalPurchaseDollars']
vendor_performance['GrossProfit_num'] = vendor_performance['GrossProfit']

# Take top 10 vendors
top_vendors = vendor_performance.head(10)

# Apply formatting ONLY to display columns
top_vendors['TotalSalesDollars'] = top_vendors['TotalSalesDollars'].apply(format_dollars)
top_vendors['TotalPurchaseDollars'] = top_vendors['TotalPurchaseDollars'].apply(format_dollars)
top_vendors['GrossProfit'] = top_vendors['GrossProfit'].apply(format_dollars)



# In[ ]:





# In[118]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# --- Pie Chart 1: Total Sales ---
axes[0].pie(
    top_vendors['TotalSalesDollars_num'],
    labels=top_vendors['VendorName'],
    autopct='%1.1f%%',
    startangle=140
)
axes[0].set_title('Top Vendors by Total Sales Dollars')

# --- Pie Chart 2: Total Purchase ---
axes[1].pie(
    top_vendors['TotalPurchaseDollars_num'],
    labels=top_vendors['VendorName'],
    autopct='%1.1f%%',
    startangle=140
)
axes[1].set_title('Top Vendors by Total Purchase Dollars')

# --- Pie Chart 3: Gross Profit ---
axes[2].pie(
    top_vendors['GrossProfit_num'],
    labels=top_vendors['VendorName'],
    autopct='%1.1f%%',
    startangle=140
)
axes[2].set_title('Top Vendors by Gross Profit')

plt.tight_layout()
plt.show()


# In[119]:


df['UnitPurchasePrice'] = df ['TotalPurchaseDollars']/df ['TotalPurchaseQuantity']


# In[120]:


df['OrderSize']= pd.qcut(df['TotalPurchaseQuantity'],q = 3, labels = ['Small','Medium','Large'])


# In[121]:


df.groupby('OrderSize')[['UnitPurchasePrice']].mean()


# In[122]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='OrderSize', y='UnitPurchasePrice', data=df)

plt.title('Unit Purchase Price by Order Size')
plt.xlabel('Order Size')
plt.ylabel('Unit Purchase Price')

plt.tight_layout()
plt.show()


# In[125]:


# which vendors have low inventory turnover, indicating excess stock and slow - moving product ?
df[df['StockTurnover']<1].groupby('VendorName')[['StockTurnover']].mean().sort_values('StockTurnover',ascending = True).head(10)


# In[128]:


# How much capital is locked in unsold per vendor, and which vendors contribute the most of it ?
df['UnsoldInventoryValue'] = (df['TotalPurchaseQuantity'] - df ['TotalSalesQuantity']) * df ['PurchasePrice']
print ('Total Unsold Capital: ',format_dollars(df['UnsoldInventoryValue'] .sum()))
                             


# In[129]:


# aggregate capital locked per vendor
inventory_value_per_vendor = df.groupby ('VendorName')['UnsoldInventoryValue'].sum().reset_index()


# In[130]:


inventory_value_per_vendor


# In[131]:


# who are top and low performing vendors ?
top_threshold = df ['TotalSalesDollars'].quantile(0.75)
low_threshold = df ['TotalSalesDollars'].quantile(0.25)


# In[133]:


top_vendors = df [df['TotalSalesDollars'] >= top_threshold]['ProfitMargin'].dropna()
low_vendors = df [df['TotalSalesDollars'] <= low_threshold]['ProfitMargin'].dropna()


# In[134]:


top_vendors


# In[135]:


low_vendors


# In[137]:


# Hypothesis testing 

top_threshold = df ['TotalSalesDollars'].quantile(0.75)
low_threshold = df ['TotalSalesDollars'].quantile(0.25)

top_vendors = df [df['TotalSalesDollars'] >= top_threshold]['ProfitMargin'].dropna()
low_vendors = df [df['TotalSalesDollars'] <= low_threshold]['ProfitMargin'].dropna()

#Two_sample T test
t_stat, p_value =ttest_ind(top_vendors,low_vendors, equal_var = False)
# Print result
print(f'T-Statistic: {t_stat : .4f}, P_Value : {p_value: .4f}')

if p_value < 0.05:
    print('Reject Hypothesis : There is a significant difference in profit margin b/w top and low performing vendors.')
else:
    print("Fail to reject hypothesis : No significant difference in profit margin")
          


# In[ ]:




