#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\Admin\customer_ subscription_ churn.csv")


# # 1: Dataset Exploration

# In[5]:


# total number of records
df.shape


# ### The dataset contains 2,800 customer records and 10 variables. Each record represents one customer and contains information about their subscription, usage, support activity, payment behavior, tenure and churn status.

# In[4]:


# Data Types
df.dtypes


# In[6]:


# Convert signup date
df["signup_date"] = pd.to_datetime(df["signup_date"])

df.dtypes


# ### Now signup_date becomes a proper date variable.

# In[7]:


#Summary statistics
df.describe()


# ### The average customer pays around ₹434 per month and uses the product for approximately 12.9 hours per week. The average customer has been subscribed for about 18.6 months and last logged in approximately 30 days ago.

# # 2: Data Preparation

# In[8]:


# Check missing values
df.isnull().sum()


# In[10]:


# Verify Column Data Types
df.dtypes


# In[ ]:


# Check Inconsistent Subscription Values


# In[11]:


df["plan_type"].unique()


# # Customer Engagement Analysis

# In[12]:


# Analyze Usage and Tenure
correlation = df[
    ["avg_weekly_usage_hours", "tenure_months"]
].corr()

print(correlation)


# In[13]:


# engagement patterns across subscription types
df["plan_type"].value_counts()


# In[14]:


# relationship between usage hours and tenure
correlation = df[
    ["avg_weekly_usage_hours", "tenure_months"]
].corr()

print(correlation)


# ### There is almost no relationship between customer tenure and weekly product usage. This means that customers who have been with the company for a longer period do not necessarily use the product more frequently.

# # Customer Retention Analysis

# In[15]:


# Churn Distribution Across Subscription Plans
churn_by_plan = pd.crosstab(
    df["plan_type"],
    df["churn"]
)

print(churn_by_plan)


# In[16]:


churn_rate_plan = df.groupby("plan_type")["churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print(churn_rate_plan.round(2))


# ### Churn rates are fairly similar across Basic, Standard and Premium plans. This suggests that subscription type alone does not explain customer churn. Other factors such as product usage, support activity, payment problems and customer inactivity may be more useful for identifying churn risk.

# In[17]:


# Churn Patterns Based on Tenure
df["tenure_group"] = pd.cut(
    df["tenure_months"],
    bins=[0, 6, 12, 24, 36],
    labels=[
        "1-6 Months",
        "7-12 Months",
        "13-24 Months",
        "25-36 Months"
    ]
)


# In[18]:


tenure_churn = df.groupby(
    "tenure_group",
    observed=True
)["churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print(tenure_churn.round(2))


# ### Churn remains fairly consistent across different tenure groups, ranging from approximately 55% to 58%. Therefore, tenure by itself does not appear to be a strong indicator of churn in this dataset.

# In[20]:


# Relationship Between Product Usage and Churn
df["usage_group"] = pd.cut(
    df["avg_weekly_usage_hours"],
    bins=[0, 5, 10, 15, 20, 25],
    labels=[
        "0-5 Hours",
        "5-10 Hours",
        "10-15 Hours",
        "15-20 Hours",
        "20-25 Hours"
    ]
)


# In[21]:


usage_churn = df.groupby(
    "usage_group",
    observed=True
)["churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print(usage_churn.round(2))


# ### Customers who use the product for only 0-5 hours per week have a churn rate of approximately 76%, which is significantly higher than the overall churn rate of 57.3%. This suggests that very low product usage is an important warning signal for potential churn.

# # Revenue Analysis

# In[22]:


# Revenue Distribution Across Subscription Plans
revenue_by_plan = df.groupby("plan_type")["monthly_fee"].sum()

print(revenue_by_plan)


# ### The Premium plan generates the highest monthly revenue at 659,856, followed by Standard at 372,267 and Basic at 183,677. Premium contributes the largest share of the companys monthly subscription revenue.

# In[23]:


# Average Revenue per Customer
average_revenue = df.groupby("plan_type")["monthly_fee"].mean()

print(average_revenue)


# In[24]:


# relationship between engagement and revenue
correlation = df[
    ["avg_weekly_usage_hours", "monthly_fee"]
].corr()

print(correlation)


# ### Engagement and revenue are not necessarily directly connected because revenue is mainly determined by the customers subscription plan. A highly engaged Basic customer can generate less revenue than a less engaged Premium customer. Therefore the company should consider both engagement and subscription value when identifying important customer segments.

# # Customer Support Analysis

# In[25]:


# Support Tickets vs Churn
df["support_group"] = pd.cut(
    df["support_tickets"],
    bins=[-1, 1, 3, 5, 8],
    labels=["0-1", "2-3", "4-5", "6-8"]
)


# In[26]:


support_churn = df.groupby(
    "support_group",
    observed=True
)["churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print(support_churn.round(2))


# ### Customers who raise more support tickets show higher churn rates. Customers with 6-8 support tickets have a churn rate of approximately 65%, compared with 49% for customers with 0-1 tickets. This suggests that customers experiencing repeated problems may be more likely to leave.

# In[27]:


# Support Tickets vs Subscription Plan
support_by_plan = df.groupby("plan_type")[
    "support_tickets"
].mean()

print(support_by_plan.round(2))


# ### Average support activity is almost identical across Basic, Standard and Premium customers. This means that customers on the Premium plan do not appear to require substantially more support than customers on the other plans.

# In[28]:


# Support Tickets vs Customer Engagement
support_engagement = df.groupby(
    "support_group",
    observed=True
)["avg_weekly_usage_hours"].mean()

print(support_engagement.round(2))


# # Data Visualization

# In[29]:


# Churn Distribution Chart
import matplotlib.pyplot as plt

churn_counts = df["churn"].value_counts()

plt.figure(figsize=(7, 4))

plt.bar(
    churn_counts.index,
    churn_counts.values
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.show()


# ### The dataset contains 1,605 churned customers compared with 1,195 retained customers. The overall churn rate is approximately 57.3% showing that customer retention is an important business issue.

# In[30]:


# Usage Patterns Across Subscription Types
usage_by_plan = df.groupby(
    "plan_type"
)["avg_weekly_usage_hours"].mean()

plt.figure(figsize=(7, 4))

plt.bar(
    usage_by_plan.index,
    usage_by_plan.values
)

plt.title("Average Weekly Usage by Subscription Plan")
plt.xlabel("Subscription Plan")
plt.ylabel("Average Weekly Usage (Hours)")

plt.show()


# ### Basic customers have the highest average weekly usage while Premium customers have the lowest. The difference is relatively small, showing that the higher priced Premium plan does not necessarily result in higher product usage.

# In[31]:


# Revenue Comparison Chart
revenue_by_plan = df.groupby(
    "plan_type"
)["monthly_fee"].sum()

plt.figure(figsize=(7, 4))

plt.bar(
    revenue_by_plan.index,
    revenue_by_plan.values
)

plt.title("Monthly Revenue by Subscription Plan")
plt.xlabel("Subscription Plan")
plt.ylabel("Monthly Revenue (₹)")

plt.show()


# ### Premium generates the highest monthly revenue at approximately 659,856 and contributes around 54% of total monthly revenue. This makes Premium customers particularly important from a revenue perspective.

# In[32]:


# Customer Engagement Analysis
df["engagement_level"] = pd.cut(
    df["avg_weekly_usage_hours"],
    bins=[0, 5, 10, 15, 20, 25],
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

engagement_counts = df["engagement_level"].value_counts(
    sort=False
)

plt.figure(figsize=(8, 4))

plt.bar(
    engagement_counts.index,
    engagement_counts.values
)

plt.title("Customer Engagement Distribution")
plt.xlabel("Engagement Level")
plt.ylabel("Number of Customers")

plt.show()


# ### The chart shows how customers are distributed across different product usage levels. This can help the company identify whether a large proportion of customers have low or high engagement.

# # Key Business Insights

# ### 1. The dataset has an overall churn rate of 57.3%, with 1,605 out of 2,800 customers classified as churned.
# ### 2. Customers using the product for 0–5 hours per week have a 76% churn rate, which is significantly higher than the overall churn rate.
# ### 3. Churn rates across Basic, Standard and Premium plans are relatively similar, ranging from approximately 56% to 58%.
# ### 4. Since Premium customers generate the highest revenue per customer, the company should closely monitor Premium customers who show signs of disengagement or payment problems.
# ### 5. Churned customers average 12.25 weekly usage hours compared with 13.75 hours among retained customers.
# ### 6. Customers with 6-8 support tickets have a 65% churn rate compared with 49% for customers with 0-1 tickets.
# ### 7. Churned customers have an average of 2.80 payment failures compared with 2.07 among retained customers, suggesting that payment problems may be an additional retention risk.

# # Business Recommendations

# ### 1. The company should identify customers whose weekly usage falls below 5 hours and provide product tutorials, reminders and personalized engagement campaigns.
# ### 2. Customers showing multiple warning signs such as low usage, long periods since login, payment failures and frequent support tickets should be identified early for retention campaigns.
# ### 3. Customers with repeated support tickets should receive proactive follow-up to understand and solve the underlying problems. This may help prevent frustration and potential churn.
# ### 4. Since Premium customers generate the highest revenue per customer, the company should closely monitor Premium customers who show signs of disengagement or payment problems.
# ### 5. The company should send timely payment reminders and make it easy for customers to update their payment details when payment failures occur.

# In[ ]:




