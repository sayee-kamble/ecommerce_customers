import pandas as pd

# STEP 1: Load dataset
df = pd.read_excel("data/online_retail_II.xlsx")

print("Original data:")
print(df.head())
print(df.shape)


# STEP 2: Clean the data

# Remove missing Customer IDs
df = df.dropna(subset=["Customer ID"])

# Remove invalid quantities
df = df[df["Quantity"] > 0]

# Remove invalid prices
df = df[df["Price"] > 0]

# Calculate total amount
df["TotalAmount"] = df["Quantity"] * df["Price"]

print("After cleaning:", df.shape)
print(df.head())

# STEP 3: Create RFM features

# Reference date
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Create RFM table
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate",
             lambda x: (snapshot_date - x.max()).days),

    Frequency=("Invoice", "nunique"),

    Monetary=("TotalAmount", "sum")
)

print("\nRFM Customer Data:")
print(rfm.head())

print("\nNumber of customers:", len(rfm))

print("\nRFM Statistics:")
print(rfm.describe())

# STEP 4: Scale RFM features

from sklearn.preprocessing import StandardScaler

features = rfm[["Recency", "Frequency", "Monetary"]]

scaler = StandardScaler()

X = scaler.fit_transform(features)

print("\nScaled RFM Data:")
print(X[:5])

# STEP 5: K-Means Clustering

from sklearn.cluster import KMeans

# Create K-Means model
kmeans = KMeans(n_clusters=4, random_state=42)

# Train the model
kmeans.fit(X)

# Assign cluster to each customer
rfm["Cluster"] = kmeans.labels_

print("\nCustomer Clusters:")
print(rfm.head(10))

print("\nNumber of customers in each cluster:")
print(rfm["Cluster"].value_counts())

# STEP 6: Analyze each cluster

cluster_summary = rfm.groupby("Cluster")[[
    "Recency",
    "Frequency",
    "Monetary"
]].mean()

print("\nCluster Summary:")
print(cluster_summary)

# STEP 7: Visualize customer clusters

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    rfm["Frequency"],
    rfm["Monetary"],
    c=rfm["Cluster"]
)

plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.title("Customer Segmentation using K-Means")

plt.show()

# STEP 8: Evaluate clustering

from sklearn.metrics import silhouette_score

score = silhouette_score(X, rfm["Cluster"])

print("\nSilhouette Score:", score)