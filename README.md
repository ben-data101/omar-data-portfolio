# --------------------------
# Omar's Data Analysis Portfolio
# --------------------------
# This script showcases data analysis projects using Python libraries:
# Pandas for data manipulation,
# Matplotlib for plotting,
# Seaborn for advanced visualization,
# Scikit-Learn for machine learning.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------
# Project 1: Personal Table
# --------------------------
print("\n🔹 Project 1: My Profile")

data = {
    "Name": ["Omar"],
    "Country": ["Nigeria"],
    "Best Food": ["Jollof Rice"]
}

df_profile = pd.DataFrame(data)
print(df_profile)

# --------------------------
# Project 2: Height & Weight Analysis
# --------------------------
print("\n🔹 Project 2: Height & Weight Data")

# Load height and weight data from URL
url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df_hw = pd.read_csv(url)

# Rename columns for clarity
df_hw.columns = ["Index", "Height (in)", "Weight (lbs)"]

# Drop rows with missing data
df_hw.dropna(inplace=True)

# Plot height vs weight scatter plot
plt.scatter(df_hw["Height (in)"], df_hw["Weight (lbs)"], color="blue")
plt.title("Height vs Weight")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.grid(True)
plt.savefig("height_vs_weight.png")  # Save plot as image file
plt.close()

# --------------------------
# Project 3: Sales Analysis
# --------------------------
print("\n🔹 Project 3: Sales Data Analysis")

# Sample sales data dictionary
data = {
    "OrderID": [1001, 1002, 1003, 1004, 1005, 1006],
    "CustomerName": ["Omar", "Amina", "Tunde", "Omar", "Sara", "Tunde"],
    "ProductCategory": ["Electronics", "Books", "Clothing", "Books", "Electronics", "Electronics"],
    "Quantity": [2, 5, 3, 1, 1, 1],
    "PricePerUnit": [299.99, 9.99, 19.99, 9.99, 199.99, 299.99],
    "OrderDate": ["2024-06-01", "2024-06-02", "2024-06-02", "2024-06-03", "2024-06-04", "2024-06-05"]
}

df_sales = pd.DataFrame(data)

# Calculate total price per order
df_sales["TotalPrice"] = df_sales["Quantity"] * df_sales["PricePerUnit"]

# Save cleaned sales data to CSV
df_sales.to_csv("cleaned_sales_data.csv", index=False)

# Summarize total sales by product category
sales_summary = df_sales.groupby("ProductCategory")["TotalPrice"].sum().reset_index()

# Plot total sales per product category as a bar chart
plt.bar(sales_summary["ProductCategory"], sales_summary["TotalPrice"], color="skyblue")
plt.title("Total Sales per Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales ($)")
plt.savefig("total_sales_chart.png")  # Save bar chart as image file
plt.close()

# --------------------------
# Project 4: Machine Learning
# --------------------------
print("\n🔹 Project 4: Machine Learning")

# Prepare features and target for regression
X = df_sales[["Quantity", "PricePerUnit"]]
y = df_sales["TotalPrice"]

# Train a linear regression model to predict total price
model = LinearRegression()
model.fit(X, y)

# Predict total price for an order with 4 items costing $50 each
prediction = model.predict([[4, 50]])
print("🔸 Predicted Total Price (4 items, $50 each):", round(prediction[0], 2))

# Prepare data for classification (predicting product category)
label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(df_sales["ProductCategory"])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict categories for test data
y_pred = clf.predict(X_test)

# Calculate and print model accuracy
acc = accuracy_score(y_test, y_pred)
print("🔸 Classification Accuracy:", round(acc * 100, 2), "%")

# Predict product category for new input
new_pred = clf.predict([[3, 250]])
predicted_category = label_encoder.inverse_transform(new_pred)
print("🔸 Predicted Product Category:", predicted_category[0])
 
