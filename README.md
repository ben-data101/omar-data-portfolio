# omar_portfolio.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# Project 1: Personal Profile
print("🔹 Omar's Personal Info")
data = {'Name': ['Omar'], 'Country': ['Nigeria'], 'Best Food': ['Jollof Rice']}
df_profile = pd.DataFrame(data)
print(df_profile)

# Project 2: Height vs Weight
print("\n🔹 Height & Weight Analysis")
data_hw = {
    'Name': ['Ayo', 'Zara', 'Uche', 'Musa', 'Lola'],
    'Height_cm': [160, 165, 170, 175, 180],
    'Weight_kg': [55, 60, 65, 70, 75]
}
df_hw = pd.DataFrame(data_hw)

# Save scatter plot
plt.figure()
plt.scatter(df_hw['Height_cm'], df_hw['Weight_kg'], color='blue')
plt.title('Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.savefig('height_vs_weight.png')

# Project 3: Sales Data
print("\n🔹 Sales Data Analysis")
sales_data = {
    'OrderID': [1, 2, 3, 4],
    'Product': ['Rice', 'Oil', 'Bread', 'Sugar'],
    'Price': [2500, 3000, 800, 1200],
    'Quantity': [2, 1, 5, 3]
}
df_sales = pd.DataFrame(sales_data)

# Add total column
df_sales['Total'] = df_sales['Price'] * df_sales['Quantity']
print(df_sales)

# Save cleaned CSV
df_sales.to_csv('cleaned_sales_data.csv', index=False)

# Save bar chart
plt.figure()
plt.bar(df_sales['Product'], df_sales['Total'], color='green')
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total')
plt.savefig('total_sales_chart.png')

# Project 4: Linear Regression
print("\n🔹 Predicting Sales (ML)")
model = LinearRegression()
X = df_sales[['Price']]
y = df_sales['Total']
model.fit(X, y)
predicted = model.predict(X)
print("Predicted Totals:", predicted)

# Project 5: Decision Tree
print("\n🔹 Classifying Products (ML)")
clf = DecisionTreeClassifier()
X_tree = df_sales[['Price', 'Quantity']]
y_tree = df_sales['Product']
clf.fit(X_tree, y_tree)
print("Predicted Product:", clf.predict([[2500, 2]]))  # Example
