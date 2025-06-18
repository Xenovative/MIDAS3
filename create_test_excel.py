import pandas as pd

# Create sample data for the first sheet
products = pd.DataFrame({
    'product_id': [101, 102, 103, 104],
    'name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'price': [1200.00, 25.50, 45.00, 199.99],
    'stock': [15, 100, 75, 30]
})

# Create sample data for the second sheet
customers = pd.DataFrame({
    'customer_id': [1001, 1002, 1003],
    'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
    'join_date': ['2023-01-15', '2023-02-20', '2023-03-10']
})

# Create sample data for the third sheet
orders = pd.DataFrame({
    'order_id': [5001, 5002, 5003, 5004],
    'customer_id': [1001, 1002, 1001, 1003],
    'product_id': [101, 103, 102, 104],
    'quantity': [1, 2, 1, 1],
    'order_date': ['2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04']
})

# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter('test_files/test_data.xlsx', engine='openpyxl') as writer:
    # Write each dataframe to a different worksheet
    products.to_excel(writer, sheet_name='Products', index=False)
    customers.to_excel(writer, sheet_name='Customers', index=False)
    orders.to_excel(writer, sheet_name='Orders', index=False)

print("Test Excel file created successfully at test_files/test_data.xlsx")
