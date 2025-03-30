#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:





# In[6]:


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset
data = {
    'Product_Category': ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Electronics', 'Clothing', 'Home & Kitchen', 'Books'],
    'Customer_Location': ['Urban', 'Suburban', 'Rural', 'Urban', 'Rural', 'Suburban', 'Urban', 'Suburban'],
    'Shipping_Method': ['Standard', 'Express', 'Same-day', 'Standard', 'Express', 'Same-day', 'Standard', 'Express'],
    'Delivery_Time': [48, 24, 12, 72, 36, 18, 60, 30]
}
df = pd.DataFrame(data)

# Split data
X = df[['Product_Category', 'Customer_Location', 'Shipping_Method']]
y = df['Delivery_Time']

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Product_Category', 'Customer_Location', 'Shipping_Method'])
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('delivery_time_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Load the trained model
with open('delivery_time_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the Streamlit app
st.title('Delivery Time Prediction App')
st.write('Enter order details to predict expected delivery time.')

# Input form with three fields
with st.form("order_form"):
    product_category = st.selectbox('Product Category', ['Electronics', 'Clothing', 'Home & Kitchen', 'Books'])
    customer_location = st.selectbox('Customer Location', ['Urban', 'Suburban', 'Rural'])
    shipping_method = st.selectbox('Shipping Method', ['Standard', 'Express', 'Same-day'])
    
    # Submit button
    submit_button = st.form_submit_button("Predict Delivery Time")

# Process user input and make prediction
if submit_button:
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Product_Category': [product_category],
        'Customer_Location': [customer_location],
        'Shipping_Method': [shipping_method]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    st.success(f'Expected delivery time: {round(prediction[0], 2)} hours')


# In[ ]:




