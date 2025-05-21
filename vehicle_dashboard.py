import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="Vehicle Sales Dashboard", layout="wide")

# Title and introduction
st.title("🚗 Vehicle Sales Dashboard")
st.markdown("""
This dashboard showcases vehicle sales data, including types of vehicles and their sales figures for 2024.
Explore the visualizations and images below to understand market trends!
""")

# Sample vehicle data
data = {
    'Vehicle_Type': ['Sedan', 'SUV', 'Truck', 'Electric', 'Hatchback', 'Minivan'],
    'Sales_2024': [150000, 220000, 180000, 90000, 70000, 50000],
    'Market_Share': [0.25, 0.35, 0.20, 0.10, 0.05, 0.05],
    'Average_Price': [25000, 35000, 40000, 50000, 20000, 30000]
}
df = pd.DataFrame(data)

# Display raw data
st.subheader("📊 Vehicle Sales Data")
st.dataframe(df)

# Layout columns for visualizations
col1, col2 = st.columns(2)

# Bar chart for sales
with col1:
    st.subheader("Sales by Vehicle Type")
    fig_bar = px.bar(df, x='Vehicle_Type', y='Sales_2024', 
                     color='Vehicle_Type', 
                     title="Vehicle Sales in 2024",
                     labels={'Sales_2024': 'Number of Vehicles Sold'})
    st.plotly_chart(fig_bar, use_container_width=True)

# Pie chart for market share
with col2:
    st.subheader("Market Share by Vehicle Type")
    fig_pie = px.pie(df, names='Vehicle_Type', values='Market_Share',
                     title="Market Share Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

# Price vs Sales scatter plot
st.subheader("Price vs Sales Analysis")
fig_scatter = px.scatter(df, x='Average_Price', y='Sales_2024', 
                         color='Vehicle_Type', size='Sales_2024',
                         hover_data=['Vehicle_Type'],
                         title="Average Price vs Sales Volume")
st.plotly_chart(fig_scatter, use_container_width=True)

# Vehicle images section
st.subheader("🖼️ Vehicle Gallery")
st.markdown("Sample images of different vehicle types:")

# Placeholder image URLs (from Unsplash, replace with desired URLs or local paths)
image_data = {
    'Vehicle_Type': ['Sedan', 'SUV', 'Truck', 'Electric'],
    'Image_URL': [
        'https://images.unsplash.com/photo-1503376780353-7e6692767b70',  # Sedan
        'https://images.unsplash.com/photo-1504214208698-ea1916a2195a',  # SUV
        'https://images.unsplash.com/photo-1502744688674-c619d0586c07',  # Truck
        'https://images.unsplash.com/photo-1560958089-b8a1929cea89'   # Electric
    ]
}
img_df = pd.DataFrame(image_data)

# Display images in a grid
cols = st.columns(4)
for idx, row in img_df.iterrows():
    with cols[idx % 4]:
        st.image(row['Image_URL'], caption=row['Vehicle_Type'], use_column_width=True)

# Footer
st.markdown("---")
st.markdown("Data source: Simulated vehicle sales data for 2024. Images from Unsplash.")