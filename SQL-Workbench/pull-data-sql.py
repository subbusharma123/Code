import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px

# Streamlit page configuration
st.set_page_config(page_title="Car Data Dashboard", layout="wide")

# Function to fetch data from MySQL
def fetch_data():
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1234",  # Replace with your MySQL root password
            database="subbu"
        )
        query = "SELECT * FROM cars;"
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# Fetch the data
st.title("Car Data Dashboard")
st.write("Data fetched from MySQL `subbu.cars` table")
data = fetch_data()

if data is not None:
    # Display the raw data in a table
    st.subheader("Raw Data")
    st.dataframe(data)

    # Bar chart: Average price by make
    st.subheader("Average Price by Make")
    avg_price_by_make = data.groupby('make')['price'].mean().reset_index()
    fig_bar = px.bar(avg_price_by_make, x='make', y='price', title="Average Price by Make",
                     labels={'price': 'Average Price', 'make': 'Car Make'},
                     color='price', color_continuous_scale='Viridis')
    fig_bar.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter plot: Horsepower vs Price
    st.subheader("Horsepower vs Price")
    fig_scatter = px.scatter(data, x='horsepower', y='price', color='make',
                             title="Horsepower vs Price",
                             labels={'horsepower': 'Horsepower', 'price': 'Price'},
                             hover_data=['make', 'body_style'])
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.write("No data to display. Check your MySQL connection or data.")