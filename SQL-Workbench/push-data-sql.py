import pandas as pd
import mysql.connector
from mysql.connector import Error

# Path to your CSV file
csv_file_path = r"C:\Users\subra\Documents\GitHub\Code\Dashboard\car_data.csv"

# Read the CSV file into a DataFrame, explicitly setting the first column as 'id'
df = pd.read_csv(csv_file_path)
# Rename the first column (which is unnamed in the CSV) to 'id'
df.columns = ['id'] + list(df.columns[1:])

# Initialize connection and cursor as None
connection = None
cursor = None

try:
    # Connect to MySQL
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",  # Replace with your MySQL root password
        database="subbu"
    )
    
    if connection.is_connected():
        print("Connected to MySQL database")
        cursor = connection.cursor()

        # SQL INSERT statement for the cars table
        insert_query = """
        INSERT INTO cars (
            id, make, fuel_type, aspiration, num_of_doors, body_style, drive_wheels, 
            engine_location, wheel_base, length, width, height, curb_weight, engine_type, 
            num_of_cylinders, engine_size, fuel_system, compression_ratio, horsepower, 
            peak_rpm, city_mpg, highway_mpg, price
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Convert DataFrame rows to tuples for insertion
        for _, row in df.iterrows():
            values = (
                row['id'], row['make'], row['fuel_type'], row['aspiration'], row['num_of_doors'], 
                row['body_style'], row['drive_wheels'], row['engine_location'], row['wheel_base'], 
                row['length'], row['width'], row['height'], row['curb_weight'], row['engine_type'], 
                row['num_of_cylinders'], row['engine_size'], row['fuel_system'], 
                row['compression_ratio'], row['horsepower'], row['peak_rpm'], row['city_mpg'], 
                row['highway_mpg'], row['price']
            )
            cursor.execute(insert_query, values)
        
        # Commit the transaction
        connection.commit()
        print(f"Successfully inserted {cursor.rowcount} rows into the cars table")

except Error as e:
    print(f"Error: {e}")

finally:
    # Check if cursor and connection exist before closing
    if cursor is not None:
        cursor.close()
    if connection is not None and connection.is_connected():
        connection.close()
        print("MySQL connection closed")