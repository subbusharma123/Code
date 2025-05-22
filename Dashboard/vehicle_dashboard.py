import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import os
import requests
from PIL import Image
import scipy.stats as stats

# Set page configuration for a wide layout to ensure the dashboard looks good
st.set_page_config(page_title="Ultimate Vehicle Dashboard", layout="wide")

# Define base directory and image storage path for storing downloaded images
base_dir = r"C:\Users\subra\Documents\GitHub\Code"
images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)  # Create images directory if it doesn't exist

# Helper function to download images silently with error handling
def download_image(url, save_path):
    """
    Downloads an image from a given URL and saves it to the specified path.
    This function operates silently without displaying any messages.
    Args:
        url (str): The URL of the image to download.
        save_path (str): The local path where the image will be saved.
    Returns:
        tuple: (bool, str or None) indicating success and an error message if failed.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True, None
    except Exception as e:
        return False, str(e)

# Helper function to load images with error handling
def load_image(image_path, caption):
    """
    Loads an image from a local path and displays it in Streamlit.
    If the image fails to load, a placeholder is displayed.
    Args:
        image_path (str): Path to the local image file.
        caption (str): Caption to display below the image.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
        img = Image.open(image_path)
        st.image(img, caption=caption, use_container_width=True, cls="car-image")
    except Exception as e:
        st.warning(f"Failed to load image for {caption}: {str(e)}")
        st.markdown(f"**{caption}** (Image not available)")

# Updated image URLs from Pexels (verified car images, focusing on 1980s styles where possible)
car_image_urls = {
    'dodge': 'https://images.pexels.com/photos/1545743/pexels-photo-1545743.jpeg',  # Classic Dodge-like car
    'honda': 'https://images.pexels.com/photos/210182/pexels-photo-210182.jpeg',  # 1980s Honda style
    'isuzu': 'https://images.pexels.com/photos/3954427/pexels-photo-3954427.jpeg',  # Isuzu pickup truck
    'jaguar': 'https://images.pexels.com/photos/2127733/pexels-photo-2127733.jpeg',  # Jaguar
    'mazda': 'https://images.pexels.com/photos/3786091/pexels-photo-3786091.jpeg',  # Mazda RX-7 style
    'mercedes-benz': 'https://images.pexels.com/photos/120049/pexels-photo-120049.jpeg',  # Mercedes-Benz
    'mercury': 'https://images.pexels.com/photos/164634/pexels-photo-164634.jpeg',  # Mercury Cougar style
    'mitsubishi': 'https://images.pexels.com/photos/244553/pexels-photo-244553.jpeg',  # Mitsubishi
    'nissan': 'https://images.pexels.com/photos/248747/pexels-photo-248747.jpeg',  # Nissan 300ZX style
    'peugeot': 'https://images.pexels.com/photos/3729464/pexels-photo-3729464.jpeg',  # Peugeot
    'plymouth': 'https://images.pexels.com/photos/1545743/pexels-photo-1545743.jpeg',  # Classic Plymouth-like car
    'porsche': 'https://images.pexels.com/photos/112460/pexels-photo-112460.jpeg',  # Porsche 911
    'renault': 'https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg',  # Renault (generic 1980s car)
    'saab': 'https://images.pexels.com/photos/1592384/pexels-photo-1592384.jpeg',  # Saab (classic car)
    'subaru': 'https://images.pexels.com/photos/385003/pexels-photo-385003.jpeg',  # Subaru
    'toyota': 'https://images.pexels.com/photos/210182/pexels-photo-210182.jpeg',  # Toyota Supra style
    'volkswagen': 'https://images.pexels.com/photos/1637859/pexels-photo-1637859.jpeg',  # Volkswagen Beetle
    'volvo': 'https://images.pexels.com/photos/397857/pexels-photo-397857.jpeg',  # Volvo 240 style
}

# Silently download car images without displaying any messages
with st.spinner("Initializing dashboard... Please wait."):
    car_images = {}
    for make, url in car_image_urls.items():
        image_path = os.path.join(images_dir, f"{make.replace('-', '_')}.jpg")
        if not os.path.exists(image_path):
            success, _ = download_image(url, image_path)
            car_images[make] = image_path if success else None
        else:
            car_images[make] = image_path

# Custom CSS to set a Porsche background image directly via URL
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/112460/pexels-photo-112460.jpeg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-color: #333; /* Fallback color if image fails */
        color: white;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp div {
        color: white !important;
        text-shadow: 1px 1px 2px black;
    }
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: white !important;
        text-shadow: 1px 1px 2px black;
    }
    .car-image {
        border: 2px solid white;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and introduction with a car emoji and detailed description
st.title("🏎️ Ultimate Vehicle Data Dashboard (1980s)")
st.markdown("""
Welcome to the Ultimate Vehicle Dashboard! This interactive dashboard lets you explore detailed vehicle data from the 1980s, 
covering various makes, body styles, fuel types, and performance metrics. Use the filters to dive deep into the data, 
check out stunning visualizations, and enjoy images of iconic cars from the era!  
- **Visualizations**: Explore trends with bar charts, scatter plots, and more.  
- **Car Gallery**: View images of cars from the 1980s.  
- **Advanced Analysis**: Dive into correlations and detailed statistics.  
- **Top Cars**: See the most expensive and fuel-efficient vehicles.  
- **Compare Cars**: Select two cars for a side-by-side comparison of their specs.
""")

# Helper function to load and clean data with detailed cleaning steps
@st.cache_data
def load_data():
    """
    Loads and cleans the car_data.csv file with comprehensive data preprocessing.
    Steps:
    1. Load the CSV file from the specified path.
    2. Replace missing value indicators ('?') with NaN.
    3. Convert relevant columns to numeric types.
    4. Impute missing values for price, horsepower, engine_size, city_mpg, and highway_mpg.
    Returns:
        pandas.DataFrame or None if loading fails.
    """
    csv_path = os.path.join(base_dir, "car_data.csv")
    if not os.path.exists(csv_path):
        st.error(f"Error: The file {csv_path} was not found. Please ensure it exists in the specified directory.")
        return None
    try:
        # Define expected column names for the car dataset
        expected_columns = [
            'make', 'fuel_type', 'body_style', 'wheel_base', 'length', 'width', 'height',
            'curb_weight', 'engine_size', 'compression_ratio', 'horsepower', 'peak_rpm',
            'city_mpg', 'highway_mpg', 'price'
        ]
        
        # Load the CSV with specified column names to fix incorrect headers
        df = pd.read_csv(csv_path, names=expected_columns, header=0)
        
        # Step 1: Replace '?' with NaN for missing values
        df.replace('?', np.nan, inplace=True)
        
        # Step 2: Convert price to numeric and impute missing prices with mean by body style
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price'] = df.groupby('body_style')['price'].transform(lambda x: x.fillna(x.mean()))
        
        # Step 3: Convert other numeric columns to appropriate types
        numeric_cols = ['wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_size', 
                        'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Step 4: Impute missing values in horsepower, engine_size, city_mpg, and highway_mpg with their means
        for col in ['horsepower', 'engine_size', 'city_mpg', 'highway_mpg']:
            df[col] = df[col].fillna(df[col].mean())
        
        # Step 5: Ensure categorical columns are strings
        categorical_cols = ['make', 'fuel_type', 'body_style']
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

# Proceed only if data is successfully loaded
if df is not None:
    # Summary statistics section with detailed metrics
    st.subheader("📊 Data Summary")
    total_vehicles = len(df)
    avg_price = df['price'].mean()
    most_common_make = df['make'].mode()[0]
    highest_price = df['price'].max()
    lowest_price = df['price'].min()
    avg_horsepower = df['horsepower'].mean()
    avg_mpg = df['city_mpg'].mean()
    st.markdown(f"""
    **Overview of the Dataset**  
    - **Total Vehicles**: {total_vehicles}  
    - **Average Price**: ${avg_price:,.2f}  
    - **Most Common Make**: {most_common_make}  
    - **Highest Price**: ${highest_price:,.2f}  
    - **Lowest Price**: ${lowest_price:,.2f}  
    - **Average Horsepower**: {avg_horsepower:.2f} HP  
    - **Average City MPG**: {avg_mpg:.2f} MPG
    """)

    # Filters section with multiple interactive elements
    st.subheader("🔎 Filter Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        makes = ['All'] + sorted(df['make'].unique())
        selected_make = st.selectbox("Select Make", makes)
    with col2:
        fuel_types = ['All'] + sorted(df['fuel_type'].unique())
        selected_fuel = st.selectbox("Select Fuel Type", fuel_types)
    with col3:
        body_styles = ['All'] + sorted(df['body_style'].unique())
        selected_styles = st.multiselect("Select Body Styles", body_styles, default=['All'])
    
    # Price range slider with clear labels
    min_price, max_price = int(df['price'].min()), int(df['price'].max())
    price_range = st.slider("Select Price Range ($)", min_price, max_price, (min_price, max_price))

    # Apply filters to the DataFrame
    filtered_df = df.copy()
    if selected_make != 'All':
        filtered_df = filtered_df[filtered_df['make'] == selected_make]
    if selected_fuel != 'All':
        filtered_df = filtered_df[filtered_df['fuel_type'] == selected_fuel]
    if 'All' not in selected_styles:
        filtered_df = filtered_df[filtered_df['body_style'].isin(selected_styles)]
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]

    # Display filtered data in a table with styling
    st.subheader("📋 Filtered Vehicle Data")
    # Convert all columns to appropriate types to avoid ArrowTypeError
    display_df = filtered_df.copy()
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            display_df[col] = display_df[col].astype(str)
        elif display_df[col].dtype in [np.float64, np.int64]:
            display_df[col] = display_df[col].astype(float)
    st.dataframe(display_df, use_container_width=True)

    # Tabs for different sections of the dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualizations", "Car Gallery", "Advanced Analysis", "Top Cars", "Compare Cars"])

    # Tab 1: Visualizations with additional charts
    with tab1:
        st.subheader("📈 Visualizations")
        st.markdown("Explore various trends and relationships in the vehicle data through interactive charts.")

        col4, col5 = st.columns(2)

        # Bar chart: Average price by body style
        with col4:
            avg_price_by_style = filtered_df.groupby('body_style')['price'].mean().reset_index()
            fig_bar = px.bar(avg_price_by_style, x='body_style', y='price',
                             title="Average Price by Body Style",
                             labels={'price': 'Average Price ($)', 'body_style': 'Body Style'},
                             color='body_style')
            st.plotly_chart(fig_bar, use_container_width=True)

        # Pie chart: Distribution of body styles
        with col5:
            body_style_counts = filtered_df['body_style'].value_counts().reset_index()
            body_style_counts.columns = ['body_style', 'count']
            fig_pie = px.pie(body_style_counts, names='body_style', values='count',
                             title="Distribution of Body Styles")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Scatter plot: Horsepower vs Price
        st.subheader("Horsepower vs Price")
        scatter_df = filtered_df.copy()

# Ensure correct dtypes and drop NaNs
        required_columns = ['horsepower', 'price', 'engine_size', 'fuel_type']
        scatter_df = scatter_df.dropna(subset=required_columns)
        for col in ['horsepower', 'price', 'engine_size']:
            scatter_df[col] = pd.to_numeric(scatter_df[col], errors='coerce')
        scatter_df = scatter_df.dropna(subset=['horsepower', 'price', 'engine_size'])

        fig_scatter = px.scatter(scatter_df, x='horsepower', y='price',
                         color='fuel_type', size='engine_size',
                         hover_data=['make', 'body_style'],
                         title="Horsepower vs Price by Fuel Type",
                         labels={'horsepower': 'Horsepower', 'price': 'Price ($)'})
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Box plot: Price distribution by make (top 5 makes by count)
        top_makes = filtered_df['make'].value_counts().head(5).index
        box_df = filtered_df[filtered_df['make'].isin(top_makes)]
        fig_box = px.box(box_df, x='make', y='price',
                         title="Price Distribution by Top Makes",
                         labels={'price': 'Price ($)', 'make': 'Make'})
        st.plotly_chart(fig_box, use_container_width=True)

        # Line chart: City vs Highway MPG by body style
        mpg_df = filtered_df.groupby('body_style')[['city_mpg', 'highway_mpg']].mean().reset_index()
        mpg_df = mpg_df.melt(id_vars='body_style', value_vars=['city_mpg', 'highway_mpg'],
                             var_name='MPG_Type', value_name='MPG')
        fig_line = px.line(mpg_df, x='body_style', y='MPG', color='MPG_Type',
                           title="City vs Highway MPG by Body Style",
                           labels={'MPG': 'Miles Per Gallon', 'body_style': 'Body Style'})
        st.plotly_chart(fig_line, use_container_width=True)

        # Histogram: Price distribution
        st.subheader("Price Distribution")
        fig_hist = px.histogram(filtered_df, x='price', nbins=30,
                                title="Distribution of Vehicle Prices",
                                labels={'price': 'Price ($)'})
        st.plotly_chart(fig_hist, use_container_width=True)

        # Radar Chart: Performance Metrics by Make (Top 5 Makes)
        st.subheader("Performance Metrics Radar Chart (Top 5 Makes)")
        radar_df = filtered_df[filtered_df['make'].isin(top_makes)]
        radar_metrics = radar_df.groupby('make')[['horsepower', 'engine_size', 'city_mpg', 'highway_mpg']].mean().reset_index()
        fig_radar = go.Figure()
        for _, row in radar_metrics.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['horsepower'], row['engine_size'], row['city_mpg'], row['highway_mpg'], row['horsepower']],
                theta=['Horsepower', 'Engine Size', 'City MPG', 'Highway MPG', 'Horsepower'],
                name=row['make'],
                fill='toself'
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Performance Metrics by Make",
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Tab 2: Car Gallery with detailed display
    with tab2:
        st.subheader("🖼️ Car Gallery")
        st.markdown("Explore images of cars from the makes in the dataset. These images represent iconic vehicles from the 1980s.")

        # Filter car images based on selected make
        if selected_make != 'All':
            display_makes = [selected_make]
        else:
            display_makes = list(car_images.keys())

        # Debugging: Show which makes are being processed
        st.markdown(f"**Displaying images for the following makes:** {', '.join(display_makes)}")

        # Display car images in a grid with 3 columns
        cols = st.columns(3)
        for idx, make in enumerate(display_makes):
            with cols[idx % 3]:
                st.markdown(f"**{make.capitalize()}**")
                image_path = car_images.get(make)
                if image_path:
                    load_image(image_path, make.capitalize())
                else:
                    st.markdown(f"**{make.capitalize()}** (Image not available)")

    # Tab 3: Advanced Analysis with expanded statistics
    with tab3:
        st.subheader("🔍 Advanced Analysis")
        st.markdown("Dive deeper into the data with correlation analysis and detailed statistical insights.")

        # Correlation matrix heatmap
        st.subheader("Correlation Matrix")
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig_corr = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                annotation_text=corr_matrix.round(2).values,
                showscale=True
            )
            fig_corr.update_layout(title="Correlation Matrix of Numeric Features")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No numeric data available for correlation matrix.")

        # Scatter plot: Engine Size vs City MPG
        st.subheader("Engine Size vs City MPG")
        scatter_df = filtered_df.dropna(subset=['engine_size', 'city_mpg', 'horsepower'])
        if not scatter_df.empty:
            fig_scatter2 = px.scatter(scatter_df, x='engine_size', y='city_mpg',
                                      color='fuel_type', size='horsepower',
                                      hover_data=['make', 'body_style'],
                                      title="Engine Size vs City MPG",
                                      labels={'engine_size': 'Engine Size (cc)', 'city_mpg': 'City MPG'})
            st.plotly_chart(fig_scatter2, use_container_width=True)
        else:
            st.warning("Not enough data to display Engine Size vs City MPG scatter plot.")

        # Expander for detailed statistics with additional metrics
        with st.expander("Detailed Statistics"):
            st.markdown("### Detailed Statistical Analysis")
            st.write("#### Mean Values")
            st.write(filtered_df.describe().loc['mean'])
            st.markdown("#### Median Values")
            st.write(filtered_df.describe().loc['50%'])
            st.markdown("#### Standard Deviation")
            st.write(filtered_df.describe().loc['std'])
            st.markdown("#### Skewness (Measure of Asymmetry)")
            skewness = filtered_df.select_dtypes(include=[np.number]).skew()
            st.write(skewness)
            st.markdown("#### Kurtosis (Measure of Tailedness)")
            kurtosis = filtered_df.select_dtypes(include=[np.number]).kurtosis()
            st.write(kurtosis)

    # Tab 4: Top Cars with detailed rankings
    with tab4:
        st.subheader("🏆 Top Cars")
        st.markdown("Discover the top-performing cars based on price and fuel efficiency.")

        # Top 5 most expensive cars
        st.markdown("### Top 5 Most Expensive Cars")
        top_expensive = filtered_df.nlargest(5, 'price')[['make', 'body_style', 'price', 'horsepower', 'fuel_type']]
        # Convert types for display
        top_expensive_display = top_expensive.copy()
        for col in top_expensive_display.columns:
            if top_expensive_display[col].dtype == 'object':
                top_expensive_display[col] = top_expensive_display[col].astype(str)
            elif top_expensive_display[col].dtype in [np.float64, np.int64]:
                top_expensive_display[col] = top_expensive_display[col].astype(float)
        st.dataframe(top_expensive_display, use_container_width=True)
        
        cols = st.columns(5)

        # Loop using enumerate to get integer index
        for count, (_, row) in enumerate(top_expensive.iterrows()):
            make = row['make']

            with cols[count % 5]:
                image_path = car_images.get(make)

                if image_path:
                    load_image(
                        image_path,
                        f"{make.capitalize()} (${row['price']:,.2f})"
                    )
                else:
                    st.markdown(
                        f"**{make.capitalize()} (${row['price']:,.2f})**\n\n_Image not available_"
                    )




        # Top 5 most fuel-efficient cars (by city MPG)
        st.markdown("### Top 5 Most Fuel-Efficient Cars (City MPG)")
        top_efficient = filtered_df.nlargest(5, 'city_mpg')[['make', 'body_style', 'city_mpg', 'highway_mpg', 'fuel_type']]
        # Convert types for display
        top_efficient_display = top_efficient.copy()
        for col in top_efficient_display.columns:
            if top_efficient_display[col].dtype == 'object':
                top_efficient_display[col] = top_efficient_display[col].astype(str)
            elif top_efficient_display[col].dtype in [np.float64, np.int64]:
                top_efficient_display[col] = top_efficient_display[col].astype(float)
        st.dataframe(top_efficient_display, use_container_width=True)

        # Display images for top fuel-efficient cars
        cols = st.columns(5)

        # Defensive check: ensure cols is a list
        if not isinstance(cols, list):
            st.error("Failed to create columns.")
        else:
            for idx, row in top_efficient.iterrows():
                make = row.get("make", "Unknown")
                city_mpg = row.get("city_mpg", "N/A")

                col = cols[idx % len(cols)]  # Use len(cols) to be safe

                with col:
                    image_path = car_images.get(make)
                    if image_path:
                        load_image(image_path, f"{make.capitalize()} ({city_mpg} MPG)")
                    else:
                        st.markdown(f"**{make.capitalize()} ({city_mpg} MPG)**\n\n_Image not available_")


    # Tab 5: Compare Cars (New Feature)
    with tab5:
        st.subheader("🔄 Compare Two Cars")
        st.markdown("Select two cars to compare their specifications side-by-side.")

        # Select two cars for comparison
        car1, car2 = st.columns(2)
        with car1:
            car1_selection = st.selectbox("Select First Car", filtered_df.index, format_func=lambda x: f"{filtered_df.loc[x, 'make']} {filtered_df.loc[x, 'body_style']}")
        with car2:
            car2_selection = st.selectbox("Select Second Car", filtered_df.index, format_func=lambda x: f"{filtered_df.loc[x, 'make']} {filtered_df.loc[x, 'body_style']}", index=1 if len(filtered_df) > 1 else 0)

        # Display comparison
        if car1_selection != car2_selection:
            car1_data = filtered_df.loc[car1_selection]
            car2_data = filtered_df.loc[car2_selection]
            comparison_df = pd.DataFrame({
                'Metric': ['Make', 'Body Style', 'Price ($)', 'Horsepower', 'Engine Size (cc)', 'City MPG', 'Highway MPG'],
                f"{car1_data['make']} {car1_data['body_style']}": [
                    car1_data['make'], car1_data['body_style'], f"{car1_data['price']:,.2f}",
                    car1_data['horsepower'], car1_data['engine_size'], car1_data['city_mpg'], car1_data['highway_mpg']
                ],
                f"{car2_data['make']} {car2_data['body_style']}": [
                    car2_data['make'], car2_data['body_style'], f"{car2_data['price']:,.2f}",
                    car2_data['horsepower'], car2_data['engine_size'], car2_data['city_mpg'], car2_data['highway_mpg']
                ]
            })
            # Convert types for display
            for col in comparison_df.columns:
                if comparison_df[col].dtype == 'object':
                    comparison_df[col] = comparison_df[col].astype(str)
            st.table(comparison_df.set_index('Metric'))

            # Display images of the compared cars
            col_compare1, col_compare2 = st.columns(2)
            with col_compare1:
                image_path = car_images.get(car1_data['make'])
                if image_path:
                    load_image(image_path, f"{car1_data['make'].capitalize()} {car1_data['body_style']}")
                else:
                    st.markdown(f"**{car1_data['make'].capitalize()} {car1_data['body_style']}** (Image not available)")
            with col_compare2:
                image_path = car_images.get(car2_data['make'])
                if image_path:
                    load_image(image_path, f"{car2_data['make'].capitalize()} {car2_data['body_style']}")
                else:
                    st.markdown(f"**{car2_data['make'].capitalize()} {car2_data['body_style']}** (Image not available)")
        else:
            st.warning("Please select two different cars to compare.")

    # Additional section: Car images alongside visualizations
    st.subheader("🚗 Visualizations with Car Images")
    st.markdown("Visualize data alongside images of iconic cars from the dataset.")
    col6, col7 = st.columns([2, 1])
    with col6:
        # Scatter plot: Weight vs Price
        fig_scatter3 = px.scatter(filtered_df, x='curb_weight', y='price',
                                  color='body_style', size='engine_size',
                                  hover_data=['make', 'fuel_type'],
                                  title="Curb Weight vs Price",
                                  labels={'curb_weight': 'Curb Weight (lbs)', 'price': 'Price ($)'})
        st.plotly_chart(fig_scatter3, use_container_width=True)
    with col7:
        # Display a car image (e.g., Porsche)
        image_path = car_images.get('porsche')
        if image_path:
            load_image(image_path, "Porsche")
        else:
            st.markdown("**Porsche** (Image not available)")

    # Section: Car Performance Metrics with detailed analysis
    st.subheader("🏁 Car Performance Metrics")
    st.markdown("Analyze the performance of cars across different makes.")
    col8, col9 = st.columns(2)
    with col8:
        # Bar chart: Average horsepower by make
        avg_hp_by_make = filtered_df.groupby('make')['horsepower'].mean().reset_index()
        fig_hp = px.bar(avg_hp_by_make, x='make', y='horsepower',
                        title="Average Horsepower by Make",
                        labels={'horsepower': 'Average Horsepower', 'make': 'Make'},
                        color='make')
        st.plotly_chart(fig_hp, use_container_width=True)
    with col9:
        # Display a car image (e.g., Jaguar)
        image_path = car_images.get('jaguar')
        if image_path:
            load_image(image_path, "Jaguar")
        else:
            st.markdown("**Jaguar** (Image not available)")

    # Section: Fuel Efficiency Analysis with additional insights
    st.subheader("⛽ Fuel Efficiency Analysis")
    st.markdown("Compare fuel efficiency metrics across different vehicles.")
    col10, col11 = st.columns(2)
    with col10:
        # Scatter plot: City MPG vs Highway MPG
        fig_mpg_scatter = px.scatter(filtered_df, x='city_mpg', y='highway_mpg',
                                     color='fuel_type', size='engine_size',
                                     hover_data=['make', 'body_style'],
                                     title="City MPG vs Highway MPG",
                                     labels={'city_mpg': 'City MPG', 'highway_mpg': 'Highway MPG'})
        st.plotly_chart(fig_mpg_scatter, use_container_width=True)
    with col11:
        # Display a car image (e.g., Honda)
        image_path = car_images.get('honda')
        if image_path:
            load_image(image_path, "Honda")
        else:
            st.markdown("**Honda** (Image not available)")

    # Footer with credits and additional information
    st.markdown("---")
    st.markdown("""
    **Data Source**: Car dataset with 1980s vehicle specifications.  
    **Images**: Sourced from Pexels and stored locally.  
    **Developed by**: Subramanya Sharma, Data Scientist.  
    **GitHub**: [Subramanya Sharma]  
    **Note**: Images are representative and may not exactly match the specific models in the dataset but aim to reflect the style of 1980s vehicles.
    """)

else:
    st.warning("Please address the data loading issue to view the dashboard.")