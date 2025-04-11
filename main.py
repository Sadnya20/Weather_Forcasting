# MUST BE FIRST - before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Pune Weather Forecast Pro", 
    page_icon="⛅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now other imports
from meteostat import Point, Daily, Hourly
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Custom CSS for gradient background and styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #87CEEB 0%, #E0F7FA 100%);
    }
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-text {
        color: #1E88E5;
        font-weight: bold;
    }
    .stSelectbox, .stSlider {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="header-text">⛅WeatherPulse: Advanced Forecasts for Pune</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <p><h4>☁️🌞Predicting Pune’s mood swings. Because even the weather needs a little machine learning therapy.</h4></p>
</div>
""", unsafe_allow_html=True)

# Sidebar with controls
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Forecast Settings")
    forecast_days = st.slider(
        "Select number of days to forecast (1-14):",
        min_value=1, max_value=14, value=7
    )
    show_details = st.checkbox("Show detailed forecast", True)
    
    # File upload for Virtual Crossing data
    st.header("Compare With")
    vc_file = st.file_uploader("Upload Virtual Crossing Data", type="csv")
    
    st.markdown("---")
    st.info("""
     About this application:
    - Powered by Meteostat's rich historical data (2015–present) and advanced XGBoost algorithms, this app delivers reliable weather forecasts with a data-driven edge.
    - Built on robust historical data and cutting-edge machine learning (XGBoost), this app provides weather predictions that combine science, data, and accuracy.
    - From 2015 to today, historical weather trends meet modern AI. Using XGBoost, we turn past patterns into tomorrow’s forecasts.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load and prepare weather data"""
    try:
        location = Point(18.5204, 73.8567)  # Pune
        start = datetime(2015, 1, 1)
        end = datetime.now()
        
        # Daily weather
        daily_data = Daily(location, start, end).fetch().reset_index()
        daily_data.rename(columns={
            'time': 'date_time',
            'tavg': 'temperature',
            'prcp': 'precipitation',
            'wspd': 'wind_speed',
            'pres': 'pressure'
        }, inplace=True)
        daily_data = daily_data[['date_time', 'temperature', 'tmin', 'tmax', 'precipitation', 'wind_speed', 'pressure']]
        daily_data.fillna(daily_data.median(numeric_only=True), inplace=True)
        
        # Hourly humidity
        hourly_data = Hourly(location, start, end).fetch().reset_index()
        hourly_data['date_time'] = hourly_data['time'].dt.date
        humidity = hourly_data.groupby('date_time')['rhum'].mean().reset_index()
        humidity.rename(columns={'rhum': 'humidity'}, inplace=True)
        humidity['date_time'] = pd.to_datetime(humidity['date_time'])
        
        # Merge
        df = pd.merge(daily_data, humidity, on='date_time', how='left')
        
        # Add targets
        df['forecasted_temperature'] = df['temperature'].shift(-1)
        max_precip = df['precipitation'].max()
        df['precipitation_probability'] = df['precipitation'] / (max_precip + 0.001)
        df['forecasted_precip_prob'] = df['precipitation_probability'].shift(-1)
        df['weather_condition'] = np.where(df['precipitation'] > 1.0, 'rainy', 'clear')
        df['forecasted_condition'] = df['weather_condition'].shift(-1)
        
        # Time features
        df['month'] = df['date_time'].dt.month
        df['day_of_year'] = df['date_time'].dt.dayofyear
        df['season'] = df['month'] % 12 // 3 + 1
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        
        # Drop NaNs
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading weather data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def train_models(df):
    """Train and cache ML models"""
    try:
        features = ['temperature', 'tmin', 'tmax', 'wind_speed', 'pressure',
                    'precipitation', 'humidity', 'month', 'day_of_year', 'season', 'is_monsoon']
        
        target_temp = 'forecasted_temperature'
        target_precip = 'forecasted_precip_prob'
        target_cond = 'forecasted_condition'
        
        # Label encode classification target
        le = LabelEncoder()
        df['forecasted_condition_encoded'] = le.fit_transform(df['forecasted_condition'])
        
        # Feature set
        X = df[features]
        
        # Regression - temperature
        y_temp = df[target_temp]
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)
        
        # Regression - precipitation
        y_precip = df[target_precip]
        X_train_precip, X_test_precip, y_train_precip, y_test_precip = train_test_split(X, y_precip, test_size=0.2, random_state=42)
        
        # Classification - condition
        y_cond = df['forecasted_condition_encoded']
        X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_cond, test_size=0.2, random_state=42)
        
        # Oversample classification training set
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train_cond, y_train_cond)
        
        # Train models
        reg_temp = XGBRegressor(n_estimators=100, random_state=42)
        reg_precip = XGBRegressor(n_estimators=100, random_state=42)
        cls_cond = XGBClassifier(n_estimators=100, random_state=42)
        
        reg_temp.fit(X_train_temp, y_train_temp)
        reg_precip.fit(X_train_precip, y_train_precip)
        cls_cond.fit(X_resampled, y_resampled)
        
        return reg_temp, reg_precip, cls_cond, le, X_test_temp, y_test_temp, X_test_precip, y_test_precip, X_test_cond, y_test_cond
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None

# Load data and train models
df = load_data()
if not df.empty:
    reg_temp, reg_precip, cls_cond, le, X_test_temp, y_test_temp, X_test_precip, y_test_precip, X_test_cond, y_test_cond = train_models(df)

    # Extended forecast function
    def generate_extended_forecast(last_row, days=7):
        """Generate forecasts for multiple days"""
        try:
            features = ['temperature', 'tmin', 'tmax', 'wind_speed', 'pressure',
                        'precipitation', 'humidity', 'month', 'day_of_year', 'season', 'is_monsoon']
            
            forecast_data = []
            current_data = last_row.copy()
            max_precip = df['precipitation'].max()
            
            for day in range(days):
                # Prepare features for the forecast day
                forecast_date = datetime.now() + timedelta(days=day)
                current_data['month'] = forecast_date.month
                current_data['day_of_year'] = forecast_date.timetuple().tm_yday
                current_data['is_monsoon'] = current_data['month'] in [6, 7, 8, 9]
                
                input_data = pd.DataFrame([current_data[features]])
                
                # Make predictions
                temp_pred = reg_temp.predict(input_data)[0]
                precip_pred = reg_precip.predict(input_data)[0]
                cond_pred = le.inverse_transform(cls_cond.predict(input_data))[0]
                
                # Store forecast
                forecast_data.append({
                    'Date': forecast_date.strftime('%Y-%m-%d'),
                    'Day': forecast_date.strftime('%A'),
                    'Temperature (°C)': round(temp_pred, 1),
                    'Min Temp (°C)': round(temp_pred - 2, 1),
                    'Max Temp (°C)': round(temp_pred + 2, 1),
                    'Precip Probability (%)': round(precip_pred * 100),
                    'Condition': cond_pred.capitalize(),
                    'Wind Speed (km/h)': round(current_data['wind_speed'], 1)
                })
                
                # Update for next day's forecast
                current_data['temperature'] = temp_pred
                current_data['tmin'] = temp_pred - 2
                current_data['tmax'] = temp_pred + 2
                current_data['precipitation'] = precip_pred * max_precip
            
            return pd.DataFrame(forecast_data)
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return pd.DataFrame()

    # Get the most recent data and generate forecast
    last_row = df.iloc[-1]
    forecast_df = generate_extended_forecast(last_row, days=forecast_days)

    # Current weather display
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Current Weather")
    # You can wrap this in your layout or Streamlit container
    col1, col2, col3, col4 = st.columns(4)
    
    # Set weather icons
    condition = last_row['weather_condition'].lower()
    if 'sun' in condition:
        weather_icon = "☀️"
    elif 'cloud' in condition:
        weather_icon = "☁️"
    elif 'rain' in condition or 'shower' in condition:
        weather_icon = "🌧️"
    elif 'storm' in condition:
        weather_icon = "⛈️"
    elif 'snow' in condition:
        weather_icon = "❄️"
    else:
        weather_icon = "🌡️"
    
    # Blue background and icon styling
    card_style = """
        background-color: #e0f2ff;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    """
    
    with col1:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:40px;">🌡️</div>
            <h3>{last_row['temperature']:.1f}°C</h3>
            <p>Temperature</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:40px;">🌧️</div>
            <h3>{last_row['precipitation']:.1f} mm</h3>
            <p>Precipitation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:40px;">💧</div>
            <h3>{last_row['humidity']:.0f}%</h3>
            <p>Humidity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:40px;">{weather_icon}</div>
            <h3>{last_row['weather_condition'].capitalize()}</h3>
            <p>Condition</p>
        </div>
        """, unsafe_allow_html=True)
    
        
    
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>{last_row['temperature']:.1f}°C</h3>
    #         <p>Temperature</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    # with col2:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>{last_row['precipitation']:.1f} mm</h3>
    #         <p>Precipitation</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    # with col3:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>{last_row['humidity']:.0f}%</h3>
    #         <p>Humidity</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    # with col4:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>{last_row['weather_condition'].capitalize()}</h3>
    #         <p>Condition</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Extended forecast display
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"{forecast_days}-Day Weather Forecast")

    # Interactive Plotly chart
    fig = go.Figure()

    # Add temperature traces
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Temperature (°C)'],
        mode='lines+markers',
        name='Avg Temperature',
        line=dict(color='red', width=2),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}°C<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Min Temp (°C)'],
        fill=None,
        mode='lines',
        name='Min Temperature',
        line=dict(color='blue', width=1, dash='dot'),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}°C<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Max Temp (°C)'],
        fill='tonexty',
        mode='lines',
        name='Temperature Range',
        line=dict(color='blue', width=1, dash='dot'),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}°C<extra></extra>"
    ))

    # Add precipitation probability
    fig.add_trace(go.Bar(
        x=forecast_df['Date'],
        y=forecast_df['Precip Probability (%)'],
        name='Precip Probability',
        marker_color='skyblue',
        opacity=0.6,
        yaxis='y2',
        hovertemplate="<b>%{x}</b><br>%{y}%<extra></extra>"
    ))

    # Update layout
    fig.update_layout(
        title=f"{forecast_days}-Day Weather Forecast",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Temperature (°C)', side='left'),
        yaxis2=dict(title='Precipitation Probability (%)', overlaying='y', side='right', range=[0, 100]),
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', y=1.1),
        plot_bgcolor='rgba(255,255,255,0.8)'
    )

    st.plotly_chart(fig, use_container_width=True)

    if show_details:
        st.dataframe(
            forecast_df.style
            .background_gradient(subset=['Temperature (°C)'], cmap='YlOrRd')
            .background_gradient(subset=['Precip Probability (%)'], cmap='Blues'),
            use_container_width=True
        )
    
    # Virtual Crossing comparison if file uploaded
    if vc_file is not None:
        try:
            vc_data = pd.read_csv(vc_file)
            vc_data.columns = [col.strip() for col in vc_data.columns]
            
            # Standardize date format
            date_col = next((col for col in vc_data.columns if 'date' in col.lower()), None)
            if date_col:
                vc_data['Date'] = pd.to_datetime(vc_data[date_col]).dt.strftime('%Y-%m-%d')
            
            # Find temperature and precipitation columns
            vc_temp_col = next((col for col in vc_data.columns if 'temp' in col.lower()), None)
            vc_precip_col = next((col for col in vc_data.columns if 'precip' in col.lower()), None)
            
            if vc_temp_col and vc_precip_col:
                comparison_df = pd.merge(
                    forecast_df,
                    vc_data[['Date', vc_temp_col, vc_precip_col]],
                    on='Date',
                    how='left'
                ).rename(columns={
                    vc_temp_col: 'VC Temp (°C)',
                    vc_precip_col: 'VC Precip (mm)'
                })
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Comparison with Virtual Crossing")
                
                # Temperature comparison plot
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=comparison_df['Date'],
                    y=comparison_df['Temperature (°C)'],
                    name='Our Forecast',
                    line=dict(color='red')
                ))
                fig_temp.add_trace(go.Scatter(
                    x=comparison_df['Date'],
                    y=comparison_df['VC Temp (°C)'],
                    name='Virtual Crossing',
                    line=dict(color='blue')
                ))
                fig_temp.update_layout(
                    title='Temperature Comparison',
                    xaxis_title='Date',
                    yaxis_title='Temperature (°C)'
                )
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Precipitation comparison plot
                fig_precip = go.Figure()
                fig_precip.add_trace(go.Bar(
                    x=comparison_df['Date'],
                    y=comparison_df['Precip Probability (%)'],
                    name='Our Forecast',
                    marker_color='lightcoral'
                ))
                fig_precip.add_trace(go.Bar(
                    x=comparison_df['Date'],
                    y=comparison_df['VC Precip (mm)'],
                    name='Virtual Crossing',
                    marker_color='lightskyblue'
                ))
                fig_precip.update_layout(
                    title='Precipitation Comparison',
                    xaxis_title='Date',
                    yaxis_title='Precipitation',
                    barmode='group'
                )
                st.plotly_chart(fig_precip, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Could not find temperature/precipitation columns in the uploaded file")
        except Exception as e:
            st.error(f"Error processing Virtual Crossing data: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Model performance
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Performance")
    tab1, tab2, tab3 = st.tabs(["Temperature", "Precipitation", "Weather Condition"])

    with tab1:
        y_pred_temp = reg_temp.predict(X_test_temp)
        rmse_temp = np.sqrt(mean_squared_error(y_test_temp, y_pred_temp))
        st.write(f"Temperature Prediction RMSE: {rmse_temp:.2f} °C")
    
            # Create DataFrame for visualization
        df_temp_comparison = pd.DataFrame({
            "Actual Temperature (°C)": y_test_temp,
            "Predicted Temperature (°C)": y_pred_temp
        })
    
        # Melt the DataFrame to long format for color distinction
        df_melted = df_temp_comparison.reset_index().melt(id_vars='index', 
                            value_vars=["Actual Temperature (°C)", "Predicted Temperature (°C)"],
                            var_name='Type', value_name='Temperature')
    
        # Scatter Plot
        fig = px.scatter(
            df_melted, x="index", y="Temperature", color="Type",
            color_discrete_map={
                "Actual Temperature (°C)": "blue",
                "Predicted Temperature (°C)": "red"
            },
            title="Actual vs Predicted Temperature"
        )
        st.plotly_chart(fig, use_container_width=True)

        
        # fig = px.scatter(
        #     x=y_test_temp, y=y_pred_temp,
        #     labels={'x': 'Actual Temperature (°C)', 'y': 'Predicted Temperature (°C)'},
        #     title="Actual vs Predicted Temperature"
        # )
        # fig.add_shape(type="line", x0=y_test_temp.min(), y0=y_test_temp.min(),
        #             x1=y_test_temp.max(), y1=y_test_temp.max(),
        #             line=dict(color="red", dash="dash"))
        # st.plotly_chart(fig, use_container_width=True)

    with tab2:
        y_pred_precip = reg_precip.predict(X_test_precip)
        rmse_precip = np.sqrt(mean_squared_error(y_test_precip, y_pred_precip))
        st.write(f"Precipitation Probability RMSE: {rmse_precip:.3f}")
        
        # Create a DataFrame for visualization
        df_vis = pd.DataFrame({
            'Actual': y_test_precip,
            'Predicted': y_pred_precip
        })
    
        # Melt the dataframe to long format
        df_melt = df_vis.reset_index().melt(id_vars='index', value_vars=['Actual', 'Predicted'],
                                            var_name='Type', value_name='Probability')
    
        # Scatter plot with color differentiation
        fig = px.scatter(
            df_melt,
            x='index',
            y='Probability',
            color='Type',
            color_discrete_map={'Actual': 'blue', 'Predicted': 'pink'},
            title="Actual vs Predicted Precipitation Probability",
            labels={'index': 'Sample Index', 'Probability': 'Precipitation Probability'}
        )
    
        st.plotly_chart(fig, use_container_width=True)
        
        # y_pred_precip = reg_precip.predict(X_test_precip)
        # rmse_precip = np.sqrt(mean_squared_error(y_test_precip, y_pred_precip))
        # st.write(f"Precipitation Probability RMSE: {rmse_precip:.3f}")
        
        # fig = px.scatter(
        #     x=y_test_precip, y=y_pred_precip,
        #     labels={'x': 'Actual Probability', 'y': 'Predicted Probability'},
        #     title="Actual vs Predicted Precipitation Probability"
        # )
        # fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
        #             line=dict(color="red", dash="dash"))
        # st.plotly_chart(fig, use_container_width=True)

    with tab3:
        y_pred_cond = cls_cond.predict(X_test_cond)
        acc_cond = accuracy_score(y_test_cond, y_pred_cond)
        st.write(f"Weather Condition Accuracy: {acc_cond:.2%}")
        st.write(classification_report(y_test_cond, y_pred_cond, target_names=le.classes_))
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test_cond, y_pred_cond)
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=le.classes_, y=le.classes_,
                        title="Confusion Matrix - Weather Condition")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Historical data visualization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Historical Weather Trends")
    plot_options = ["Temperature", "Precipitation", "Humidity", "Wind Speed"]
    selected_plot = st.selectbox("Choose a metric to visualize", plot_options)

    if selected_plot == "Temperature":
        fig = px.line(df.tail(90), x='date_time', y=['temperature', 'tmin', 'tmax'],
                    labels={'value': 'Temperature (°C)'},
                    title="90-Day Temperature Trend")
    elif selected_plot == "Precipitation":
        fig = px.bar(df.tail(90), x='date_time', y='precipitation',
                    labels={'precipitation': 'Precipitation (mm)'},
                    title="90-Day Precipitation")
    elif selected_plot == "Humidity":
        fig = px.line(df.tail(90), x='date_time', y='humidity',
                    labels={'humidity': 'Humidity (%)'},
                    title="90-Day Humidity Trend")
    elif selected_plot == "Wind Speed":
        fig = px.line(df.tail(90), x='date_time', y='wind_speed',
                    labels={'wind_speed': 'Wind Speed (km/h)'},
                    title="90-Day Wind Speed Trend")

    fig.update_layout(plot_bgcolor='rgba(255,255,255,0.8)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Data download
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Download Data")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Historical Data (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="pune_weather_historical.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="Download Forecast Data (CSV)",
            data=forecast_df.to_csv(index=False).encode('utf-8'),
            file_name=f"pune_{forecast_days}day_forecast.csv",
            mime="text/csv"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="card">
        <p><strong>Note:</strong> Forecasts are generated using machine learning models trained on historical weather data. Actual conditions may differ from predictions.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Failed to load weather data. Please check your internet connection and try again.")