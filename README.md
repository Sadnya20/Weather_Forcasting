# 🌤️ Weather Forecasting Project
![Cover Page](https://github.com/user-attachments/assets/ce29e5f7-6560-447c-9132-df28701f2bee)

## 📌 Key Features

- ✅ **Exploratory Data Analysis (EDA):** Insightful analysis of historical weather data to uncover trends and patterns.
- 🔁 **Scheduling Mechanism:** Automatically updates forecasts on a daily basis using a Python scheduling mechanism.
- 🤖 **Machine Learning Model:** Implements a Random Forest Regressor and XG Boost to predict weather metrics like temperature and rainfall.
- 🌐 **Streamlit Web App:** Provides an interactive frontend for users to view the future 7-day weather forecast.
- 📊 **Power BI Dashboard:** Combines historical and forecasted data to deliver actionable insights via a visually compelling dashboard.

## ⚙️ Technologies Used

- **Languages & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Machine Learning:** Random Forest Regressor, XG Boost
- **Scheduling:** Python `schedule`
- **Frontend:** Streamlit
- **Visualization:** Power BI Desktop

## 📈 Power BI Dashboard

The Power BI `.pbix` dashboard includes:

- Historical weather trends (temperature, precipitation)
- Forecasted values for the upcoming 7 days
- Comparative visuals between actual and predicted metrics
- Filters for date ranges and weather categories

> 📌 Open the `weather forecasting.pbix` file in **Power BI Desktop** to explore the visuals.

## DashBoard Visualisations
![page-1](https://github.com/user-attachments/assets/6968a0d0-4170-458a-b36e-a7c33f87d15c)
![page-2](https://github.com/user-attachments/assets/5c23cb7d-cca9-49b1-9def-0d7e59f26530)

## Deploymeny Link: [Streamlit](https://hackthonweatherforcast-6zobobkd2vee42ljb2wkpt.streamlit.app/)

## 🔮 Forecasting Model

- **Algorithm:** Random Forest Regressor
- **Input Features:** Date, temperature, humidity, rainfall, etc.
- **Predicted Outputs:** Temperature, Rainfall, Weather Conditions
- **Evaluation Metrics:** MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
- **Data Source:** Historical weather data (Pune)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/weather-forecasting.git
cd weather-forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web App

```bash
cd streamlit_app
streamlit run app.py
```

### 4. Run the Scheduler Script

```bash
python scheduler.py
```

### 📅 Scheduling
The forecasting pipeline is automated to:

- Fetch new data regularly (daily/hourly)

- Run the trained model on the new data

- Save predictions for dashboard and app updates

- This is handled using schedule  within the notebook.

### 🌐 Deployment
- To deploy the Streamlit app online, you can use:

- Streamlit Community Cloud

- Heroku / Render

- Docker container on AWS, GCP, or Azure

### 🔧 Future Improvements
1. 🔄 Integrate real-time weather data using APIs

2. 📦 Containerize the app using Docker

3. 📈 Add advanced models like LightGBM, or LSTM

4. ☁️ Deploy on cloud with automatic updates and monitoring

5. 📧 Add email notifications for daily forecast delivery

### 🙌 Acknowledgments
1. Inspired by real-world weather analysis and forecasting use cases.

2. Thanks to open-source contributors and data providers.

## 📜 **License**

This project is open-source. Feel free to modify and enhance it based on your requirements.

## **Team Members**👤🤝👥
**[Sadnya Kolhe](https://github.com/Sadnya20)** 

- Developed the Streamlit application

- Deployed the web app

- Performed Exploratory Data Analysis (EDA)

- Built and trained the Random Forest model

- Extracted weather data using APIs

**[Prince Srivastava](https://github.com/PrinceSrivastava182)**

- Designed and developed the Power BI dashboard

- Implemented the XGBoost model for forecasting

- Built the time-based scheduling mechanism in Python



**☀️ Built with passion for data, forecasting, and creating smarter weather solutions.**
