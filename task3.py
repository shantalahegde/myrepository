import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import schedule
import time
from twilio.rest import Client
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('OPENWEATHER_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
RECIPIENT_NUMBERS = os.getenv('RECIPIENT_NUMBERS').split(',')  # Comma-separated list
CITY = "New York"
COUNTRY_CODE = "US"
MODEL_FILE = "heatwave_model.pkl"
DATA_FILE = "weather_data.csv"
THRESHOLDS = {
    'heat_wave': 35,  # Temperature in °C
    'high_risk': 40,  # Temperature in °C
    'humidity_threshold': 70  # Percentage
}

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

class WeatherDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        
    def get_current_weather(self, city, country_code):
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_forecast(self, city, country_code):
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric',
            'cnt': 40  # 5-day forecast with 3-hour intervals
        }
        response = requests.get(self.forecast_url, params=params)
        return response.json()
    
    def process_weather_data(self, data):
        processed = {
            'timestamp': datetime.fromtimestamp(data['dt']),
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'clouds': data['clouds']['all'],
            'weather_main': data['weather'][0]['main'],
            'weather_desc': data['weather'][0]['description']
        }
        return processed
    
    def save_to_csv(self, data, filename):
        df = pd.DataFrame([data])
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(filename, index=False)

class HeatWavePredictor:
    def __init__(self, model_file=MODEL_FILE):
        self.model_file = model_file
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
        else:
            self.train_model()
    
    def train_model(self):
        # This would ideally use historical weather data
        # For demo purposes, we'll create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        temps = np.random.normal(30, 10, n_samples)
        humidities = np.random.normal(60, 20, n_samples)
        wind_speeds = np.random.normal(15, 5, n_samples)
        
        # Generate labels (1 = heat wave, 0 = no heat wave)
        labels = ((temps > 35) & (humidities > 70)).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'temperature': temps,
            'humidity': humidities,
            'wind_speed': wind_speeds,
            'heat_wave': labels
        })
        
        # Split data
        X = data[['temperature', 'humidity', 'wind_speed']]
        y = data['heat_wave']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Save model
        joblib.dump(self.model, self.model_file)
    
    def predict_heat_wave(self, weather_data):
        features = pd.DataFrame([{
            'temperature': weather_data['temp'],
            'humidity': weather_data['humidity'],
            'wind_speed': weather_data['wind_speed']
        }])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]
        return prediction, probability

class AlertSystem:
    def __init__(self):
        self.predictor = HeatWavePredictor()
        self.collector = WeatherDataCollector(API_KEY)
        self.alert_history = []
    
    def check_conditions(self, weather_data):
        """Check weather conditions against thresholds"""
        temp = weather_data['temp']
        humidity = weather_data['humidity']
        
        alerts = []
        
        if temp >= THRESHOLDS['high_risk']:
            alerts.append(('Extreme Heat Warning', f"Dangerous heat: {temp}°C"))
        elif temp >= THRESHOLDS['heat_wave']:
            alerts.append(('Heat Wave Alert', f"Heat wave conditions: {temp}°C"))
        
        if humidity >= THRESHOLDS['humidity_threshold'] and temp > 30:
            alerts.append(('High Humidity Warning', f"Humid conditions: {humidity}% at {temp}°C"))
        
        # Add ML prediction
        prediction, probability = self.predictor.predict_heat_wave(weather_data)
        if prediction == 1:
            alerts.append(('ML Heat Wave Prediction', 
                         f"High probability ({probability:.0%}) of heat wave conditions"))
        
        return alerts
    
    def send_sms_alerts(self, alerts):
        for recipient in RECIPIENT_NUMBERS:
            for alert_type, message in alerts:
                full_message = f"{alert_type}: {message} in {CITY}. Take precautions."
                try:
                    twilio_client.messages.create(
                        body=full_message,
                        from_=TWILIO_PHONE_NUMBER,
                        to=recipient.strip()
                    )
                    print(f"Alert sent to {recipient}: {full_message}")
                    self.alert_history.append({
                        'timestamp': datetime.now(),
                        'recipient': recipient,
                        'alert_type': alert_type,
                        'message': message
                    })
                except Exception as e:
                    print(f"Failed to send alert to {recipient}: {str(e)}")
    
    def generate_dashboard(self):
        """Generate a simple text-based dashboard"""
        if not os.path.exists(DATA_FILE):
            return "No data available yet."
        
        df = pd.read_csv(DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get latest data
        latest = df.iloc[-1]
        
        # Get forecast
        forecast = self.collector.get_forecast(CITY, COUNTRY_CODE)
        forecast_data = []
        for item in forecast['list']:
            forecast_data.append(self.collector.process_weather_data(item))
        
        # Create dashboard text
        dashboard = [
            f"Weather Dashboard for {CITY}, {COUNTRY_CODE}",
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\nCurrent Conditions:",
            f"Temperature: {latest['temp']}°C (Feels like {latest['feels_like']}°C)",
            f"Humidity: {latest['humidity']}%",
            f"Wind Speed: {latest['wind_speed']} m/s",
            f"Weather: {latest['weather_main']} ({latest['weather_desc']})",
            "\nForecast for next 24 hours:"
        ]
        
        for i, item in enumerate(forecast_data[:8]):  # Next 24 hours (8 * 3-hour intervals)
            dashboard.append(
                f"{item['timestamp'].strftime('%H:%M')}: {item['temp']}°C, "
                f"{item['humidity']}% humidity, {item['weather_desc']}"
            )
        
        # Add alert status
        alerts = self.check_conditions(latest)
        dashboard.append("\nAlert Status:")
        if alerts:
            for alert_type, message in alerts:
                dashboard.append(f"⚠️ {alert_type}: {message}")
        else:
            dashboard.append("✅ No active alerts")
        
        return "\n".join(dashboard)
    
    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        print(f"\nRunning monitoring cycle at {datetime.now()}")
        
        # Get current weather
        current_weather = self.collector.get_current_weather(CITY, COUNTRY_CODE)
        processed_data = self.collector.process_weather_data(current_weather)
        self.collector.save_to_csv(processed_data, DATA_FILE)
        
        # Check conditions
        alerts = self.check_conditions(processed_data)
        
        # Send alerts if needed
        if alerts:
            self.send_sms_alerts(alerts)
        
        # Generate dashboard
        dashboard = self.generate_dashboard()
        print(dashboard)
        
        return alerts

def main():
    alert_system = AlertSystem()
    
    # Run immediately
    alert_system.run_monitoring_cycle()
    
    # Schedule to run every hour
    schedule.every().hour.do(alert_system.run_monitoring_cycle)
    
    print("Heat Wave Alert System is running. Monitoring every hour...")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
