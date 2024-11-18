import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
import io
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Set up OpenAI API key
#openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to load data
@st.cache_data
def load_data():
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/RDataset-RMn5ty8YwIlJ107FdfeQk7XWZ8C8Wu.csv"
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Function to generate insights using OpenAI
def generate_insights(data):
    prompt = f"""
    Analyze the following financial data and provide insights:
    
    1. Average Historical Sales: {data['Historical Sales (PHP)'].mean():.2f}
    2. Average Revenue Forecast: {data['Revenue Forecast (PHP)'].mean():.2f}
    3. Average Market Trend Factor: {data['Market Trend Factor'].mean():.2f}
    4. Average User Behavior Score: {data['User Behavior Score'].mean():.2f}
    5. Average New Opportunity Score: {data['New Opportunity Score'].mean():.2f}
    
    Provide a concise summary of the data, including potential trends, opportunities, and recommendations for the business.
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()

# Function to train and evaluate the model
def train_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return model, mae, rmse

# Main Streamlit app
def main():
    st.title("Revenue Forecasting AI for Philippine Startups")
    st.write("Analyze historical sales data, market trends, and user behavior to provide accurate revenue forecasts and identify new monetization opportunities.")

    # Load data
    data = load_data()

    # Sidebar for date range selection
    st.sidebar.header("Date Range Selection")
    start_date = st.sidebar.date_input("Start Date", min(data['Date']))
    end_date = st.sidebar.date_input("End Date", max(data['Date']))

    # Filter data based on selected date range
    filtered_data = data[(data['Date'] >= pd.Timestamp(start_date)) & (data['Date'] <= pd.Timestamp(end_date))]

    # Display raw data
    st.subheader("Raw Data")
    st.write(filtered_data)

    # Visualizations
    st.subheader("Data Visualizations")

    # Historical Sales vs Revenue Forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_data['Date'], filtered_data['Historical Sales (PHP)'], label='Historical Sales')
    ax.plot(filtered_data['Date'], filtered_data['Revenue Forecast (PHP)'], label='Revenue Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount (PHP)')
    ax.set_title('Historical Sales vs Revenue Forecast')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Market Trend Factor vs User Behavior Score
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=filtered_data, x='Market Trend Factor', y='User Behavior Score', hue='New Opportunity Score', palette='viridis', ax=ax)
    ax.set_title('Market Trend Factor vs User Behavior Score')
    st.pyplot(fig)

    # Generate insights using OpenAI
    st.subheader("AI-Generated Insights")
    insights = generate_insights(filtered_data)
    st.write(insights)

    # Train and evaluate the model
    st.subheader("Revenue Forecast Model")
    X = filtered_data[['Market Trend Factor', 'User Behavior Score', 'New Opportunity Score']]
    y = filtered_data['Revenue Forecast (PHP)']
    model, mae, rmse = train_evaluate_model(X, y)

    st.write(f"Model Performance:")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")

    # Allow users to input values for prediction
    st.subheader("Revenue Forecast Prediction")
    market_trend = st.slider("Market Trend Factor", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    user_behavior = st.slider("User Behavior Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    new_opportunity = st.slider("New Opportunity Score", min_value=0.0, max_value=5.0, value=2.5, step=0.1)

    # Make prediction
    input_data = np.array([[market_trend, user_behavior, new_opportunity]])
    prediction = model.predict(input_data)[0]

    st.write(f"Predicted Revenue Forecast: PHP {prediction:.2f}")

if __name__ == "__main__":
    main()