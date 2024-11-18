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
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load data
@st.cache_data
def load_data():
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/RDataset-RMn5ty8YwIlJ107FdfeQk7XWZ8C8Wu.csv"
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Function to create and update vector database
@st.cache_resource
def create_vector_db(data):
    texts = [
        f"On {row['Date']}, historical sales were {row['Historical Sales (PHP)']:.2f} PHP, "
        f"market trend factor was {row['Market Trend Factor']:.2f}, "
        f"user behavior score was {row['User Behavior Score']:.2f}, "
        f"revenue forecast was {row['Revenue Forecast (PHP)']:.2f} PHP, "
        f"and new opportunity score was {row['New Opportunity Score']:.2f}."
        for _, row in data.iterrows()
    ]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, texts

# Function to generate insights using OpenAI with more advanced NLG
def generate_insights(data):
    summary_stats = {
        "avg_historical_sales": data['Historical Sales (PHP)'].mean(),
        "avg_revenue_forecast": data['Revenue Forecast (PHP)'].mean(),
        "avg_market_trend": data['Market Trend Factor'].mean(),
        "avg_user_behavior": data['User Behavior Score'].mean(),
        "avg_new_opportunity": data['New Opportunity Score'].mean(),
        "total_records": len(data),
        "date_range": f"{data['Date'].min().date()} to {data['Date'].max().date()}"
    }
    
    prompt = f"""
    You are an AI financial analyst specializing in revenue forecasting for Philippine startups. Analyze the following financial data summary and provide insights:

    1. Date Range: {summary_stats['date_range']}
    2. Total Records: {summary_stats['total_records']}
    3. Average Historical Sales: {summary_stats['avg_historical_sales']:.2f} PHP
    4. Average Revenue Forecast: {summary_stats['avg_revenue_forecast']:.2f} PHP
    5. Average Market Trend Factor: {summary_stats['avg_market_trend']:.2f}
    6. Average User Behavior Score: {summary_stats['avg_user_behavior']:.2f}
    7. Average New Opportunity Score: {summary_stats['avg_new_opportunity']:.2f}

    Provide a concise summary of the data, including:
    1. Overall trends in historical sales and revenue forecasts
    2. The relationship between market trends and user behavior
    3. Potential new opportunities based on the data
    4. Recommendations for Philippine startups to improve their revenue forecasting and identify growth opportunities

    Format your response in markdown, using headers, bullet points, and emphasis where appropriate.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI financial analyst specializing in revenue forecasting for Philippine startups."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].message['content']

# Function for RAG-based query
def rag_query(query, index, texts):
    query_vector = model.encode([query])
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_vector.astype('float32'), k)
    
    context = "\n".join([texts[i] for i in indices[0]])
    
    prompt = f"""
    You are an AI financial analyst specializing in revenue forecasting for Philippine startups. Use the following context to answer the user's question:

    Context:
    {context}

    User's question: {query}

    Provide a concise and informative answer based on the given context. If the question cannot be answered using the context, say so.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI financial analyst specializing in revenue forecasting for Philippine startups."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].message['content']

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

    # Create vector database
    index, texts = create_vector_db(data)

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
    st.markdown(insights)

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

    # RAG-based query section
    st.subheader("Ask Questions About the Data")
    user_question = st.text_input("Enter your question about the financial data:")
    if user_question:
        answer = rag_query(user_question, index, texts)
        st.markdown(answer)

if __name__ == "__main__":
    main()