import openai
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set OpenAI API Key (assumed it's stored as an environment variable or in Streamlit secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]



#st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to retrieve relevant data from the dataset
def retrieve_relevant_data(data, query):
    """
    Retrieve relevant data from the dataset based on the user's query.
    This could be based on keywords or user-selected filters.
    """
    # Simple keyword-based retrieval (can be extended to more complex logic)
    if "satisfaction" in query.lower():
        return data[['age', 'industry', 'satisfaction_score']].dropna()
    elif "retention" in query.lower():
        return data[['age', 'industry', 'retention_7_day']].dropna()
    elif "behavior" in query.lower():
        return data[['age', 'device_type', 'time_of_day', 'satisfaction_score']].dropna()
    else:
        # Default: return all data if no specific query is found
        return data.head()

# Function to call OpenAI's API for text generation
def generate_nlg_response(prompt):
    """
    Generate text using OpenAI's GPT model for NLG.
    """
    response = openai.Completion.create(
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        prompt=prompt
    )
    return response.choices[0].text.strip()

# Title of the app
st.title("Baldwin Predictions AI - User Behavior Insights with NLG & RAG")

# File uploader for the dataset
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Check if file is uploaded and read it
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!", data.head())

    # Display the columns of the dataset to confirm proper loading
    st.write("Columns in the dataset:", data.columns)

    # Option for the user to select the analysis type
    analysis_type = st.sidebar.selectbox("Select Analysis", ("User Satisfaction Prediction", "Retention Forecast", "User Behavior Insights", "Ask AI"))

    # User Satisfaction Prediction
    if analysis_type == "User Satisfaction Prediction":
        st.subheader("Predicting User Satisfaction Based on User Data")
        # Select columns for analysis
        age = st.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (20, 40))
        industry = st.selectbox("Select Industry", data["industry"].unique())

        filtered_data = data[(data["age"] >= age[0]) & (data["age"] <= age[1])]
        filtered_data = filtered_data[filtered_data["industry"] == industry]
        
        # Fetch relevant data for RAG
        retrieved_data = retrieve_relevant_data(filtered_data, "satisfaction")
        prompt = f"Based on the data of user behavior, predict the user satisfaction score for users in the {industry} industry, aged between {age[0]} and {age[1]}."

        # Use OpenAI API for NLG
        nlg_response = generate_nlg_response(prompt)
        st.write("Prediction of user satisfaction:", nlg_response)

    # Retention Forecast
    elif analysis_type == "Retention Forecast":
        st.subheader("Forecasting Retention Rate (7-day)")
        
        # Select user characteristics for filtering
        age = st.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (20, 40))
        industry = st.selectbox("Select Industry", data["industry"].unique())
        
        # Filter data based on user selection
        filtered_data = data[(data["age"] >= age[0]) & (data["age"] <= age[1])]
        filtered_data = filtered_data[filtered_data["industry"] == industry]
        
        # Fetch relevant data for RAG
        retrieved_data = retrieve_relevant_data(filtered_data, "retention")
        prompt = f"Based on the data of user retention over the first 7 days, forecast the retention rate for users in the {industry} industry, aged between {age[0]} and {age[1]}."

        # Use OpenAI API for NLG
        nlg_response = generate_nlg_response(prompt)
        st.write("Forecasted Retention Rate (7 days):", nlg_response)

    # User Behavior Insights
    elif analysis_type == "User Behavior Insights":
        st.subheader("Analyzing User Behavior Based on Time of Day")
        
        # Select variables for behavior analysis
        time_of_day = st.selectbox("Select Time of Day", data["time_of_day"].unique())
        device_type = st.selectbox("Select Device Type", data["device_type"].unique())

        filtered_data = data[(data["time_of_day"] == time_of_day) & (data["device_type"] == device_type)]
        
        # Calculate mean satisfaction score for the filtered data
        mean_satisfaction = filtered_data["satisfaction_score"].mean()
        st.write(f"Average Satisfaction Score for {time_of_day} and {device_type} users: {mean_satisfaction}")

        # Fetch relevant data for RAG
        retrieved_data = retrieve_relevant_data(filtered_data, "behavior")
        
        # Plot the distribution of satisfaction scores
        plt.figure(figsize=(8, 6))
        plt.hist(filtered_data["satisfaction_score"], bins=10, color='skyblue', edgecolor='black')
        plt.title(f"Satisfaction Scores for {time_of_day} Users on {device_type}")
        plt.xlabel("Satisfaction Score")
        plt.ylabel("Frequency")
        st.pyplot()

    # "Ask AI" feature (user query-based)
    elif analysis_type == "Ask AI":
        st.subheader("Ask AI for Insights on User Data")
        query = st.text_area("Ask a question about user behavior or satisfaction:")
        
        if query:
            # Fetch relevant data for the user's question
            retrieved_data = retrieve_relevant_data(data, query)
            
            # Use OpenAI API for NLG to respond to the user's query
            prompt = f"Answer the following question based on this data: {query}\n\n{retrieved_data.to_string(index=False)}"
            nlg_response = generate_nlg_response(prompt)
            st.write("AI Response:", nlg_response)

# If no file is uploaded, show a message
else:
    st.write("Please upload a CSV file to get started.")
