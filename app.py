import openai   
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from io import StringIO

# Set OpenAI API Key (assumed it's stored as an environment variable or in Streamlit secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Embedded CSV data
csv_data = """
user_id,age,device_type,time_of_day,preferred_language,industry,role,tech_savviness,completion_time_mins,tutorials_watched,help_accessed,features_explored,path_chosen,onboarding_complete,retention_7_day,satisfaction_score
1,24,mobile,evening,English,Technology,Developer,high,12,4,1,8,technical,true,true,9
2,35,desktop,morning,Filipino,Education,Teacher,medium,18,6,3,5,guided,true,true,8
3,42,tablet,afternoon,English,Healthcare,Manager,low,25,8,5,3,basic,true,false,6
4,19,mobile,night,Filipino,Student,Student,high,10,3,0,7,technical,true,true,9
5,28,desktop,morning,English,Finance,Analyst,medium,15,5,2,6,guided,true,true,8
6,31,mobile,evening,Filipino,Retail,Owner,low,22,7,4,4,basic,false,false,5
7,45,desktop,afternoon,English,Technology,Manager,high,14,4,1,9,technical,true,true,9
8,29,mobile,morning,Filipino,Healthcare,Nurse,medium,20,6,3,5,guided,true,false,7
9,33,tablet,night,English,Education,Professor,high,16,5,2,7,technical,true,true,8
10,27,mobile,evening,Filipino,Finance,Accountant,medium,19,5,2,6,guided,true,true,8
11,38,desktop,morning,English,Retail,Manager,low,23,7,4,4,basic,false,false,5
12,22,mobile,afternoon,Filipino,Technology,Developer,high,11,4,1,8,technical,true,true,9
13,36,desktop,night,English,Healthcare,Doctor,medium,17,5,2,6,guided,true,true,8
14,41,tablet,evening,Filipino,Education,Teacher,low,24,8,5,3,basic,true,false,6
15,25,mobile,morning,English,Finance,Analyst,high,13,4,1,8,technical,true,true,9
16,30,desktop,afternoon,Filipino,Retail,Supervisor,medium,18,6,3,5,guided,true,true,8
17,34,mobile,night,English,Technology,Designer,high,15,4,1,7,technical,true,true,9
18,43,desktop,evening,Filipino,Healthcare,Administrator,low,21,7,4,4,basic,false,false,5
19,26,mobile,morning,English,Education,Researcher,high,12,4,1,8,technical,true,true,9
20,32,tablet,afternoon,Filipino,Finance,Manager,medium,19,5,2,6,guided,true,true,8
21,39,mobile,night,English,Retail,Owner,low,23,7,4,4,basic,false,false,5
22,23,desktop,evening,Filipino,Technology,Developer,high,11,4,1,8,technical,true,true,9
23,37,mobile,morning,English,Healthcare,Nurse,medium,18,6,3,5,guided,true,true,8
24,44,desktop,afternoon,Filipino,Education,Professor,low,22,7,4,4,basic,true,false,6
25,28,tablet,night,English,Finance,Analyst,high,14,4,1,7,technical,true,true,9
26,31,mobile,evening,Filipino,Retail,Manager,medium,19,5,2,6,guided,true,true,8
27,35,desktop,morning,English,Technology,Designer,high,13,4,1,8,technical,true,true,9
28,40,mobile,afternoon,Filipino,Healthcare,Doctor,low,24,8,5,3,basic,false,false,5
29,24,desktop,night,English,Education,Teacher,high,12,4,1,8,technical,true,true,9
30,33,tablet,evening,Filipino,Finance,Accountant,medium,17,5,2,6,guided,true,true,8
31,29,mobile,morning,English,Retail,Supervisor,low,21,7,4,4,basic,true,false,6
32,36,desktop,afternoon,Filipino,Technology,Developer,high,11,4,1,8,technical,true,true,9
33,42,mobile,night,English,Healthcare,Manager,medium,18,6,3,5,guided,true,true,8
34,27,desktop,evening,Filipino,Education,Researcher,high,13,4,1,7,technical,true,true,9
35,34,tablet,morning,English,Finance,Analyst,low,22,7,4,4,basic,false,false,5
36,38,mobile,afternoon,Filipino,Retail,Owner,high,12,4,1,8,technical,true,true,9
37,25,desktop,night,English,Technology,Designer,medium,19,5,2,6,guided,true,true,8
38,32,mobile,evening,Filipino,Healthcare,Nurse,low,23,7,4,4,basic,true,false,6
39,41,desktop,morning,English,Education,Professor,high,14,4,1,7,technical,true,true,9
40,30,tablet,afternoon,Filipino,Finance,Manager,medium,18,6,3,5,guided,true,true,8
41,37,mobile,night,English,Retail,Supervisor,high,13,4,1,8,technical,true,true,9
42,43,desktop,evening,Filipino,Technology,Developer,low,24,8,5,3,basic,false,false,5
43,26,mobile,morning,English,Healthcare,Doctor,high,12,4,1,8,technical,true,true,9
44,35,desktop,afternoon,Filipino,Education,Teacher,medium,17,5,2,6,guided,true,true,8
45,39,tablet,night,English,Finance,Analyst,low,21,7,4,4,basic,true,false,6
46,28,mobile,evening,Filipino,Retail,Manager,high,11,4,1,8,technical,true,true,9
47,33,desktop,morning,English,Technology,Designer,medium,18,6,3,5,guided,true,true,8
48,44,mobile,afternoon,Filipino,Healthcare,Administrator,high,13,4,1,7,technical,true,true,9
49,31,desktop,night,English,Education,Researcher,low,22,7,4,4,basic,false,false,5
50,36,tablet,evening,Filipino,Finance,Accountant,high,12,4,1,8,technical,true,true,9
51,40,mobile,morning,English,Retail,Owner,medium,19,5,2,6,guided,true,true,8
52,27,desktop,afternoon,Filipino,Technology,Developer,low,23,7,4,4,basic,true,false,6
53,34,mobile,night,English,Healthcare,Nurse,high,14,4,1,7,technical,true,true,9
54,38,desktop,evening,Filipino,Education,Professor,medium,18,6,3,5,guided,true,true,8
55,42,tablet,morning,English,Finance,Analyst,high,13,4,1,8,technical,true,true,9
56,29,mobile,afternoon,Filipino,Retail,Supervisor,low,24,8,5,3,basic,false,false,5
57,35,desktop,night,English,Technology,Designer,high,12,4,1,8,technical,true,true,9
58,41,mobile,evening,Filipino,Healthcare,Doctor,medium,17,5,2,6,guided,true,true,8
59,32,desktop,morning,English,Education,Teacher,low,21,7,4,4,basic,true,false,6
60,37,tablet,afternoon,Filipino,Finance,Manager,high,11,4,1,8,technical,true,true,9
61,43,mobile,night,English,Retail,Owner,medium,18,6,3,5,guided,true,true,8
62,30,desktop,evening,Filipino,Technology,Developer,high,13,4,1,7,technical,true,true,9
63,36,mobile,morning,English,Healthcare,Administrator,low,22,7,4,4,basic,false,false,5
64,40,desktop,afternoon,Filipino,Education,Researcher,high,12,4,1,8,technical,true,true,9
65,28,tablet,night,English,Finance,Analyst,medium,19,5,2,6,guided,true,true,8
66,33,mobile,evening,Filipino,Retail,Manager,low,23,7,4,4,basic,true,false,6
67,39,desktop,morning,English,Technology,Designer,high,14,4,1,7,technical,true,true,9
68,44,mobile,afternoon,Filipino,Healthcare,Nurse,medium,18,6,3,5,guided,true,true,8
69,31,desktop,night,English,Education,Professor,high,13,4,1,8,technical,true,true,9
70,35,tablet,evening,Filipino,Finance,Accountant,low,24,8,5,3,basic,false,false,5
71,41,mobile,morning,English,Retail,Supervisor,high,12,4,1,8,technical,true,true,9
72,27,desktop,afternoon,Filipino,Technology,Developer,medium,17,5,2,6,guided,true,true,8
73,34,mobile,night,English,Healthcare,Doctor,low,21,7,4,4,basic,true,false,6
74,38,desktop,evening,Filipino,Education,Teacher,high,11,4,1,8,technical,true,true,9
75,42,tablet,morning,English,Finance,Analyst,medium,18,6,3,5,guided,true,true,8
76,29,mobile,afternoon,Filipino,Retail,Owner,high,13,4,1,7,technical,true,true,9
77,36,desktop,night,English,Technology,Designer,low,22,7,4,4,basic,false,false,5
78,40,mobile,evening,Filipino,Healthcare,Administrator,high,12,4,1,8,technical,true,true,9
79,32,desktop,morning,English,Education,Researcher,medium,19,5,2,6,guided,true,true,8
80,37,tablet,afternoon,Filipino,Finance,Manager,low,23,7,4,4,basic,true,false,6
81,43,mobile,night,English,Retail,Supervisor,high,14,4,1,7,technical,true,true,9
82,30,desktop,evening,Filipino,Technology,Developer,medium,18,6,3,5,guided,true,true,8
83,35,mobile,morning,English,Healthcare,Nurse,high,13,4,1,8,technical,true,true,9
84,39,desktop,afternoon,Filipino,Education,Professor,low,24,8,5,3,basic,false,false,5
85,28,tablet,night,English,Finance,Analyst,high,12,4,1,8,technical,true,true,9
86,33,mobile,evening,Filipino,Retail,Manager,medium,17,5,2,6,guided,true,true,8
87,41,desktop,morning,English,Technology,Designer,low,21,7,4,4,basic,true,false,6
88,34,mobile,afternoon,Filipino,Healthcare,Doctor,high,11,4,1,8,technical,true,true,9
89,38,desktop,night,English,Education,Teacher,medium,18,6,3,5,guided,true,true,8
90,42,tablet,evening,Filipino,Finance,Accountant,high,13,4,1,7,technical,true,true,9
"""

# Function to retrieve relevant data from the dataset
def retrieve_relevant_data(data, query):
    """
    Retrieve relevant data from the dataset based on the user's query.
    """
    if "satisfaction" in query.lower():
        return data[['age', 'industry', 'satisfaction_score']]
    elif "retention" in query.lower():
        return data[['age', 'industry', 'retention_7_day']]
    elif "behavior" in query.lower():
        return data[['age', 'device_type', 'time_of_day', 'satisfaction_score']]
    else:
        return data  # Return the full dataset for general queries

# Function to call OpenAI's API for text generation
def generate_nlg_response(prompt, data):
    """
    Generate text using OpenAI's GPT model for NLG.
    """
    try:
        # Prepare a summary of the data
        data_summary = data.describe().to_string()
        data_head = data.head().to_string()
        data_tail = data.tail().to_string()
        
        full_prompt = f"""Analyze the following dataset:

Summary Statistics:
{data_summary}

First few rows:
{data_head}

Last few rows:
{data_tail}

Now, based on this data, {prompt}

Provide a detailed analysis, including exact counts and percentages where applicable."""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Using a model with larger context
            messages=[
                {"role": "system", "content": "You are an AI assistant analyzing user behavior data. Provide accurate statistics and insights based on the full dataset."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error in generating NLG response: {str(e)}")
        return "Sorry, I couldn't generate a response at this time."

# Set page config
st.set_page_config(page_title="Baldwin Predictions AI", layout="wide")

# Apply glass neumorphic design CSS
st.markdown("""
<style>
    .main {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 20px 20px 40px -6px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .stButton>button {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.3);
    }
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: none;
        border-radius: 5px;
    }
    .scroll-table {
        max-height: 300px;
        overflow-y: scroll;
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .scroll-table::-webkit-scrollbar {
        width: 12px;
    }
    .scroll-table::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    .scroll-table::-webkit-scrollbar-thumb {
        background-color: rgba(255, 255, 255, 0.3);
        border-radius: 6px;
        border: 3px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title of the app
st.title("Baldwin Predictions AI - User Behavior Insights with NLG & RAG")

# Load the CSV data
data = pd.read_csv(io.StringIO(csv_data))

# Sidebar - Ask AI feature
st.sidebar.title("Ask AI")
query = st.sidebar.text_area("Ask a question about user behavior or satisfaction:")
if st.sidebar.button("Get AI Insights"):
    if query:
        retrieved_data = retrieve_relevant_data(data, query)
        nlg_response = generate_nlg_response(query, retrieved_data)
        st.sidebar.write("AI Response:", nlg_response)

# Main content
st.subheader("User Behavior Data")
st.markdown('<div class="scroll-table">', unsafe_allow_html=True)
st.dataframe(data.style.set_properties(**{'background-color': 'rgba(255, 255, 255, 0.1)', 'color': 'white'}))
st.markdown('</div>', unsafe_allow_html=True)

# Option for the user to select the analysis type
analysis_type = st.selectbox("Select Analysis", ("User Satisfaction Prediction", "Retention Forecast", "User Behavior Insights"))

# User Satisfaction Prediction
if analysis_type == "User Satisfaction Prediction":
    st.subheader("Predicting User Satisfaction Based on User Data")
    age = st.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (20, 40))
    industry = st.selectbox("Select Industry", data["industry"].unique())

    filtered_data = data[(data["age"] >= age[0]) & (data["age"] <= age[1])]
    filtered_data = filtered_data[filtered_data["industry"] == industry]
    
    retrieved_data = retrieve_relevant_data(filtered_data, "satisfaction")
    prompt = f"Predict the user satisfaction score for users in the {industry} industry, aged between {age[0]} and {age[1]}. Provide a detailed analysis."

    nlg_response = generate_nlg_response(prompt, retrieved_data)
    st.write("Prediction of user satisfaction:", nlg_response)

# Retention Forecast
elif analysis_type == "Retention Forecast":
    st.subheader("Forecasting Retention Rate (7-day)")
    age = st.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (20,40))
    industry = st.selectbox("Select Industry", data["industry"].unique())
    
    filtered_data = data[(data["age"] >= age[0]) & (data["age"] <= age[1])]
    filtered_data = filtered_data[filtered_data["industry"] == industry]
    
    retrieved_data = retrieve_relevant_data(filtered_data, "retention")
    prompt = f"Forecast the retention rate for users in the {industry} industry, aged between {age[0]} and {age[1]}. Provide a detailed analysis."

    nlg_response = generate_nlg_response(prompt, retrieved_data)
    st.write("Forecasted Retention Rate (7 days):", nlg_response)

# User Behavior Insights
elif analysis_type == "User Behavior Insights":
    st.subheader("Analyzing User Behavior Based on Time of Day")
    time_of_day = st.selectbox("Select Time of Day", data["time_of_day"].unique())
    device_type = st.selectbox("Select Device Type", data["device_type"].unique())

    filtered_data = data[(data["time_of_day"] == time_of_day) & (data["device_type"] == device_type)]
    
    mean_satisfaction = filtered_data["satisfaction_score"].mean()
    st.write(f"Average Satisfaction Score for {time_of_day} and {device_type} users: {mean_satisfaction:.2f}")

    retrieved_data = retrieve_relevant_data(filtered_data, "behavior")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(filtered_data["satisfaction_score"], bins=10, color='skyblue', edgecolor='black')
    ax.set_title(f"Satisfaction Scores for {time_of_day} Users on {device_type}")
    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Frequency")
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)

    prompt = f"Analyze the user behavior for {time_of_day} users on {device_type} devices. Provide insights on satisfaction scores and any notable patterns."
    nlg_response = generate_nlg_response(prompt, retrieved_data)
    st.write("User Behavior Analysis:", nlg_response)