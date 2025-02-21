import streamlit as st
import pandas as pd
import os
from langchain_community.llms import Cohere
from langchain_experimental.agents import create_pandas_dataframe_agent
import re

# Set Cohere API Key securely
cohere_api_key = st.secrets["COHERE_API_KEY"]
os.environ["COHERE_API_KEY"] = cohere_api_key

# Streamlit UI
st.title("📊 Excel Data Q&A Chatbot")
st.write("Upload an Excel file and ask questions about the data!")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

#Helper method for Cleaning text cols
def clean_dataframe_text(dataframe):
    """Cleans text columns in a DataFrame by stripping spaces and removing non-alphanumeric characters."""
    text_cols = dataframe.select_dtypes(include='object').columns
    for col in text_cols:
        dataframe[col] = dataframe[col].str.strip()  # Remove extra spaces
        dataframe[col] = dataframe[col].apply(lambda text: re.sub(r'[^\w\s.@%?$/-]', '', str(text)))  # Remove special characters
    return dataframe

# Cache data loading and cleaning
@st.cache_data
def load_and_clean_data(file):
    # Read the Excel file
    df = pd.read_excel(file)
    # Test Case: If the Excel file is completely empty
    if df.empty:
        st.error("Uploaded file is empty.")
        return None  # Return None to trigger the warning

    numeric_cols = df.select_dtypes(include='number').columns
    print(f" Numeric Columns : {numeric_cols}")
    text_cols = df.select_dtypes(include='object').columns
    print(f" Text Columns : {text_cols}")

    # Data Validation
    # 1. Check for Missing Values
    missing_columns = df.columns[df.isnull().any()]
    if not missing_columns.empty:
        st.warning(f"Warning: Missing values found in the following columns: {', '.join(missing_columns)}. These will be dropped.")
        df.dropna(inplace=True)  # Drop rows with NaN values

    # 2. Check for Duplicate Rows
    if df.duplicated().any():
        st.warning("Warning: There are duplicate rows. These will be dropped.")
        df.drop_duplicates(inplace=True)

    # Clean text columns
   # df = clean_dataframe_text(df)
    df.to_excel('cleaned_df.xlsx')  # Save cleaned data for review or download
    return df

# Initialize session state for chat history and Excel data
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process uploaded file
if uploaded_file:
    # Load and clean the data
    df = load_and_clean_data(uploaded_file)

    # If data is invalid or incomplete after validation, return early
    if df is None:
        st.warning("Please resolve the issues in the dataset and upload again.")
    else:
        # Store Excel data in session state
        st.session_state.excel_data = df

        llm = Cohere(model="command-r-plus")

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            #handle_parsing_error=True,
            #handle_parsing_error= "No Answer",
            max_iterations = 10
        )

        # Display cleaned data preview
        st.subheader("📋 DataFrame Preview")
        st.write(df.head())

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user query and agent response
        if prompt := st.chat_input("Ask about the Excel data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Analyzing..."):
                try:
                    # Query the agent for the response
                    response = agent.invoke(prompt)
                    # Extract only the output from the response dictionary
                    response_output = response.get('output', response)  # If 'output' is present, use it, else display the full response
                except Exception as e:
                    if "OUTPUT_PARSING_FAILURE" in str(e):
                        response_output = "No Answer, Query not related to dataframe.. Pls try again."  # Instead of error, return this message in chat
                    else:
                        response_output = f"⚠️ Error: {str(e)}"

            with st.chat_message("assistant"):
                st.markdown(response_output)

            # Store the assistant's response
            st.session_state.messages.append({"role": "assistant", "content": response_output})

else:
    st.warning("Please upload an Excel file to proceed.")
